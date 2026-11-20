"""FastGRFNet: GRF-Net with fast weights in its spatial front end."""

from __future__ import annotations

import torch
from torch import nn

from ..grfnet.model import GRFNet

WRITE_MODES = ("off", "unsupervised", "label_feedback")


class FastWeightWriter(nn.Module):
    """Small network mapping a trial summary to a delta-rule write: key ``k``, value ``v`` and rate ``eta``.

    The value projection starts at zero, so the first writes leave the fast weights at zero.
    """

    def __init__(self, n_inputs: int, n_channels: int, spatial_components: int, hidden: int, eta_max: float) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(n_inputs, hidden), nn.Tanh())
        self.key = nn.Linear(hidden, n_channels)
        self.value = nn.Linear(hidden, spatial_components)
        self.rate = nn.Linear(hidden, 1)
        self.eta_max = eta_max
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    def forward(self, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.hidden(summary)
        k = nn.functional.normalize(self.key(h), dim=-1)
        v = self.value(h)
        eta = self.eta_max * torch.sigmoid(self.rate(h)).squeeze(-1)
        return k, v, eta


class FastGRFNet(GRFNet):
    """GRF-Net whose spatial filters adapt within a stream of trials through fast weights.

    The spatial filters are ``W_slow + ΔW``. ``ΔW`` is the state: zero at the start of each
    subject's sequence and updated after every trial by the delta rule
    ``ΔW <- λ ΔW + η (v - ΔW k) kᵀ``, with ``k``, ``v`` and ``η`` from a writer network. The
    Frobenius norm of ``ΔW`` is clipped to ``delta_clip``. All other arguments are those of
    :class:`GRFNet`.

    Parameters
    ----------
    write_mode : {"off", "unsupervised", "label_feedback"}
        ``"off"`` never writes and reproduces GRFNet. ``"unsupervised"`` writes from a summary of the
        trial. ``"label_feedback"`` also gives the writer the trial's label, after prediction.
    decay : float
        Fast-weight decay ``λ`` per trial, in (0, 1).
    eta_max : float
        Largest write rate ``η``.
    delta_clip : float
        Largest Frobenius norm of ``ΔW`` per sequence.
    writer_hidden : int
        Hidden units of the writer network.

    References
    ----------
    Schlag, Irie and Schmidhuber (2021). Linear transformers are secretly fast weight programmers. ICML.
    """

    def __init__(self, n_channels: int, n_times: int | None = None, n_classes: int | None = None, *,
                 write_mode: str = "unsupervised", decay: float = 0.95, eta_max: float = 0.5,
                 delta_clip: float = 1.0, writer_hidden: int = 32, **kwargs) -> None:
        super().__init__(n_channels, n_times, n_classes, **kwargs)
        if write_mode not in WRITE_MODES:
            raise ValueError(f"write_mode must be one of {WRITE_MODES}; got {write_mode!r}")
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1); got {decay}")
        if eta_max < 0 or delta_clip <= 0 or writer_hidden < 1:
            raise ValueError("eta_max must be >= 0, delta_clip > 0 and writer_hidden >= 1")
        if write_mode == "label_feedback" and self.n_outputs is None:
            raise ValueError("write_mode='label_feedback' needs n_classes")
        self._config.update(write_mode=write_mode, decay=decay, eta_max=eta_max, delta_clip=delta_clip,
                            writer_hidden=writer_hidden)
        self.write_mode = write_mode
        self.decay = decay
        self.delta_clip = delta_clip
        self.spatial_components = self.spatial.conv.out_channels
        self.writer: FastWeightWriter | None = None
        if write_mode != "off":
            n_inputs = n_channels + ((self.n_outputs or 0) + 1 if write_mode == "label_feedback" else 0)
            self.writer = FastWeightWriter(n_inputs, n_channels, self.spatial_components, writer_hidden, eta_max)

    def init_state(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        """Zero fast weights ``ΔW`` of shape (batch, spatial_components, n_channels)."""
        weight = self.spatial.conv.weight
        return torch.zeros(batch_size, self.spatial_components, self.n_channels, dtype=weight.dtype,
                           device=weight.device if device is None else device)

    def _check(self, x: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (batch, n_channels, n_times); got {x.shape}")
        if x.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels; got {x.shape[1]}")
        if state is None:
            return self.init_state(x.shape[0], x.device)
        expected = (x.shape[0], self.spatial_components, self.n_channels)
        if tuple(state.shape) != expected:
            raise ValueError(f"state must have shape {expected}; got {tuple(state.shape)}")
        return state

    def forward_features(self, x: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """Gated embedding read with ``W_slow + state``; a zero state gives GRFNet's embedding."""
        state = self._check(x, state)
        weight = self.spatial.conv.weight[:, :, 0]
        z = self.spatial.bn(torch.einsum("bkc,bct->bkt", weight + state, x))
        feats = self.feature_norm(self.bank(z))
        pooled = self._pool(feats)
        gate = torch.softmax(self.stream_gate, dim=0)
        blocks = [gate[i] * pooled[:, i * self.n_features:(i + 1) * self.n_features]
                  for i in range(self.n_statistics)]
        return torch.cat(blocks, dim=1)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None,  # pyright: ignore[reportIncompatibleMethodOverride]
                label: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict a batch of trials, then write. Returns ``(logits, new_state)``.

        ``label`` holds class indices, with -1 for unknown, and is used only by ``label_feedback``
        writes, after the logits are computed.
        """
        state = self._check(x, state)
        embedding = self.forward_features(x, state)
        out = embedding if self.classify is None else self.classify(self.drop(embedding))
        return out, self.write(x, state, label)

    def summary(self, x: torch.Tensor) -> torch.Tensor:
        """Writer input: log-variance per input channel, standardised across channels."""
        logvar = torch.log(x.var(dim=-1, unbiased=False) + self.logvar_eps)
        return (logvar - logvar.mean(dim=1, keepdim=True)) / (logvar.std(dim=1, keepdim=True) + 1e-6)

    def write(self, x: torch.Tensor, state: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:
        """Apply one delta-rule write for each trial in the batch."""
        if self.writer is None:
            return state
        summary = self.summary(x)
        if self.write_mode == "label_feedback":
            n_classes = self.n_outputs or 0
            if label is None:
                label = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
            if label.shape != (x.shape[0],):
                raise ValueError(f"label must have shape ({x.shape[0]},); got {tuple(label.shape)}")
            known = (label >= 0).to(summary.dtype)
            onehot = nn.functional.one_hot(label.clamp(min=0), n_classes).to(summary.dtype) * known[:, None]
            summary = torch.cat([summary, onehot, known[:, None]], dim=1)
        k, v, eta = self.writer(summary)
        error = v - torch.einsum("bkc,bc->bk", state, k)
        new = self.decay * state + eta[:, None, None] * error[:, :, None] * k[:, None, :]
        norm = new.flatten(1).norm(dim=1).clamp_min(1e-12)
        return new * (self.delta_clip / norm).clamp(max=1.0)[:, None, None]

    def __prepare_scriptable__(self):
        raise ValueError("FastGRFNet is stateful and cannot be compiled with TorchScript; use torch.export instead")
