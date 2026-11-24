"""GRFNetSeq: causal, streaming GRF-Net with a prediction at every time step."""

from __future__ import annotations

import torch
from torch import nn

from ...checkpoint import load_checkpoint, save_checkpoint
from ...layers import RocketFeatureBank, SpatialFrontEnd, scale_dilations

_BASE_DILATIONS = (1, 2, 4, 8)
STATISTICS = ("ppv", "logvar", "mean")


class GRFNetSeq(nn.Module):
    """GRF-Net made causal: every output depends only on the current and earlier samples.

    The kernel bank is left-padded, and each pooling statistic is computed over a sliding window
    of the last ``window`` samples instead of the whole trial. ``forward`` returns logits of shape
    (batch, n_classes, n_times); ``stream`` processes a recording chunk by chunk and gives the same
    outputs in eval mode. The spatial front end, frozen and learnable kernels, statistics (PPV,
    log-variance, mean) and gate are GRF-Net's.

    Parameters
    ----------
    n_channels, n_classes : int
        Input channels and classes.
    n_times : int, optional
        Stored for reference only.
    window : int
        Samples in the sliding window of each statistic.
    **others
        ``sfreq``, ``spatial_components``, ``spatial_init``, ``channel_names``, ``channel_coords``,
        ``kernel_length``, ``base_dilations``, ``dilations``, ``fixed_per_group``,
        ``learnable_per_group``, ``ppv_tau``, ``logvar_eps``, ``dropout`` and ``bank_seed``, as in
        GRFNet.
    """

    def __init__(self, n_channels: int, n_times: int | None = None, n_classes: int | None = None, *,
                 window: int = 64, sfreq: float | None = None, spatial_components: int = 16,
                 spatial_init: str | None = None, channel_names=None, channel_coords=None, kernel_length: int = 9,
                 base_dilations=None, dilations=None, fixed_per_group: int = 2, learnable_per_group: int = 1,
                 ppv_tau: float = 0.5, logvar_eps: float = 1e-6, dropout: float = 0.3,
                 bank_seed: int | None = None) -> None:
        super().__init__()
        self._config = {
            "n_channels": n_channels, "n_times": n_times, "n_classes": n_classes, "window": window,
            "sfreq": sfreq, "spatial_components": spatial_components, "spatial_init": spatial_init,
            "channel_names": None if channel_names is None else list(channel_names),
            "channel_coords": None if channel_coords is None else torch.as_tensor(channel_coords).tolist(),
            "kernel_length": kernel_length, "base_dilations": base_dilations, "dilations": dilations,
            "fixed_per_group": fixed_per_group, "learnable_per_group": learnable_per_group, "ppv_tau": ppv_tau,
            "logvar_eps": logvar_eps, "dropout": dropout, "bank_seed": bank_seed,
        }
        if n_classes is None or n_classes < 1:
            raise ValueError("GRFNetSeq needs n_classes")
        if window < 1:
            raise ValueError(f"window must be >= 1; got {window}")
        if ppv_tau <= 0 or logvar_eps <= 0:
            raise ValueError("ppv_tau and logvar_eps must be positive")
        if dilations is not None:
            if sfreq is not None or base_dilations is not None:
                raise ValueError("pass dilations on its own, or sfreq and base_dilations, not both")
        else:
            base = _BASE_DILATIONS if base_dilations is None else base_dilations
            dilations = base if sfreq is None else scale_dilations(base, sfreq)
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.window = window
        self.statistics = STATISTICS
        self.ppv_tau = ppv_tau
        self.logvar_eps = logvar_eps
        self.spatial = SpatialFrontEnd(n_channels, spatial_components, spatial_init=spatial_init,
                                       channel_names=channel_names, channel_coords=channel_coords)
        self.bank = RocketFeatureBank(spatial_components, kernel_length=kernel_length, dilations=dilations,
                                      fixed_per_group=fixed_per_group, learnable_per_group=learnable_per_group,
                                      seed=bank_seed)
        convs = list(self.bank.fixed) + (list(self.bank.learnable) if self.bank.learnable is not None else [])
        for conv in convs:
            assert isinstance(conv, nn.Conv1d)
            conv.padding = (0,)  # causal: padded on the left in _bank instead
        self.kernel_length = kernel_length
        self.n_features = len(self.bank.dilations) * spatial_components * (fixed_per_group + learnable_per_group)
        self.feature_norm = nn.BatchNorm1d(self.n_features, momentum=0.01, eps=1e-3)
        self.ppv_bias = nn.Parameter(torch.zeros(self.n_features))
        self.stream_gate = nn.Parameter(torch.zeros(len(STATISTICS)))
        self.drop = nn.Dropout(dropout)
        self.classify = nn.Linear(len(STATISTICS) * self.n_features, n_classes)

    @property
    def receptive_field(self) -> int:
        """Input samples behind each bank output."""
        return max(self.bank.dilations) * (self.kernel_length - 1) + 1

    @property
    def context(self) -> int:
        """Past input samples ``stream`` keeps between chunks."""
        return self.receptive_field - 1 + self.window - 1

    def _bank(self, z: torch.Tensor) -> torch.Tensor:
        maps = []
        for i, dilation in enumerate(self.bank.dilations):
            padded = nn.functional.pad(z, (dilation * (self.kernel_length - 1), 0))
            maps.append(self.bank.fixed[i](padded))
            if self.bank.learnable is not None:
                maps.append(self.bank.learnable[i](padded))
        return torch.cat(maps, dim=1)

    def _moving_mean(self, h: torch.Tensor) -> torch.Tensor:
        # Mean over the last `window` samples, or over all samples so far near the start.
        w = self.window
        total = nn.functional.avg_pool1d(nn.functional.pad(h, (w - 1, 0)), w, stride=1) * w
        count = torch.arange(1, h.shape[-1] + 1, device=h.device, dtype=h.dtype).clamp(max=w)
        return total / count

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (batch, n_channels, n_times); got {x.shape}")
        if x.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels; got {x.shape[1]}")
        h = self.feature_norm(self._bank(self.spatial(x)))
        mean = self._moving_mean(h)
        var = (self._moving_mean(h * h) - mean * mean).clamp_min(0.0)
        ppv = self._moving_mean(torch.sigmoid((h - self.ppv_bias[None, :, None]) / self.ppv_tau))
        gate = torch.softmax(self.stream_gate, dim=0)
        streams = [gate[0] * ppv, gate[1] * torch.log(var + self.logvar_eps), gate[2] * mean]
        embedding = torch.cat(streams, dim=1).transpose(1, 2)  # (batch, time, features)
        return self.classify(self.drop(embedding)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits at every time step, shape (batch, n_classes, n_times), from the start of a recording."""
        return self._logits(x)

    def init_stream(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        """Empty buffer for ``stream``."""
        weight = self.spatial.conv.weight
        return torch.zeros(batch_size, self.n_channels, 0, dtype=weight.dtype,
                           device=weight.device if device is None else device)

    def stream(self, chunk: torch.Tensor, buffer: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits for a new chunk of samples and the updated buffer of past samples.

        In eval mode, streaming a recording in chunks of any size gives the outputs of ``forward``
        on the whole recording.
        """
        if buffer is None:
            buffer = self.init_stream(chunk.shape[0], chunk.device)
        full = torch.cat([buffer, chunk], dim=-1)
        logits = self._logits(full)[..., buffer.shape[-1]:]
        return logits, full[..., max(0, full.shape[-1] - self.context):]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict) -> GRFNetSeq:
        return cls(**config)

    def save(self, path, *, metadata: dict | None = None) -> None:
        """Save config, weights and optional metadata to one file."""
        save_checkpoint(self, path, metadata=metadata)

    @classmethod
    def load(cls, path, *, map_location="cpu") -> GRFNetSeq:
        """Rebuild a model written by ``save``, in eval mode."""
        checkpoint = load_checkpoint(path, map_location=map_location)
        if checkpoint["model"] != cls.__name__:
            raise ValueError(f"checkpoint holds a {checkpoint['model']}, not a {cls.__name__}")
        model = cls.from_config(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.eval()
