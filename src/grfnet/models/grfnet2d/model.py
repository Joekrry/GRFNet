"""GRFNet2d: GRF-Net on log-power spectrograms, with electrodes as image channels."""

from __future__ import annotations

import operator

import torch
from torch import nn

from ...checkpoint import load_checkpoint, save_checkpoint
from ...layers import _channel_coords_from_names, _geometry_seed_spatial_

STATISTICS = ("ppv", "logvar", "mean")


class GRFNet2d(nn.Module):
    """GRF-Net applied to time-frequency images.

    Each electrode's signal becomes a log-power spectrogram (short-time Fourier transform with a
    Hann window), giving an image of shape (electrodes, frequencies, frames). A 1x1 convolution
    mixes the electrodes into spatial components, optionally seeded from sensor positions. A bank
    of frozen and learnable depthwise 2-D kernels, dilated along time, follows, then GRF-Net's
    pooling statistics over the whole image, gate and linear head.

    Parameters
    ----------
    n_channels, n_classes : int
        Electrodes and classes.
    n_times : int, optional
        Stored for reference only.
    sfreq : float, optional
        Sampling rate in Hz. Needed for ``fmax``.
    n_fft, hop : int
        STFT window length and step, in samples.
    fmax : float, optional
        Highest frequency kept, in Hz. ``None`` keeps all bins.
    kernel_size : (int, int)
        Kernel height (frequency bins) and width (frames). Both odd.
    dilations : sequence of int
        Dilations along time, in frames.
    **others
        ``spatial_components``, ``spatial_init``, ``channel_names``, ``channel_coords``,
        ``fixed_per_group``, ``learnable_per_group``, ``ppv_tau``, ``logvar_eps``, ``dropout`` and
        ``bank_seed``, as in GRFNet.
    """

    # The sampling rate sets the kept frequencies; it does not rescale dilations.
    needs_sfreq = True

    def __init__(self, n_channels: int, n_times: int | None = None, n_classes: int | None = None, *,
                 sfreq: float | None = None, n_fft: int = 64, hop: int = 16, fmax: float | None = None,
                 spatial_components: int = 16, spatial_init: str | None = None, channel_names=None,
                 channel_coords=None, kernel_size=(3, 5), dilations=(1, 2, 4), fixed_per_group: int = 2,
                 learnable_per_group: int = 1, ppv_tau: float = 0.5, logvar_eps: float = 1e-6,
                 dropout: float = 0.3, bank_seed: int | None = None) -> None:
        super().__init__()
        kernel_size = tuple(int(k) for k in kernel_size)
        dilations = [operator.index(d) for d in dilations]
        self._config = {
            "n_channels": n_channels, "n_times": n_times, "n_classes": n_classes, "sfreq": sfreq, "n_fft": n_fft,
            "hop": hop, "fmax": fmax, "spatial_components": spatial_components, "spatial_init": spatial_init,
            "channel_names": None if channel_names is None else list(channel_names),
            "channel_coords": None if channel_coords is None else torch.as_tensor(channel_coords).tolist(),
            "kernel_size": list(kernel_size), "dilations": dilations, "fixed_per_group": fixed_per_group,
            "learnable_per_group": learnable_per_group, "ppv_tau": ppv_tau, "logvar_eps": logvar_eps,
            "dropout": dropout, "bank_seed": bank_seed,
        }
        if n_classes is None or n_classes < 1:
            raise ValueError("GRFNet2d needs n_classes")
        if n_fft < 2 or hop < 1:
            raise ValueError(f"n_fft must be >= 2 and hop >= 1; got {n_fft} and {hop}")
        if fmax is not None and sfreq is None:
            raise ValueError("fmax needs sfreq")
        if len(kernel_size) != 2 or any(k < 1 or k % 2 == 0 for k in kernel_size):
            raise ValueError(f"kernel_size must be two odd sizes; got {kernel_size}")
        if not dilations or min(dilations) < 1:
            raise ValueError(f"dilations must be integers >= 1; got {dilations}")
        if fixed_per_group < 1 or learnable_per_group < 0:
            raise ValueError("fixed_per_group must be >= 1 and learnable_per_group >= 0")
        if ppv_tau <= 0 or logvar_eps <= 0:
            raise ValueError("ppv_tau and logvar_eps must be positive")
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.n_fft = n_fft
        self.hop = hop
        self.statistics = STATISTICS
        self.ppv_tau = ppv_tau
        self.logvar_eps = logvar_eps
        n_bins = n_fft // 2 + 1
        if fmax is not None:
            assert sfreq is not None
            n_bins = sum(1 for k in range(n_bins) if k * sfreq / n_fft <= fmax)
        self.n_bins = n_bins
        self.register_buffer("stft_window", torch.hann_window(n_fft), persistent=False)

        self.spatial = nn.Conv2d(n_channels, spatial_components, 1, bias=False)
        self.spatial_norm = nn.BatchNorm2d(spatial_components, momentum=0.01, eps=1e-3)
        if spatial_init is None:
            spatial_init = "geometry" if channel_names is not None or channel_coords is not None else "random"
        if spatial_init not in ("random", "geometry"):
            raise ValueError(f"spatial_init must be 'random' or 'geometry'; got {spatial_init!r}")
        if spatial_init == "geometry":
            if channel_coords is None:
                if channel_names is None:
                    raise ValueError("spatial_init='geometry' requires channel_names or channel_coords")
                channel_coords = _channel_coords_from_names(channel_names, n_channels)
            _geometry_seed_spatial_(self.spatial.weight[..., 0], channel_coords)

        generator = None if bank_seed is None else torch.Generator().manual_seed(operator.index(bank_seed))
        self.dilations = dilations
        self.fixed = nn.ModuleList()
        self.learnable = nn.ModuleList() if learnable_per_group > 0 else None
        for d in dilations:
            fixed = self._conv(spatial_components, fixed_per_group, kernel_size, d)
            with torch.no_grad():
                fixed.weight.normal_(generator=generator)
                fixed.weight.sub_(fixed.weight.mean(dim=(-2, -1), keepdim=True))
            fixed.weight.requires_grad_(False)
            self.fixed.append(fixed)
            if self.learnable is not None:
                self.learnable.append(self._conv(spatial_components, learnable_per_group, kernel_size, d))

        self.n_features = len(dilations) * spatial_components * (fixed_per_group + learnable_per_group)
        self.feature_norm = nn.BatchNorm2d(self.n_features, momentum=0.01, eps=1e-3)
        self.ppv_bias = nn.Parameter(torch.zeros(self.n_features))
        self.stream_gate = nn.Parameter(torch.zeros(len(STATISTICS)))
        self.drop = nn.Dropout(dropout)
        self.classify = nn.Linear(len(STATISTICS) * self.n_features, n_classes)

    @staticmethod
    def _conv(components: int, per_group: int, kernel_size: tuple[int, ...], dilation: int) -> nn.Conv2d:
        kf, kt = kernel_size
        return nn.Conv2d(components, components * per_group, (kf, kt), dilation=(1, dilation),
                         padding=((kf - 1) // 2, dilation * (kt - 1) // 2), groups=components, bias=False)

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Log-power spectrogram of each electrode, shape (batch, n_channels, n_bins, frames)."""
        if x.dim() != 3:
            raise ValueError(f"expected (batch, n_channels, n_times); got {x.shape}")
        if x.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels; got {x.shape[1]}")
        if x.shape[-1] <= self.n_fft // 2:
            raise ValueError(f"inputs need more than n_fft // 2 = {self.n_fft // 2} samples; got {x.shape[-1]}")
        batch, channels, n_times = x.shape
        window = self.stft_window
        assert isinstance(window, torch.Tensor)
        spec = torch.stft(x.reshape(batch * channels, n_times).float(), self.n_fft, self.hop,
                          window=window.to(x.device), center=True, return_complex=True)
        power = spec.abs().square()[:, :self.n_bins]
        return torch.log(power + self.logvar_eps).reshape(batch, channels, self.n_bins, -1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Gated embedding the head receives, shape (batch, 3 * n_features)."""
        z = self.spatial_norm(self.spatial(self.spectrogram(x)))
        maps = []
        for i, fixed in enumerate(self.fixed):
            maps.append(fixed(z))
            if self.learnable is not None:
                maps.append(self.learnable[i](z))
        h = self.feature_norm(torch.cat(maps, dim=1)).flatten(2)
        ppv = torch.sigmoid((h - self.ppv_bias[None, :, None]) / self.ppv_tau).mean(dim=-1)
        logvar = torch.log(h.var(dim=-1, unbiased=False) + self.logvar_eps)
        mean = h.mean(dim=-1)
        gate = torch.softmax(self.stream_gate, dim=0)
        return torch.cat([gate[0] * ppv, gate[1] * logvar, gate[2] * mean], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.drop(self.forward_features(x)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict) -> GRFNet2d:
        return cls(**config)

    def save(self, path, *, metadata: dict | None = None) -> None:
        """Save config, weights and optional metadata to one file."""
        save_checkpoint(self, path, metadata=metadata)

    @classmethod
    def load(cls, path, *, map_location="cpu") -> GRFNet2d:
        """Rebuild a model written by ``save``, in eval mode."""
        checkpoint = load_checkpoint(path, map_location=map_location)
        if checkpoint["model"] != cls.__name__:
            raise ValueError(f"checkpoint holds a {checkpoint['model']}, not a {cls.__name__}")
        model = cls.from_config(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.eval()
