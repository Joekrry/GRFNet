"""Building blocks shared by GRF-Net models: dilation scaling, geometry seeding, the spatial front-end and the convolution bank."""

import math
import operator

import torch
from torch import nn

_REFERENCE_SFREQ = 250.0


def scale_dilations(base_dilations, sfreq: float, reference_sfreq: float = _REFERENCE_SFREQ) -> list[int]:
    """Rescale dilations set at reference_sfreq to sfreq, rounding half up, with a minimum of 1."""
    if sfreq <= 0 or reference_sfreq <= 0:
        raise ValueError(f"sampling rates must be positive; got sfreq={sfreq}, reference_sfreq={reference_sfreq}")
    return [max(1, math.floor(b * sfreq / reference_sfreq + 0.5)) for b in base_dilations]


def _check_dilations(dilations) -> list[int]:
    try:
        dilations = [operator.index(d) for d in dilations]
    except TypeError:
        raise ValueError(f"dilations must be integers; got {list(dilations)}") from None
    if not dilations or min(dilations) < 1:
        raise ValueError(f"dilations must be a non-empty sequence of integers >= 1; got {dilations}")
    return dilations


def _rocket_init_(weight: torch.Tensor, generator: torch.Generator | None = None) -> None:
    """Seed a conv weight (out, 1, K) ROCKET-style: random kernels, each mean-centred over its taps (so a constant input yields no response)."""
    with torch.no_grad():
        weight.normal_(generator=generator)
        weight.sub_(weight.mean(dim=-1, keepdim=True))


def _channel_coords_from_names(channel_names, n_channels: int):
    """2-D coordinates for named EEG electrodes, via the 10-20 lookup."""
    from .channels import channel_positions

    names = list(channel_names)
    if len(names) != n_channels:
        raise ValueError(
            f"channel_names must have {n_channels} entries to match n_channels; got {len(names)}"
        )
    return channel_positions(names)


def _coordinate_patterns(coords: torch.Tensor) -> list[torch.Tensor]:
    """Smooth patterns over sensors at coords (n_channels, d): one gradient per axis, then the squared radius, cross products and differences of squares."""
    axes = list(coords.T)
    family = list(axes)
    # Left-to-right sum matches the reference outputs bit for bit.
    radius = axes[0] * axes[0]
    for a in axes[1:]:
        radius = radius + a * a
    family.append(radius)
    family += [axes[i] * axes[j] for i in range(len(axes)) for j in range(i + 1, len(axes))]
    family += [axes[i] * axes[i] - axes[i + 1] * axes[i + 1] for i in range(len(axes) - 1)]
    return family


def _geometry_seed_spatial_(weight: torch.Tensor, coords) -> int:
    """Seed a spatial-conv weight (spatial_components, n_channels, 1) with coordinate patterns as unit-norm contrasts. Rows beyond the usable patterns keep their random init, rescaled to unit-norm contrasts. Returns the number of geometry-seeded rows."""
    n_components, n_channels, _ = weight.shape
    coords = torch.as_tensor(coords, dtype=weight.dtype)
    if coords.dim() == 1:
        coords = coords[:, None]
    if coords.dim() != 2 or coords.shape[0] != n_channels or coords.shape[1] < 1:
        raise ValueError(
            f"channel_coords must have shape ({n_channels}, d) or ({n_channels},); got {tuple(coords.shape)}"
        )
    if not torch.isfinite(coords).all():
        raise ValueError("channel_coords must be finite")
    basis = []
    for feature in _coordinate_patterns(coords):
        feature = feature - feature.mean()
        norm = feature.norm()
        if norm <= 1e-6:
            continue  # degenerate for this layout
        feature = feature / norm
        if any(abs(torch.dot(feature, b)) > 1 - 1e-6 for b in basis):
            continue  # duplicate of an earlier pattern
        basis.append(feature)
    if not basis:
        return 0
    n_seeded = min(len(basis), n_components)
    with torch.no_grad():
        for out_channel in range(n_seeded):
            weight[out_channel, :, 0] = basis[out_channel]
        rest = weight[n_seeded:, :, 0]
        rest -= rest.mean(dim=1, keepdim=True)
        rest /= rest.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return n_seeded


class SpatialFrontEnd(nn.Module):
    """Learnable spatial mixer: a 1x1 conv over the input channels giving spatial_components components, then BatchNorm. Optionally geometry-seeded from sensor coordinates."""

    def __init__(
        self,
        n_channels: int,
        spatial_components: int,
        *,
        spatial_init: str | None = None,
        channel_names=None,
        channel_coords=None,
    ) -> None:
        super().__init__()
        # geometry is used when sensor positions are supplied and nothing else is asked for; random is the default
        if spatial_init is None:
            has_positions = channel_names is not None or channel_coords is not None
            spatial_init = "geometry" if has_positions else "random"
        if spatial_init not in ("random", "geometry"):
            raise ValueError(f"spatial_init must be 'random' or 'geometry'; got {spatial_init!r}")
        self.conv = nn.Conv1d(n_channels, spatial_components, 1, bias=False)
        self.bn = nn.BatchNorm1d(spatial_components, momentum=0.01, eps=1e-3)
        self.spatial_init = spatial_init
        self.n_geometry_components = 0
        if spatial_init == "geometry":
            if channel_coords is None:
                if channel_names is None:
                    raise ValueError(
                        "spatial_init='geometry' requires channel_names or channel_coords"
                    )
                channel_coords = _channel_coords_from_names(channel_names, n_channels)
            self.n_geometry_components = _geometry_seed_spatial_(self.conv.weight, channel_coords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))  # (batch, spatial_components, n_times)


class RocketFeatureBank(nn.Module):
    """Multi-dilation depthwise conv bank over the spatial components: a frozen ROCKET-style kernel set (low variance) plus an optional few learnable kernels. One kernel group per dilation. seed draws the frozen kernels from a private generator instead of the global RNG."""

    def __init__(
        self,
        spatial_components: int,
        *,
        kernel_length: int,
        dilations,
        fixed_per_group: int,
        learnable_per_group: int,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if kernel_length % 2 == 0:
            raise ValueError(f"kernel_length must be odd for symmetric padding; got {kernel_length}")
        if fixed_per_group < 1:
            raise ValueError(f"fixed_per_group must be >= 1; got {fixed_per_group}")
        if learnable_per_group < 0:
            raise ValueError(f"learnable_per_group must be >= 0; got {learnable_per_group}")
        self.dilations = _check_dilations(dilations)
        s = spatial_components

        def _conv(out_per_group: int, dilation: int) -> nn.Conv1d:
            # groups=s = depthwise
            return nn.Conv1d(
                s,
                s * out_per_group,
                kernel_length,
                padding=dilation * (kernel_length - 1) // 2,  # "same" length for odd kernels
                dilation=dilation,
                groups=s,
                bias=False,
            )

        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(operator.index(seed))
        self.seed = seed

        self.fixed = nn.ModuleList()
        for dilation in self.dilations:
            conv = _conv(fixed_per_group, dilation)
            _rocket_init_(conv.weight, generator)
            conv.weight.requires_grad_(False)  # frozen ROCKET bank (low overfitting)
            self.fixed.append(conv)

        self.learnable = (
            nn.ModuleList(_conv(learnable_per_group, d) for d in self.dilations)
            if learnable_per_group > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fixed_maps = []
        for conv in self.fixed:
            fixed_maps.append(conv(x))
        if self.learnable is None:
            return torch.cat(fixed_maps, dim=1)  # (batch, n_features, n_times)
        learnable_maps = []
        for conv in self.learnable:
            learnable_maps.append(conv(x))
        # Interleaved per dilation: fixed group, then learnable group.
        maps = []
        for i in range(len(fixed_maps)):
            maps.append(fixed_maps[i])
            maps.append(learnable_maps[i])
        return torch.cat(maps, dim=1)  # (batch, n_features, n_times)
