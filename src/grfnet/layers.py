"""Spatial front-end and kernel bank shared by GRF-Net models."""

import math
import operator

import torch
from torch import nn

_REFERENCE_SFREQ = 250.0


def scale_dilations(base_dilations, sfreq: float, reference_sfreq: float = _REFERENCE_SFREQ) -> list[int]:
    """Rescale dilations defined at ``reference_sfreq`` to ``sfreq`` (half-up rounding, minimum 1)."""
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
    # Mean-centred Gaussian kernels, as in ROCKET.
    with torch.no_grad():
        weight.normal_(generator=generator)
        weight.sub_(weight.mean(dim=-1, keepdim=True))


def _channel_coords_from_names(channel_names, n_channels: int):
    from .channels import channel_positions

    names = list(channel_names)
    if len(names) != n_channels:
        raise ValueError(
            f"channel_names must have {n_channels} entries to match n_channels; got {len(names)}"
        )
    return channel_positions(names)


def _coordinate_patterns(coords: torch.Tensor) -> list[torch.Tensor]:
    # Low-order polynomials of the sensor positions: axes, radius, cross terms, differences of squares.
    axes = list(coords.T)
    family = list(axes)
    # Summed left to right to match the reference outputs exactly.
    radius = axes[0] * axes[0]
    for a in axes[1:]:
        radius = radius + a * a
    family.append(radius)
    family += [axes[i] * axes[j] for i in range(len(axes)) for j in range(i + 1, len(axes))]
    family += [axes[i] * axes[i] - axes[i + 1] * axes[i + 1] for i in range(len(axes) - 1)]
    return family


def _geometry_seed_spatial_(weight: torch.Tensor, coords) -> int:
    """Overwrite the first rows of a (components, channels, 1) weight with geometry patterns.

    Every row ends up a zero-mean, unit-norm contrast. Returns the number of seeded rows.
    """
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
            continue  # constant over this layout
        feature = feature / norm
        if any(abs(torch.dot(feature, b)) > 1 - 1e-6 for b in basis):
            continue  # parallel to an earlier pattern
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
    """1x1 convolution over channels followed by BatchNorm, optionally seeded from sensor positions."""

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
        return self.bn(self.conv(x))


class RocketFeatureBank(nn.Module):
    """Depthwise dilated convolutions: frozen random kernels plus optional learnable ones, per dilation.

    ``seed`` draws the frozen kernels from a private generator.
    """

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
            return nn.Conv1d(
                s,
                s * out_per_group,
                kernel_length,
                padding=dilation * (kernel_length - 1) // 2,
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
            conv.weight.requires_grad_(False)
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
            return torch.cat(fixed_maps, dim=1)
        learnable_maps = []
        for conv in self.learnable:
            learnable_maps.append(conv(x))
        # Keep each dilation's fixed and learnable maps together.
        maps = []
        for i in range(len(fixed_maps)):
            maps.append(fixed_maps[i])
            maps.append(learnable_maps[i])
        return torch.cat(maps, dim=1)
