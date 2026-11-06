import math
import operator

import numpy as np
import torch
from torch import nn

CHECKPOINT_FORMAT = "grfnet-checkpoint-1"
_CUSTOM = "custom"
_STATISTICS = ("ppv", "logvar", "mean")
_BUILTIN_STATISTICS = ("ppv", "logvar", "mean", "max", "std", "bandpower")
_STATISTIC_ALIASES = {"bandpower": "logvar"}
_BASE_DILATIONS = (1, 2, 4, 8)
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
    family.append(sum(a * a for a in axes))
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


class GRFNet(nn.Module):
    """GRF-Net (Geometry-aware Rocket-Fusion Network) for multichannel time series: geometry-aware learnable spatial front-end -> ROCKET-style multi-dilation conv bank -> fusion pooling -> gated embedding -> head. statistics selects the fusion streams; learnable_per_group=0 is the pure-fixed-bank ablation. Accepts inputs of any length; n_times is recorded only.

    head is "linear" (needs n_outputs or its alias n_classes), None (forward returns the embedding), an nn.Module, or a callable mapping embedding_dim to an nn.Module."""

    def __init__(
        self,
        n_channels: int,
        n_times: int | None = None,
        n_classes: int | None = None,
        *,
        n_outputs: int | None = None,
        head="linear",
        sfreq: float | None = None,
        spatial_components: int = 16,
        spatial_init: str | None = None,
        channel_names=None,
        channel_coords=None,
        kernel_length: int = 9,
        base_dilations=None,
        dilations=None,
        fixed_per_group: int = 2,
        learnable_per_group: int = 1,
        statistics=_STATISTICS,
        ppv_tau: float = 0.5,
        ppv_bias: bool = True,
        logvar_eps: float = 1e-6,
        dropout: float = 0.3,
        bank_seed: int | None = None,
    ) -> None:
        super().__init__()
        self._config = {
            "n_channels": n_channels,
            "n_times": n_times,
            "n_classes": n_classes,
            "n_outputs": n_outputs,
            "head": head if head is None or isinstance(head, str) else _CUSTOM,
            "sfreq": sfreq,
            "spatial_components": spatial_components,
            "spatial_init": spatial_init,
            "channel_names": channel_names,
            "channel_coords": channel_coords,
            "kernel_length": kernel_length,
            "base_dilations": base_dilations,
            "dilations": dilations,
            "fixed_per_group": fixed_per_group,
            "learnable_per_group": learnable_per_group,
            "statistics": tuple(s if isinstance(s, str) else _CUSTOM for s in statistics),
            "ppv_tau": ppv_tau,
            "ppv_bias": ppv_bias,
            "logvar_eps": logvar_eps,
            "dropout": dropout,
            "bank_seed": bank_seed,
        }
        if n_outputs is None:
            n_outputs = n_classes
        elif n_classes is not None and n_classes != n_outputs:
            raise ValueError(f"n_classes and n_outputs disagree; got {n_classes} and {n_outputs}")
        if isinstance(head, str) and head == "linear":
            if n_outputs is None:
                raise TypeError("GRFNet requires n_outputs or n_classes for the linear head; pass head=None for embeddings")
        elif head is None:
            if n_outputs is not None:
                raise ValueError("head=None returns embeddings; leave n_outputs and n_classes unset")
        elif not (isinstance(head, nn.Module) or callable(head)):
            raise ValueError(f"head must be 'linear', None, an nn.Module or a callable; got {head!r}")
        if n_times is not None and n_times < 1:
            raise ValueError(f"n_times must be >= 1; got {n_times}")
        if sfreq is not None and sfreq <= 0:
            raise ValueError(f"sfreq must be positive; got {sfreq}")
        # Explicit dilations are used as given. Otherwise base dilations are set at 250 Hz and rescaled by sfreq.
        if dilations is not None:
            if sfreq is not None or base_dilations is not None:
                raise ValueError("pass dilations on its own, or sfreq and base_dilations, not both")
        else:
            base = _BASE_DILATIONS if base_dilations is None else base_dilations
            dilations = base if sfreq is None else scale_dilations(base, sfreq)
        statistics = tuple(statistics)
        if not statistics:
            raise ValueError("statistics must name at least one fusion stream")
        stat_keys = []
        for stat in statistics:
            if isinstance(stat, str) and stat in _BUILTIN_STATISTICS:
                stat_keys.append(_STATISTIC_ALIASES.get(stat, stat))
            elif callable(stat) and not isinstance(stat, str):
                stat_keys.append(stat)
            else:
                raise ValueError(f"statistics must be callables or names from {_BUILTIN_STATISTICS}; got {stat!r}")
        if len(set(stat_keys)) != len(stat_keys):
            raise ValueError(f"statistics must not repeat (bandpower is an alias of logvar); got {statistics}")
        if ppv_tau <= 0:
            raise ValueError(f"ppv_tau must be positive; got {ppv_tau}")
        if logvar_eps <= 0:
            raise ValueError(f"logvar_eps must be positive; got {logvar_eps}")

        self.n_channels = n_channels
        self.n_times = n_times
        self.n_outputs = n_outputs
        self.n_classes = n_outputs
        self.sfreq = sfreq
        self.statistics = statistics
        self.n_statistics = len(statistics)
        # Built-in names, or "custom" with the function kept in _custom_statistics.
        self._stat_names = [key if isinstance(key, str) else _CUSTOM for key in stat_keys]
        self._custom_statistics = {i: key for i, key in enumerate(stat_keys) if not isinstance(key, str)}
        self.ppv_tau = ppv_tau
        self.logvar_eps = logvar_eps

        self.spatial = SpatialFrontEnd(
            n_channels,
            spatial_components,
            spatial_init=spatial_init,
            channel_names=channel_names,
            channel_coords=channel_coords,
        )
        self.bank = RocketFeatureBank(
            spatial_components,
            kernel_length=kernel_length,
            dilations=dilations,
            fixed_per_group=fixed_per_group,
            learnable_per_group=learnable_per_group,
            seed=bank_seed,
        )

        n_features = len(self.bank.dilations) * spatial_components * (fixed_per_group + learnable_per_group)
        self.n_features = n_features
        self.embedding_dim = len(statistics) * n_features
        self.feature_norm = nn.BatchNorm1d(n_features, momentum=0.01, eps=1e-3)
        self.ppv_bias = (
            nn.Parameter(torch.zeros(n_features)) if (ppv_bias and "ppv" in stat_keys) else None
        )
        self.stream_gate = nn.Parameter(torch.zeros(len(statistics)))
        self.drop = nn.Dropout(dropout)
        if isinstance(head, str):
            self.classify = nn.Linear(self.embedding_dim, n_outputs)
        elif head is None or isinstance(head, nn.Module):
            self.classify = head
        else:
            self.classify = head(self.embedding_dim)
            if not isinstance(self.classify, nn.Module):
                raise ValueError(f"head factory must return an nn.Module; got {type(self.classify).__name__}")

    def __prepare_scriptable__(self):
        if self._custom_statistics:
            raise ValueError("custom statistics are Python functions and cannot be compiled with TorchScript; use torch.export instead")
        return self

    @torch.jit.unused
    def _custom_statistic(self, index: int, feats: torch.Tensor) -> torch.Tensor:
        stream = self._custom_statistics[index](feats)
        if stream.shape != feats.shape[:2]:
            raise ValueError(
                f"custom statistic must return (batch, n_features) = {tuple(feats.shape[:2])}; got {tuple(stream.shape)}"
            )
        return stream

    def _pool(self, feats: torch.Tensor) -> torch.Tensor:
        """Fuse statistics over the time axis into (batch, len(statistics) * n_features)."""
        streams = []
        for i, key in enumerate(self._stat_names):
            if key == "ppv":
                z = feats if self.ppv_bias is None else feats - self.ppv_bias[None, :, None]
                stream = torch.sigmoid(z / self.ppv_tau).mean(dim=-1)
            elif key == "logvar":
                stream = torch.log(feats.var(dim=-1, unbiased=False) + self.logvar_eps)
            elif key == "mean":
                stream = feats.mean(dim=-1)
            elif key == "max":
                stream = feats.amax(dim=-1)
            elif key == "std":
                stream = torch.sqrt(feats.var(dim=-1, unbiased=False) + self.logvar_eps)
            else:
                stream = self._custom_statistic(i, feats)
            streams.append(stream)
        return torch.cat(streams, dim=-1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Gated fusion embedding (batch, embedding_dim): the head's input, before dropout."""
        if x.dim() != 3:
            raise ValueError(f"expected (batch, n_channels, n_times); got {x.shape}")
        if x.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels; got {x.shape[1]}")
        components = self.spatial(x)  # (batch, spatial_components, n_times)
        feats = self.feature_norm(self.bank(components))  # (batch, n_features, n_times)
        pooled = self._pool(feats)
        # Soft per-stream gating over the pooled statistic blocks.
        gate = torch.softmax(self.stream_gate, dim=0)
        blocks = []
        for i in range(self.n_statistics):
            blocks.append(gate[i] * pooled[:, i * self.n_features : (i + 1) * self.n_features])
        return torch.cat(blocks, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.forward_features(x)
        if self.classify is None:
            return embedding
        return self.classify(self.drop(embedding))

    @torch.jit.unused
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.jit.unused
    def get_config(self) -> dict:
        """Constructor arguments as JSON-compatible values. Raises for custom heads and statistics, which cannot be stored."""
        config = dict(self._config)
        if config["head"] == _CUSTOM:
            raise ValueError("a custom head cannot be stored in a config; rebuild the model in code and load its state_dict")
        if _CUSTOM in config["statistics"]:
            raise ValueError("custom statistics cannot be stored in a config; rebuild the model in code and load its state_dict")
        config["statistics"] = list(config["statistics"])
        for key in ("channel_names", "channel_coords", "base_dilations", "dilations"):
            if config[key] is not None:
                config[key] = np.asarray(config[key]).tolist()
        return config

    @classmethod
    def from_config(cls, config: dict) -> "GRFNet":
        """Build an untrained model from get_config() output."""
        return cls(**config)

    @torch.jit.unused
    def save(self, path, *, metadata: dict | None = None) -> None:
        """Write config, weights and optional metadata (plain Python values) to a checkpoint file."""
        from . import __version__

        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "model": type(self).__name__,
                "grfnet_version": __version__,
                "config": self.get_config(),
                "state_dict": self.state_dict(),
                "metadata": {} if metadata is None else dict(metadata),
            },
            path,
        )

    @classmethod
    def load(cls, path, *, map_location="cpu") -> "GRFNet":
        """Rebuild a model saved with save(). Returns it in eval mode."""
        checkpoint = load_checkpoint(path, map_location=map_location)
        if checkpoint["model"] != cls.__name__:
            raise ValueError(f"checkpoint holds a {checkpoint['model']}, not a {cls.__name__}")
        model = cls.from_config(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.eval()


def load_checkpoint(path, *, map_location="cpu") -> dict:
    """Read a checkpoint written by save() without building the model. Keys: format, model, grfnet_version, config, state_dict, metadata."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a GRFNet checkpoint")
    return checkpoint
