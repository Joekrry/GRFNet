from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from ...checkpoint import load_checkpoint, save_checkpoint
from ...layers import RocketFeatureBank, SpatialFrontEnd, scale_dilations

_CUSTOM = "custom"
_STATISTICS = ("ppv", "logvar", "mean")
_BUILTIN_STATISTICS = ("ppv", "logvar", "mean", "max", "std", "bandpower")
_STATISTIC_ALIASES = {"bandpower": "logvar"}
_BASE_DILATIONS = (1, 2, 4, 8)


class GRFNet(nn.Module):
    """Geometry-aware Rocket-Fusion Network.

    Spatial filters mix the input channels, a bank of frozen and learnable dilated kernels
    extracts features, and pooled statistics are fused by a softmax gate into an embedding
    for the head. Inputs may have any length.

    Parameters
    ----------
    n_channels : int
        Number of input channels.
    n_times : int, optional
        Expected input length. Stored for reference only.
    n_classes, n_outputs : int, optional
        Size of the linear head. ``n_classes`` is an alias of ``n_outputs``.
    head : {"linear", None} or nn.Module or callable
        ``None`` returns the embedding. A callable receives ``embedding_dim`` and returns a module.
    sfreq : float, optional
        Sampling rate. Rescales the base dilations from their 250 Hz definition.
    spatial_components : int
        Number of spatial filters.
    spatial_init : {"random", "geometry"}, optional
        Defaults to geometry when ``channel_names`` or ``channel_coords`` is given.
    channel_names : list of str, optional
        10-20 electrode names, resolved to positions with MNE.
    channel_coords : array of shape (n_channels, d), optional
        Sensor positions for geometry seeding.
    kernel_length : int
        Length of every bank kernel. Must be odd.
    base_dilations, dilations : sequence of int, optional
        ``dilations`` are used as given; ``base_dilations`` are rescaled by ``sfreq``.
    fixed_per_group, learnable_per_group : int
        Frozen and learnable kernels per spatial filter and dilation.
    statistics : sequence of str or callable
        Pooling statistics: "ppv", "logvar" (or "bandpower"), "mean", "max", "std", or
        functions mapping (batch, features, time) to (batch, features).
    ppv_tau, ppv_bias : float, bool
        Temperature of the soft PPV and whether to learn its threshold.
    logvar_eps : float
        Variance floor for "logvar" and "std".
    dropout : float
        Dropout before the head.
    bank_seed : int, optional
        Seed for the frozen kernels, independent of the global generator.

    References
    ----------
    Dempster, Petitjean and Webb (2020). ROCKET: exceptionally fast and accurate time series
    classification using random convolutional kernels. Data Mining and Knowledge Discovery.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int | None = None,
        n_classes: int | None = None,
        *,
        n_outputs: int | None = None,
        head: str | nn.Module | Callable[[int], nn.Module] | None = "linear",
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
        # TorchScript needs plain names; custom functions are looked up by index.
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
            assert n_outputs is not None  # validated above
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
        """Return the gated embedding the head receives, shape (batch, embedding_dim)."""
        if x.dim() != 3:
            raise ValueError(f"expected (batch, n_channels, n_times); got {x.shape}")
        if x.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels; got {x.shape[1]}")
        feats = self.feature_norm(self.bank(self.spatial(x)))
        pooled = self._pool(feats)
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
        """Constructor arguments as JSON-friendly values. Custom heads and statistics cannot be stored."""
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
        return cls(**config)

    @torch.jit.unused
    def save(self, path, *, metadata: dict | None = None) -> None:
        """Save config, weights and optional metadata to one file."""
        save_checkpoint(self, path, metadata=metadata)

    @classmethod
    def load(cls, path, *, map_location="cpu") -> "GRFNet":
        """Rebuild a model written by ``save``, in eval mode."""
        checkpoint = load_checkpoint(path, map_location=map_location)
        if checkpoint["model"] != cls.__name__:
            raise ValueError(f"checkpoint holds a {checkpoint['model']}, not a {cls.__name__}")
        model = cls.from_config(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.eval()
