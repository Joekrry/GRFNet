"""Shared wrapper giving braindecode models the GRF-Net config and checkpoint interface."""

from __future__ import annotations

import importlib
import warnings

import torch
from torch import nn

from ..checkpoint import load_checkpoint, save_checkpoint


class BraindecodeModel(nn.Module):
    """A braindecode model built from plain arguments, with ``get_config``, ``save`` and ``load``.

    Subclasses set ``_braindecode_name``. Extra keyword arguments go to the braindecode
    constructor and must be JSON-friendly to be stored in a checkpoint.
    """

    _braindecode_name: str = ""

    def __init__(self, n_channels: int, n_times: int, n_classes: int, **kwargs) -> None:
        super().__init__()
        for key in ("n_chans", "n_outputs"):
            if key in kwargs:
                raise ValueError(f"pass n_channels and n_classes, not braindecode's {key}")
        try:
            module = importlib.import_module("braindecode.models")
        except ImportError as error:
            raise ImportError(
                f"{type(self).__name__} requires braindecode; install it with: pip install grfnet[baselines]"
            ) from error
        self._config = {"n_channels": n_channels, "n_times": n_times, "n_classes": n_classes, **kwargs}
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        with warnings.catch_warnings():
            # braindecode 2.0 warns about its own default for this argument.
            warnings.filterwarnings("ignore", message=".*final_layer_with_constraint.*")
            self.net = getattr(module, self._braindecode_name)(
                n_chans=n_channels, n_outputs=n_classes, n_times=n_times, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def save(self, path, *, metadata: dict | None = None) -> None:
        """Save config, weights and optional metadata to one file."""
        save_checkpoint(self, path, metadata=metadata)

    @classmethod
    def load(cls, path, *, map_location="cpu"):
        """Rebuild a model written by ``save``, in eval mode."""
        checkpoint = load_checkpoint(path, map_location=map_location)
        if checkpoint["model"] != cls.__name__:
            raise ValueError(f"checkpoint holds a {checkpoint['model']}, not a {cls.__name__}")
        model = cls.from_config(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.eval()
