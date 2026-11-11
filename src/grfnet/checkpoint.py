"""Checkpoint format shared by GRF-Net models."""

import torch
from torch import nn

CHECKPOINT_FORMAT = "grfnet-checkpoint-1"


def save_checkpoint(model: nn.Module, path, *, metadata: dict | None = None) -> None:
    """Save ``model.get_config()``, the state dict and metadata (plain Python values)."""
    from . import __version__

    get_config = getattr(model, "get_config", None)
    if not callable(get_config):
        raise TypeError(f"{type(model).__name__} has no get_config() and cannot be checkpointed")
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model": type(model).__name__,
            "grfnet_version": __version__,
            "config": get_config(),
            "state_dict": model.state_dict(),
            "metadata": {} if metadata is None else dict(metadata),
        },
        path,
    )


def load_checkpoint(path, *, map_location="cpu") -> dict:
    """Read a checkpoint dict without building the model."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a GRFNet checkpoint")
    return checkpoint
