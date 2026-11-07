"""Checkpoint files shared by GRF-Net models."""

import torch
from torch import nn

CHECKPOINT_FORMAT = "grfnet-checkpoint-1"


def save_checkpoint(model: nn.Module, path, *, metadata: dict | None = None) -> None:
    """Write model.get_config(), the state_dict and optional metadata (plain Python values) to path."""
    from . import __version__

    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model": type(model).__name__,
            "grfnet_version": __version__,
            "config": model.get_config(),
            "state_dict": model.state_dict(),
            "metadata": {} if metadata is None else dict(metadata),
        },
        path,
    )


def load_checkpoint(path, *, map_location="cpu") -> dict:
    """Read a checkpoint written by save() without building the model. Keys: format, model, grfnet_version, config, state_dict, metadata."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a GRFNet checkpoint")
    return checkpoint
