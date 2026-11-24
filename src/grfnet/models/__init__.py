"""GRF-Net and baseline models. Each model lives in its own subpackage with a README.md."""

from torch import nn

from ..checkpoint import load_checkpoint
from .eegnet import EEGNet
from .fastgrfnet import FastGRFNet
from .grfnet import GRFNet
from .grfnetseq import GRFNetSeq
from .shallowconvnet import ShallowConvNet

# Names used in experiment configs and stored in checkpoints.
MODELS: dict[str, type[nn.Module]] = {"GRFNet": GRFNet, "FastGRFNet": FastGRFNet, "GRFNetSeq": GRFNetSeq,
                                     "EEGNet": EEGNet, "ShallowConvNet": ShallowConvNet}


def load_model(path, *, map_location="cpu") -> nn.Module:
    """Rebuild any model in ``MODELS`` from a checkpoint, in eval mode."""
    name = load_checkpoint(path, map_location=map_location)["model"]
    if name not in MODELS:
        raise ValueError(f"{path} holds an unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name].load(path, map_location=map_location)  # pyright: ignore[reportAttributeAccessIssue]


__all__ = ["EEGNet", "FastGRFNet", "GRFNet", "GRFNetSeq", "MODELS", "ShallowConvNet", "load_model"]
