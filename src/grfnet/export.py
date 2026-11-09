"""ONNX export for GRF-Net models."""

from __future__ import annotations

import torch
from torch import nn


def export_onnx(
    model: nn.Module,
    path,
    *,
    n_times: int | None = None,
    opset: int = 18,
    dynamic: bool = True,
) -> None:
    """Export model in eval mode to an ONNX file. Input "input" is (batch, n_channels, n_times) and output is "output".
    dynamic=True lets batch size and input length vary. Requires the onnx extra."""
    try:
        import onnx  # noqa: F401
        import onnxscript  # noqa: F401
    except ImportError:
        raise ImportError("export_onnx requires onnx and onnxscript; install them with: pip install grfnet[onnx]") from None
    n_channels = getattr(model, "n_channels", None)
    if not isinstance(n_channels, int):
        raise TypeError(f"{type(model).__name__} has no integer n_channels attribute")
    length = n_times if n_times is not None else (getattr(model, "n_times", None) or 256)
    was_training = model.training
    model.eval()
    try:
        example = torch.zeros(2, n_channels, int(length))
        dynamic_shapes = None
        if dynamic:
            dynamic_shapes = {"x": {0: torch.export.Dim("batch", min=1), 2: torch.export.Dim("n_times", min=2)}}
        program = torch.onnx.export(
            model,
            (example,),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset,
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            # PyTorch's progress messages contain emoji that crash non-UTF-8 consoles.
            verbose=False,
        )
        if program is None:
            raise RuntimeError("torch.onnx.export returned no program")
        program.save(str(path))
    finally:
        model.train(was_training)
