"""Shared pytest helpers."""

from __future__ import annotations

import importlib.util

import pytest


def requires_mne():
    """Skip marker for when MNE-Python is not installed."""
    return pytest.mark.skipif(
        importlib.util.find_spec("mne") is None,
        reason="mne not installed",
    )


def requires_onnx():
    """Skip marker for when the onnx extra is not installed."""
    missing = [m for m in ("onnx", "onnxscript", "onnxruntime") if importlib.util.find_spec(m) is None]
    return pytest.mark.skipif(bool(missing), reason=f"not installed: {', '.join(missing)}")


def requires_torch():
    """Skip marker for when PyTorch is not installed."""
    return pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="torch not installed",
    )
