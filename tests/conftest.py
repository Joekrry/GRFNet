"""Shared pytest helpers."""

from __future__ import annotations

import importlib.util

import pytest


def requires_torch():
    """Skip marker for when PyTorch is not installed."""
    return pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="torch not installed",
    )
