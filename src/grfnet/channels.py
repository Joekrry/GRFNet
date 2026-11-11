"""EEG electrode positions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def channel_positions(
    names: Sequence[str], *, montage: str = "standard_1020", ndim: int = 2
) -> np.ndarray:
    """Electrode positions from an MNE template, centred and scaled to a unit maximum radius.

    Axes are right, anterior and up; ``ndim=2`` drops the vertical axis.
    """
    if ndim not in (2, 3):
        raise ValueError(f"ndim must be 2 or 3; got {ndim}")
    try:
        import mne
    except ImportError:
        raise ImportError(
            "channel_positions requires MNE-Python; install it with: pip install grfnet[eeg]"
        ) from None

    template = mne.channels.make_standard_montage(montage).get_positions()["ch_pos"]
    by_lower = {name.lower(): name for name in template}
    missing = [name for name in names if name.lower() not in by_lower]
    if missing:
        raise ValueError(f"channels absent from the {montage} template: {missing}")
    coords = np.array(
        [template[by_lower[name.lower()]][:ndim] for name in names], dtype=np.float64
    )

    coords -= coords.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(coords, axis=1).max())
    if radius > 0.0:
        coords /= radius
    return coords
