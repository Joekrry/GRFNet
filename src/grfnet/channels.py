"""EEG electrode coordinate lookup for geometry-seeded spatial filters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def channel_positions(
    names: Sequence[str], *, montage: str = "standard_1020", ndim: int = 2
) -> np.ndarray:
    """Normalised (n, ndim) electrode coordinates for names, from an MNE template (x = right, y = anterior, z = up; centroid-centred, unit max radius).
    ndim=2 drops z. Raises if a name is absent. Requires the eeg extra."""
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

    # centre on the montage centroid, then scale so the outermost electrode = 1
    coords -= coords.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(coords, axis=1).max())
    if radius > 0.0:
        coords /= radius
    return coords
