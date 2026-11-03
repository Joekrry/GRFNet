"""Electrode coordinate lookup for geometry-seeded spatial filters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def channel_positions(
    names: Sequence[str], *, montage: str = "standard_1020"
) -> np.ndarray:
    """2-D projected, normalised (n, 2) electrode coordinates for names, from MNE's 10-20 template (x = right, y = anterior; centroid-centred, unit max radius),
    used to seed the GRF-Net spatial conv; raises if a name is absent."""
    import mne

    template = mne.channels.make_standard_montage(montage).get_positions()["ch_pos"]
    by_lower = {name.lower(): name for name in template}
    missing = [name for name in names if name.lower() not in by_lower]
    if missing:
        raise ValueError(f"channels absent from the {montage} template: {missing}")
    coords = np.array(
        [template[by_lower[name.lower()]][:2] for name in names], dtype=np.float64
    )

    # centre on the montage centroid, then scale so the outermost electrode = 1
    coords -= coords.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(coords, axis=1).max())
    if radius > 0.0:
        coords /= radius
    return coords
