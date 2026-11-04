from __future__ import annotations

import sys

import numpy as np
import pytest

from conftest import requires_mne, requires_torch


def _rows(model):
    return model.spatial.conv.weight.detach().numpy()[:, :, 0]


def _normed(v):
    v = v - v.mean()
    return v / np.linalg.norm(v)


@requires_torch()
def test_two_dimensional_patterns_keep_their_order():
    from grfnet import GRFNet

    coords = np.random.default_rng(0).standard_normal((8, 2))
    x, y = coords[:, 0], coords[:, 1]
    expected = [x, y, x * x + y * y, x * y, x * x - y * y]
    rows = _rows(GRFNet(8, n_classes=2, channel_coords=coords))
    for row, pattern in zip(rows, expected):
        assert np.allclose(row, _normed(pattern), atol=1e-5)


@requires_torch()
@pytest.mark.parametrize("coords, n_patterns", [
    (np.linspace(-1.0, 1.0, 6), 2),
    (np.linspace(-1.0, 1.0, 6)[:, None], 2),
    (np.random.default_rng(1).standard_normal((10, 2)), 5),
    (np.random.default_rng(2).standard_normal((12, 3)), 9),
])
def test_patterns_per_dimension(coords, n_patterns):
    from grfnet import GRFNet

    model = GRFNet(len(coords), n_classes=2, channel_coords=coords)
    assert model.spatial.n_geometry_components == n_patterns


@requires_torch()
def test_three_dimensional_coords_seed_gradients_per_axis():
    from grfnet import GRFNet

    coords = np.random.default_rng(3).standard_normal((12, 3))
    rows = _rows(GRFNet(12, n_classes=2, channel_coords=coords))
    for axis in range(3):
        assert np.allclose(rows[axis], _normed(coords[:, axis]), atol=1e-5)


@requires_torch()
@pytest.mark.parametrize("coords", [
    [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]],
    np.stack([np.linspace(-1.0, 1.0, 4), np.zeros(4)], axis=1),
    np.random.default_rng(4).standard_normal((6, 3)),
])
def test_every_seeded_filter_is_a_distinct_unit_contrast(coords):
    from grfnet import GRFNet

    coords = np.asarray(coords)
    rows = _rows(GRFNet(len(coords), n_classes=2, spatial_components=16, channel_coords=coords))
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-5)
    assert np.allclose(rows.sum(axis=1), 0.0, atol=1e-5)
    similarity = np.abs(rows @ rows.T) - np.eye(len(rows))
    assert similarity.max() < 1 - 1e-6


@requires_torch()
def test_fewer_components_than_patterns():
    from grfnet import GRFNet

    coords = np.random.default_rng(5).standard_normal((12, 3))
    model = GRFNet(12, n_classes=2, spatial_components=4, channel_coords=coords)
    assert model.spatial.n_geometry_components == 4


@requires_torch()
def test_single_channel_falls_back_to_random_init():
    import torch

    from grfnet import GRFNet

    model = GRFNet(1, n_classes=2, channel_coords=[[0.0, 0.0]])
    assert model.spatial.n_geometry_components == 0
    assert model(torch.randn(2, 1, 64)).shape == (2, 2)


@requires_torch()
@pytest.mark.parametrize("coords", [np.zeros((4, 2, 1)), np.full((4, 2), np.nan)])
def test_rejects_malformed_coords(coords):
    from grfnet import GRFNet

    with pytest.raises(ValueError, match="channel_coords"):
        GRFNet(4, n_classes=2, channel_coords=coords)


@requires_torch()
def test_coords_work_without_mne(monkeypatch):
    from grfnet import GRFNet

    monkeypatch.setitem(sys.modules, "mne", None)
    model = GRFNet(4, n_classes=2, channel_coords=np.random.default_rng(6).standard_normal((4, 2)))
    assert model.spatial.spatial_init == "geometry"


def test_channel_positions_explains_missing_mne(monkeypatch):
    from grfnet import channel_positions

    monkeypatch.setitem(sys.modules, "mne", None)
    with pytest.raises(ImportError, match=r"grfnet\[eeg\]"):
        channel_positions(["C3", "Cz"])


def test_channel_positions_rejects_bad_ndim():
    from grfnet import channel_positions

    with pytest.raises(ValueError, match="ndim must be 2 or 3"):
        channel_positions(["C3"], ndim=4)


@requires_mne()
@pytest.mark.parametrize("ndim", [2, 3])
def test_channel_positions_shape_and_scale(ndim):
    from grfnet import channel_positions

    coords = channel_positions(["C3", "Cz", "C4", "Pz", "Fz"], ndim=ndim)
    assert coords.shape == (5, ndim)
    assert np.allclose(coords.mean(axis=0), 0.0)
    assert np.isclose(np.linalg.norm(coords, axis=1).max(), 1.0)


@requires_mne()
@requires_torch()
def test_three_dimensional_electrode_positions_seed_the_model():
    from grfnet import GRFNet, channel_positions

    names = ["Fz", "C3", "Cz", "C4", "Pz", "O1", "O2", "F3"]
    model = GRFNet(len(names), n_classes=2, channel_coords=channel_positions(names, ndim=3))
    assert model.spatial.n_geometry_components > 5
