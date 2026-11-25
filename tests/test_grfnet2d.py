from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from grfnet import GRFNet2d

from conftest import fake_eegmmidb, requires_matplotlib, requires_mne, tiny_config

ARGS: dict[str, Any] = dict(n_channels=4, n_classes=3, sfreq=160.0, fmax=40.0, spatial_components=3)


def _model(**kwargs) -> GRFNet2d:
    torch.manual_seed(0)
    return GRFNet2d(**{**ARGS, **kwargs}).eval()


def test_spectrogram_keeps_bins_up_to_fmax():
    model = _model()
    spec = model.spectrogram(torch.randn(2, 4, 320))
    assert spec.shape == (2, 4, 17, 320 // 16 + 1)  # 0-40 Hz in 2.5 Hz bins; centred frames
    assert _model(fmax=None, sfreq=None).n_bins == 33


def test_spectrogram_finds_a_sinusoid():
    model = _model(fmax=None)
    t = torch.arange(640) / 160.0
    x = torch.sin(2 * math.pi * 10.0 * t).repeat(1, 4, 1)
    peak = model.spectrogram(x).mean(dim=-1)[0, 0].argmax().item()
    assert peak * 160.0 / 64 == pytest.approx(10.0)


def test_output_shape_and_any_length():
    model = _model()
    assert model(torch.randn(5, 4, 320)).shape == (5, 3)
    assert model(torch.randn(1, 4, 101)).shape == (1, 3)
    assert model.forward_features(torch.randn(2, 4, 320)).shape == (2, 3 * model.n_features)


def test_frozen_kernels_are_mean_centred_and_untrained():
    model = _model()
    for conv in model.fixed:
        assert isinstance(conv, torch.nn.Conv2d) and not conv.weight.requires_grad
        torch.testing.assert_close(conv.weight.mean(dim=(-2, -1)), torch.zeros(conv.weight.shape[:2]),
                                   atol=1e-6, rtol=0)
    assert model.learnable is not None and all(c.weight.requires_grad for c in model.learnable)


def test_bank_seed_fixes_the_frozen_kernels():
    a, b = _model(bank_seed=3), _model(bank_seed=3)
    torch.manual_seed(99)
    c = GRFNet2d(**ARGS, bank_seed=3)
    for x, y, z in zip(a.fixed, b.fixed, c.fixed, strict=True):
        assert isinstance(x, torch.nn.Conv2d) and isinstance(y, torch.nn.Conv2d) and isinstance(z, torch.nn.Conv2d)
        assert torch.equal(x.weight, y.weight) and torch.equal(x.weight, z.weight)


def test_geometry_seeding_gives_centred_unit_filters():
    model = GRFNet2d(n_channels=4, n_classes=2, channel_coords=[[-1, 0], [0, 0], [1, 0], [0, 1]])
    weight = model.spatial.weight[:, :, 0, 0].detach()
    torch.testing.assert_close(weight.sum(dim=1), torch.zeros(weight.shape[0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(weight.norm(dim=1), torch.ones(weight.shape[0]), atol=1e-5, rtol=0)


@pytest.mark.parametrize("bad, message", [
    (dict(fmax=40.0, sfreq=None), "fmax needs sfreq"),
    (dict(kernel_size=(2, 5)), "odd"),
    (dict(dilations=[0]), "dilations"),
    (dict(n_classes=None), "n_classes"),
])
def test_bad_arguments_are_rejected(bad, message):
    with pytest.raises(ValueError, match=message):
        GRFNet2d(**{**ARGS, **bad})


def test_short_inputs_are_rejected():
    with pytest.raises(ValueError, match="n_fft"):
        _model()(torch.randn(1, 4, 20))


def test_checkpoint_round_trip(tmp_path):
    from grfnet import load_model

    model = _model(kernel_size=(3, 3), dilations=[1, 3])
    model.save(tmp_path / "m.pt")
    loaded = load_model(tmp_path / "m.pt")
    assert isinstance(loaded, GRFNet2d) and loaded.get_config()["kernel_size"] == [3, 3]
    x = torch.randn(2, 4, 320)
    torch.testing.assert_close(model(x), loaded(x))


def test_trains_and_reloads_with_the_data_sampling_rate(monkeypatch, tmp_path):
    from grfnet.experiments.runner import evaluate_checkpoint, run_experiment

    root = fake_eegmmidb(monkeypatch, tmp_path)
    config = tiny_config(root)
    config.model.name = "GRFNet2d"
    config.model.kwargs = {"fmax": 40.0, "dilations": [1, 2], "spatial_components": 4}
    summary = run_experiment(config, tmp_path / "run", folds=[0], log=lambda *_: None)
    result = evaluate_checkpoint(tmp_path / "run" / "fold_0" / "best.pt", device="cpu")
    assert result["test"] == summary["folds"][0]["test"]
    loaded = GRFNet2d.load(tmp_path / "run" / "fold_0" / "best.pt")
    assert loaded.get_config()["sfreq"] == 160.0


@requires_mne()
@requires_matplotlib()
def test_report_skips_one_dimensional_figures(monkeypatch, tmp_path):
    from grfnet.experiments.runner import run_experiment
    from grfnet.reporting import write_run_report

    root = fake_eegmmidb(monkeypatch, tmp_path)
    config = tiny_config(root)
    config.model.name = "GRFNet2d"
    config.model.kwargs = {"fmax": 40.0, "dilations": [1, 2], "spatial_components": 4}
    run_experiment(config, tmp_path / "run", folds=[0], log=lambda *_: None)
    summary = write_run_report(tmp_path / "run", tmp_path / "report", progress=False, log=lambda *_: None)
    figures = set(summary["figures"])
    assert {"stream_gates.png", "spatial_filters.png", "embeddings.png", "saliency.png"} <= figures
    assert not {"kernel_responses.png", "spatial_patterns.png"} & figures
    assert "kernel responses and activation patterns" in summary["skipped"]


def test_config_loads():
    from pathlib import Path

    from grfnet.experiments.config import load_config

    path = Path(__file__).resolve().parents[1] / "configs" / "grfnet2d_eegmmidb_imagery_lr.toml"
    config = load_config(path)
    assert config.model.name == "GRFNet2d" and config.run_name == path.stem
