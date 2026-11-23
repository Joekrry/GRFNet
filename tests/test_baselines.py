from __future__ import annotations

import sys

import pytest

from conftest import fake_eegmmidb, requires_braindecode, requires_matplotlib, requires_mne, tiny_config

BASELINES = ["EEGNet", "ShallowConvNet"]


def _model(name, **kwargs):
    import grfnet

    return getattr(grfnet, name)(n_channels=8, n_times=320, n_classes=3, **kwargs)


@requires_braindecode()
@pytest.mark.parametrize("name", BASELINES)
def test_baselines_return_logits(name):
    import torch

    torch.manual_seed(0)
    model = _model(name).eval()
    out = model(torch.randn(5, 8, 320))
    assert out.shape == (5, 3)
    assert not torch.allclose(out.exp().sum(1), torch.ones(5))  # raw scores, not probabilities
    assert model.count_parameters() == sum(p.numel() for p in model.parameters() if p.requires_grad)


@requires_braindecode()
@pytest.mark.parametrize("name", BASELINES)
def test_baselines_round_trip_through_a_checkpoint(name, tmp_path):
    import torch

    from grfnet import EEGNet, ShallowConvNet, load_model

    torch.manual_seed(0)
    model = _model(name, drop_prob=0.1).eval()
    model.save(tmp_path / "m.pt", metadata={"fold": 0})
    loaded = load_model(tmp_path / "m.pt")
    assert isinstance(loaded, (EEGNet, ShallowConvNet))
    assert type(loaded).__name__ == name
    assert loaded.get_config() == {"n_channels": 8, "n_times": 320, "n_classes": 3, "drop_prob": 0.1}
    x = torch.randn(4, 8, 320)
    assert torch.equal(model(x), loaded(x))
    other = ShallowConvNet if name == "EEGNet" else EEGNet
    with pytest.raises(ValueError, match="not a"):
        other.load(tmp_path / "m.pt")


@requires_braindecode()
def test_braindecode_argument_names_are_rejected():
    with pytest.raises(ValueError, match="n_classes"):
        _model("EEGNet", n_outputs=3)


def test_missing_braindecode_is_explained(monkeypatch):
    from grfnet import EEGNet

    monkeypatch.setitem(sys.modules, "braindecode.models", None)
    with pytest.raises(ImportError, match=r"grfnet\[baselines\]"):
        EEGNet(n_channels=8, n_times=320, n_classes=2)


def test_load_model_rejects_unknown_models(tmp_path):
    import torch

    from grfnet import load_model
    from grfnet.checkpoint import CHECKPOINT_FORMAT

    torch.save({"format": CHECKPOINT_FORMAT, "model": "Mystery", "config": {}, "state_dict": {}, "metadata": {}},
               tmp_path / "m.pt")
    with pytest.raises(ValueError, match="unknown model 'Mystery'"):
        load_model(tmp_path / "m.pt")


@requires_braindecode()
@pytest.mark.parametrize("name", BASELINES)
def test_baselines_train_and_reload_under_the_protocol(name, monkeypatch, tmp_path):
    from grfnet.experiments.runner import evaluate_checkpoint, run_experiment

    root = fake_eegmmidb(monkeypatch, tmp_path)
    config = tiny_config(root)
    config.model.name = name
    config.model.kwargs = {}
    summary = run_experiment(config, tmp_path / "run", folds=[0], log=lambda *_: None)
    assert summary["folds"][0]["n_parameters"] > 0
    result = evaluate_checkpoint(tmp_path / "run" / "fold_0" / "best.pt", device="cpu")
    assert result["test"] == summary["folds"][0]["test"]


@requires_braindecode()
def test_channel_names_are_grfnet_only(monkeypatch, tmp_path):
    from grfnet.experiments.runner import run_experiment

    root = fake_eegmmidb(monkeypatch, tmp_path)
    config = tiny_config(root)
    config.model.name = "EEGNet"
    config.model.kwargs = {}
    config.model.use_channel_names = True
    with pytest.raises(ValueError, match="only to GRFNet"):
        run_experiment(config, tmp_path / "run", folds=[0], log=lambda *_: None)


def test_baseline_configs_load_and_name_their_runs():
    from pathlib import Path

    from grfnet.experiments.config import load_config

    configs = Path(__file__).resolve().parents[1] / "configs"
    for name in BASELINES:
        path = configs / f"{name.lower()}_eegmmidb_imagery_lr.toml"
        config = load_config(path)
        assert config.model.name == name
        assert config.run_name == path.stem


@requires_braindecode()
@requires_mne()
@requires_matplotlib()
def test_report_skips_grfnet_figures_for_baselines(monkeypatch, tmp_path):
    from grfnet.experiments.runner import run_experiment
    from grfnet.reporting import write_run_report

    root = fake_eegmmidb(monkeypatch, tmp_path)
    config = tiny_config(root)
    config.model.name = "EEGNet"
    config.model.kwargs = {}
    run_experiment(config, tmp_path / "run", folds=[0], log=lambda *_: None)
    summary = write_run_report(tmp_path / "run", tmp_path / "report", progress=False, log=lambda *_: None)
    assert set(summary["figures"]) == {"learning_curves.png", "confusion_matrix.png", "subject_scores.png",
                                       "adaptation.png", "saliency.png"}
    assert "GRFNet figures" in summary["skipped"]
    assert summary["efficiency"]["flops_per_trial"] > 0
