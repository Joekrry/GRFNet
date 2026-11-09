from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from grfnet.data import eegmmidb
from grfnet.experiments import metrics
from grfnet.experiments.config import ExperimentConfig, apply_override, load_config, to_toml
from grfnet.experiments.splits import check_folds, load_splits, save_splits, subject_kfold

from conftest import requires_torch

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs" / "eegmmidb_imagery_lr.toml"
NAMES = ["FC1", "C3", "Cz", "C4"]
requires_scipy = pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="scipy not installed")


# ---- config ----------------------------------------------------------------

def test_baseline_config_loads_with_distinct_dilations():
    config = load_config(BASELINE)
    assert config.data.task == "imagery_lr"
    assert config.model.kwargs["dilations"] == [1, 2, 3, 5]
    assert config.split.n_folds == 5


def test_overrides_parse_as_toml_values(tmp_path):
    config = load_config(BASELINE, ["training.epochs=3", "data.subjects=\"1-4\"", "split.folds=[0, 2]",
                                    "model.kwargs.dropout=0.1", "name=quick"])
    assert config.training.epochs == 3
    assert config.data.subjects == "1-4"
    assert config.split.folds == [0, 2]
    assert config.model.kwargs["dropout"] == 0.1
    assert config.name == "quick"
    # text that is not valid TOML is kept as a string
    assert load_config(BASELINE, ["data.subjects=1-10"]).data.subjects == "1-10"


def test_override_needs_key_and_value():
    with pytest.raises(ValueError, match="section.key=value"):
        apply_override({}, "training.epochs")


def test_unknown_keys_and_bad_values_are_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[training]\nepoch = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys in training"):
        load_config(path)
    path.write_text("[split]\nn_folds = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="n_folds"):
        load_config(path)
    path.write_text("[data]\nnormalize = \"trial\"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="normalize"):
        load_config(path)


def test_optional_fields_default_to_none():
    # TOML has no null; to_toml omits None, which must reload as the default.
    import dataclasses
    import typing

    from grfnet.experiments import config as module

    for cls in (module.DataConfig, module.SplitConfig, module.TrainingConfig):
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            if type(None) in typing.get_args(hints[f.name]):
                assert f.default is None, f"{cls.__name__}.{f.name}"


def test_toml_round_trip(tmp_path):
    config = load_config(BASELINE, ["data.channels=[\"C3\", \"C4\"]"])
    path = tmp_path / "copy.toml"
    path.write_text(to_toml(config), encoding="utf-8")
    assert load_config(path) == config


# ---- splits ----------------------------------------------------------------

def test_subject_kfold_is_disjoint_and_covers_every_subject():
    subjects = list(range(1, 24))
    folds = subject_kfold(subjects, 5, val_fraction=0.1, seed=3)
    check_folds(folds)
    tested = sorted(s for fold in folds for s in fold["test"])
    assert tested == subjects
    for fold in folds:
        assert sorted(fold["train"] + fold["val"] + fold["test"]) == subjects
        assert len(fold["val"]) >= 1
    assert folds == subject_kfold(subjects, 5, val_fraction=0.1, seed=3)
    assert folds != subject_kfold(subjects, 5, val_fraction=0.1, seed=4)


def test_subject_kfold_without_validation():
    folds = subject_kfold(range(1, 7), 3, val_fraction=0.0)
    assert all(fold["val"] == [] for fold in folds)


def test_overlapping_folds_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="shares subjects between train and test"):
        check_folds([{"fold": 0, "train": [1, 2], "val": [], "test": [2, 3]}])
    with pytest.raises(ValueError, match="n_folds"):
        subject_kfold([1, 2], 3)


def test_splits_file_round_trip(tmp_path):
    folds = subject_kfold(range(1, 11), 2)
    save_splits(tmp_path / "splits.json", folds, seed=0)
    assert load_splits(tmp_path / "splits.json") == folds


# ---- metrics ---------------------------------------------------------------

def test_metrics_on_known_values():
    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 0]
    assert metrics.confusion_matrix(y_true, y_pred, 2).tolist() == [[3, 1], [1, 1]]
    assert metrics.balanced_accuracy(y_true, y_pred, 2) == pytest.approx((3 / 4 + 1 / 2) / 2)
    # observed 4/6; expected (4*4 + 2*2) / 36
    assert metrics.cohen_kappa(y_true, y_pred, 2) == pytest.approx((4 / 6 - 20 / 36) / (1 - 20 / 36))
    assert metrics.macro_f1(y_true, y_pred, 2) == pytest.approx((0.75 + 0.5) / 2)


def test_balanced_accuracy_ignores_absent_classes():
    assert metrics.balanced_accuracy([0, 0, 1], [0, 0, 2], 3) == pytest.approx(0.5)


def test_per_subject_scores():
    scores = metrics.per_subject_scores([1, 1, 2, 2], [0, 1, 0, 1], [0, 1, 1, 1], 2)
    assert scores[1]["balanced_accuracy"] == 1.0
    assert scores[2]["accuracy"] == 0.5


@requires_scipy
def test_paired_wilcoxon_matches_scipy():
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 0.9, 20)
    b = a - rng.uniform(-0.02, 0.08, 20)
    result = metrics.paired_wilcoxon(a, b)
    _, p_value = wilcoxon(a, b)
    assert result["p_value"] == pytest.approx(p_value)
    assert result["n_wins_a"] + result["n_wins_b"] + result["n_ties"] == 20
    assert metrics.paired_wilcoxon(a, a)["p_value"] == 1.0


def test_paired_wilcoxon_explains_missing_scipy(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    with pytest.raises(ImportError, match=r"grfnet\[eval\]"):
        metrics.paired_wilcoxon([0.1, 0.2], [0.2, 0.1])


# ---- runner ----------------------------------------------------------------

def _fake_dataset(monkeypatch, tmp_path, n_subjects=6):
    """Fake EDF runs: cues every 4 s; class 1 carries a 10 Hz rhythm at C3."""
    for subject in range(1, n_subjects + 1):
        for run in (4, 8, 12):
            path = eegmmidb.run_path(tmp_path / "data", subject, run)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def read(path, sfreq, l_freq, h_freq):
        subject, run = int(path.name[1:4]), int(path.name[5:7])
        rng = np.random.default_rng(subject * 100 + run)
        n = int(40 * sfreq)
        signal = rng.standard_normal((len(NAMES), n)).astype(np.float32) * 10
        onsets = np.arange(0.0, 36.0, 4.0)
        codes = np.array(["T1", "T2"] * 5)[: len(onsets)]
        t = np.arange(int(2 * sfreq)) / sfreq
        for onset, code in zip(onsets, codes, strict=True):
            if code == "T2":
                start = int(onset * sfreq)
                signal[1, start:start + len(t)] += 20 * np.sin(2 * np.pi * 10 * t).astype(np.float32)
        return signal, list(NAMES), onsets, codes

    monkeypatch.setattr(eegmmidb, "_read_run", read)
    return tmp_path / "data"


def _config(root, **training):
    config = ExperimentConfig.from_dict({
        "name": "test",
        "data": {"root": str(root), "tmax": 2.0, "cache_dir": None},
        "split": {"n_folds": 3, "val_fraction": 0.25},
        "model": {"kwargs": {"dilations": [1, 2, 3, 5], "spatial_components": 4}},
        "training": {"epochs": 2, "batch_size": 16, "device": "cpu", "patience": None, **training},
    })
    return config.validate()


@requires_torch()
def test_run_experiment_writes_every_output(monkeypatch, tmp_path):
    from grfnet.experiments.runner import run_experiment

    root = _fake_dataset(monkeypatch, tmp_path)
    out = tmp_path / "run"
    summary = run_experiment(_config(root), out, log=lambda *_: None)
    for name in ("config.toml", "environment.json", "splits.json", "results.json", "per_subject.csv"):
        assert (out / name).exists(), name
    for k in range(3):
        for name in ("history.csv", "best.pt", "metrics.json", "predictions.csv"):
            assert (out / f"fold_{k}" / name).exists(), name
    assert sorted(summary["per_subject"], key=int) == [str(s) for s in range(1, 7)]
    assert summary["test"]["n"] == sum(fold["test"]["n"] for fold in summary["folds"])
    # the saved config reloads to the same experiment
    assert load_config(out / "config.toml") == _config(root)
    json.loads((out / "results.json").read_text(encoding="utf-8"))


@requires_torch()
def test_run_experiment_is_reproducible_and_refuses_to_overwrite(monkeypatch, tmp_path):
    from grfnet.experiments.runner import run_experiment

    root = _fake_dataset(monkeypatch, tmp_path)
    a = run_experiment(_config(root), tmp_path / "a", folds=[1], log=lambda *_: None)
    b = run_experiment(_config(root), tmp_path / "b", folds=[1], log=lambda *_: None)
    assert a["folds"][0]["test"] == b["folds"][0]["test"]
    assert [f["fold"] for f in a["folds"]] == [1]
    with pytest.raises(FileExistsError, match="overwrite"):
        run_experiment(_config(root), tmp_path / "a", log=lambda *_: None)
    run_experiment(_config(root), tmp_path / "a", folds=[0], overwrite=True, log=lambda *_: None)


@requires_torch()
def test_training_learns_the_planted_rhythm(monkeypatch, tmp_path):
    from grfnet.experiments.runner import run_experiment

    root = _fake_dataset(monkeypatch, tmp_path)
    summary = run_experiment(_config(root, epochs=15, lr=0.01), tmp_path / "run", folds=[0], log=lambda *_: None)
    assert summary["folds"][0]["test"]["balanced_accuracy"] > 0.8


@requires_torch()
def test_evaluate_checkpoint_reproduces_fold_metrics(monkeypatch, tmp_path):
    from grfnet.experiments.runner import evaluate_checkpoint, run_experiment

    root = _fake_dataset(monkeypatch, tmp_path)
    summary = run_experiment(_config(root), tmp_path / "run", folds=[0], log=lambda *_: None)
    result = evaluate_checkpoint(tmp_path / "run" / "fold_0" / "best.pt", out_dir=tmp_path / "eval", device="cpu")
    assert result["test"] == summary["folds"][0]["test"]
    assert (tmp_path / "eval" / "predictions.csv").exists()
    other = evaluate_checkpoint(tmp_path / "run" / "fold_0" / "best.pt", subjects=[1, 2], device="cpu")
    assert other["subjects"] == [1, 2]


@requires_torch()
@requires_scipy
def test_compare_runs_pairs_subjects(monkeypatch, tmp_path):
    from grfnet.experiments.runner import compare_runs, run_experiment

    root = _fake_dataset(monkeypatch, tmp_path)
    run_experiment(_config(root), tmp_path / "a", log=lambda *_: None)
    run_experiment(_config(root, lr=0.01), tmp_path / "b", log=lambda *_: None)
    result = compare_runs(tmp_path / "a", tmp_path / "b")
    assert result["n"] == 6
    assert 0.0 <= result["p_value"] <= 1.0


@requires_torch()
def test_command_line_entry_points(monkeypatch, tmp_path, capsys):
    from grfnet import evaluate, train

    root = _fake_dataset(monkeypatch, tmp_path)
    config = tmp_path / "config.toml"
    config.write_text(to_toml(_config(root)), encoding="utf-8")
    assert train.main([str(config), "--out", str(tmp_path / "run"), "--folds", "2", "--set", "training.epochs=1"]) == 0
    assert (tmp_path / "run" / "fold_2" / "best.pt").exists()
    capsys.readouterr()
    assert evaluate.main(["--checkpoint", str(tmp_path / "run" / "fold_2" / "best.pt"), "--device", "cpu"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["fold"] == 2


def test_unknown_dataset_and_model_are_rejected():
    from grfnet.data import EpochsData
    from grfnet.experiments.runner import build_model, load_dataset

    config = ExperimentConfig.from_dict({"data": {"dataset": "bci_iv_2a"}, "model": {"name": "EEGNet"}})
    data = EpochsData(np.zeros((2, 2, 64), np.float32), np.array([0, 1]), np.array([1, 1]), np.array([4, 4]),
                      ["C3", "C4"], 160.0, 0.0, ("left_fist", "right_fist"))
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset(config.data)
    with pytest.raises(ValueError, match="unknown model"):
        build_model(config.model, data)
