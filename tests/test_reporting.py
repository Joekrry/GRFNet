from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from conftest import fake_eegmmidb, requires_matplotlib, requires_mne, requires_torch, tiny_config

ALL_FIGURES = {"learning_curves.png", "confusion_matrix.png", "subject_scores.png", "stream_gates.png",
               "spatial_filters.png", "kernel_responses.png", "spatial_patterns.png", "embeddings.png",
               "saliency.png"}
DATA_FIGURES = {"spatial_patterns.png", "embeddings.png", "saliency.png"}


@pytest.fixture
def trained_run(monkeypatch, tmp_path):
    from grfnet.experiments.runner import run_experiment

    root = fake_eegmmidb(monkeypatch, tmp_path)
    run_experiment(tiny_config(root), tmp_path / "run", folds=[0, 2], progress=False, log=lambda *_: None)
    return tmp_path / "run"


@requires_torch()
@requires_mne()
@requires_matplotlib()
def test_full_report(trained_run, tmp_path):
    from grfnet.reporting import write_run_report

    out = tmp_path / "report"
    summary = write_run_report(trained_run, out, progress=False, log=lambda *_: None)
    for name in ALL_FIGURES:
        assert (out / name).stat().st_size > 0, name
    assert set(summary["figures"]) == ALL_FIGURES
    assert summary["skipped"] == {}
    assert summary["model_figures_fold"] == 0
    assert [f["fold"] for f in summary["folds"]] == [0, 2]

    # every number in the report comes from the run folder
    results = json.loads((trained_run / "results.json").read_text(encoding="utf-8"))
    assert summary["test"]["balanced_accuracy"] == results["test"]["balanced_accuracy"]
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert f"{results['test']['balanced_accuracy']:.4f}" in markdown
    for name in ALL_FIGURES:
        assert f"]({name})" in markdown

    efficiency = json.loads((out / "efficiency.json").read_text(encoding="utf-8"))
    assert efficiency["parameters"]["trainable"] > 0 and efficiency["parameters"]["frozen"] > 0
    assert efficiency["flops_per_trial"] > 0
    assert {t["device"] for t in efficiency["inference"]} >= {"cpu"}
    assert set(efficiency["training"]["fold_seconds"]) == {"0", "2"}
    json.loads((out / "report.json").read_text(encoding="utf-8"))


@requires_torch()
@requires_mne()
@requires_matplotlib()
def test_report_without_data_or_benchmark(trained_run, tmp_path):
    from grfnet.reporting import write_run_report

    out = tmp_path / "report"
    summary = write_run_report(trained_run, out, fold=2, data=False, bench=False, progress=False,
                               log=lambda *_: None)
    assert set(summary["figures"]) == ALL_FIGURES - DATA_FIGURES
    assert set(summary["skipped"]) == {"data analyses", "benchmark"}
    assert summary["model_figures_fold"] == 2
    assert not (out / "efficiency.json").exists()
    assert "## Skipped" in (out / "report.md").read_text(encoding="utf-8")


@requires_torch()
@requires_mne()
@requires_matplotlib()
def test_missing_dataset_is_skipped_not_fatal(trained_run, tmp_path):
    from grfnet.reporting import write_run_report

    summary = write_run_report(trained_run, tmp_path / "report", root=str(tmp_path / "moved"), bench=False,
                               progress=False, log=lambda *_: None)
    assert "GRFNET_EEGMMIDB" in summary["skipped"]["data analyses"]
    assert not DATA_FIGURES & set(summary["figures"])


@requires_torch()
@requires_mne()
@requires_matplotlib()
def test_report_rejects_unknown_fold(trained_run):
    from grfnet.reporting import write_run_report

    with pytest.raises(ValueError, match="fold 1 has no checkpoint"):
        write_run_report(trained_run, fold=1, progress=False, log=lambda *_: None)


def test_read_run_explains_a_wrong_folder(tmp_path):
    from grfnet.reporting import read_run

    with pytest.raises(FileNotFoundError, match="python -m grfnet.train"):
        read_run(tmp_path)


@requires_torch()
@requires_mne()
@requires_matplotlib()
def test_command_line(trained_run, tmp_path):
    from grfnet import report

    assert report.main([str(trained_run), "--no-benchmark", "--no-progress", "--dpi", "40"]) == 0
    assert (trained_run / "report" / "report.md").exists()


def test_spatial_patterns_recover_the_mixing_matrix():
    from grfnet.reporting.figures import spatial_patterns

    rng = np.random.default_rng(0)
    mixing = rng.standard_normal((5, 5))
    sources = rng.standard_normal((200, 5, 400))
    X = np.einsum("cs,nst->nct", mixing, sources)
    # the unmixing filters' patterns are the mixing matrix's columns
    patterns = spatial_patterns(np.linalg.inv(mixing), X)
    assert np.allclose(patterns, mixing.T, atol=0.1 * np.abs(mixing).max())


def test_dilated_kernel_places_taps():
    from grfnet.reporting.figures import _dilated

    assert _dilated(np.array([1.0, 2.0, 3.0]), 3).tolist() == [1, 0, 0, 2, 0, 0, 3]


@requires_torch()
def test_benchmark_parts():
    from grfnet import GRFNet
    from grfnet.reporting.benchmark import count_flops, count_parameters, time_inference, training_time

    model = GRFNet(4, n_classes=2, dilations=[1, 2])
    counts = count_parameters(model)
    assert counts["total"] == counts["trainable"] + counts["frozen"]
    assert counts["trainable"] == model.count_parameters()
    flops = count_flops(model, 4, 128)
    assert flops is not None and flops > 0
    timing = time_inference(model, 4, 128, batch_size=2, device="cpu", repeats=2, warmup=1)
    assert timing["median_ms"] > 0 and timing["trials_per_second"] > 0
    times = training_time({0: [{"seconds": 5.0}, {"seconds": 1.0}, {"seconds": 3.0}]})
    assert times["median_epoch_seconds"] == 2.0 and times["total_seconds"] == 9.0


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed")
def test_projection_uses_tsne_when_available():
    from grfnet.reporting.analysis import project_2d

    points, method = project_2d(np.random.default_rng(0).standard_normal((40, 6)))
    assert method == "t-SNE" and points.shape == (40, 2)


def test_projection_falls_back_to_pca(monkeypatch):
    from grfnet.reporting import analysis

    monkeypatch.setattr(analysis.importlib.util, "find_spec", lambda name: None)
    points, method = analysis.project_2d(np.random.default_rng(0).standard_normal((40, 6)))
    assert method == "PCA" and points.shape == (40, 2)
