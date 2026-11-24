from __future__ import annotations

from typing import Any

import pytest
import torch

from grfnet import GRFNet, GRFNetSeq

from conftest import fake_eegmmidb, requires_matplotlib, requires_mne, tiny_config

ARGS: dict[str, Any] = dict(n_channels=5, n_classes=3, dilations=[1, 2, 3], spatial_components=4, window=16)


def _model(**kwargs) -> GRFNetSeq:
    torch.manual_seed(0)
    return GRFNetSeq(**{**ARGS, **kwargs}).eval()


def test_predicts_at_every_time_step():
    model = _model()
    assert model(torch.randn(2, 5, 120)).shape == (2, 3, 120)
    assert model(torch.randn(1, 5, 7)).shape == (1, 3, 7)


def test_outputs_never_depend_on_later_samples():
    model = _model()
    x = torch.randn(2, 5, 120)
    changed = x.clone()
    changed[..., 60:] = torch.randn(2, 5, 60) * 10
    torch.testing.assert_close(model(x)[..., :60], model(changed)[..., :60])
    assert not torch.allclose(model(x)[..., 60:], model(changed)[..., 60:])


@pytest.mark.parametrize("chunk", [1, 5, 17, 64, 200])
def test_streaming_in_chunks_matches_one_pass(chunk):
    model = _model()
    x = torch.randn(2, 5, 200)
    buffer, outputs = None, []
    for start in range(0, 200, chunk):
        logits, buffer = model.stream(x[..., start:start + chunk], buffer)
        outputs.append(logits)
    torch.testing.assert_close(torch.cat(outputs, dim=-1), model(x), atol=1e-5, rtol=1e-5)
    assert buffer is not None and buffer.shape[-1] == model.context


def test_context_covers_the_bank_and_the_window():
    model = _model()
    assert model.receptive_field == 3 * (9 - 1) + 1
    assert model.context == model.receptive_field - 1 + 16 - 1


def test_frozen_kernels_and_parameter_count_match_grfnet():
    seq = _model(dilations=[1, 2, 3, 5], window=64, spatial_components=16, n_channels=64, n_classes=2)
    plain = GRFNet(n_channels=64, n_classes=2, dilations=[1, 2, 3, 5])
    assert seq.count_parameters() == plain.count_parameters()
    assert all(not conv.weight.requires_grad for conv in seq.bank.fixed)


def test_checkpoint_round_trip(tmp_path):
    from grfnet import load_model

    model = _model(window=8)
    model.save(tmp_path / "seq.pt")
    loaded = load_model(tmp_path / "seq.pt")
    assert isinstance(loaded, GRFNetSeq) and loaded.window == 8
    x = torch.randn(2, 5, 50)
    torch.testing.assert_close(model(x), loaded(x))


@pytest.mark.parametrize("bad", [dict(window=0), dict(n_classes=None), dict(sfreq=160.0)])
def test_bad_arguments_are_rejected(bad):
    with pytest.raises(ValueError):
        GRFNetSeq(**{**ARGS, **bad})


def test_trains_on_every_time_step_and_reloads(monkeypatch, tmp_path):
    from grfnet.experiments.runner import evaluate_checkpoint, run_experiment

    root = fake_eegmmidb(monkeypatch, tmp_path)
    config = tiny_config(root)
    config.model.name = "GRFNetSeq"
    config.model.kwargs = {"dilations": [1, 2, 3, 5], "spatial_components": 4, "window": 32}
    summary = run_experiment(config, tmp_path / "run", folds=[0], log=lambda *_: None)
    result = evaluate_checkpoint(tmp_path / "run" / "fold_0" / "best.pt", device="cpu")
    assert result["test"] == summary["folds"][0]["test"]


@requires_mne()
@requires_matplotlib()
def test_report_adds_the_time_course(monkeypatch, tmp_path):
    from grfnet.experiments.runner import run_experiment
    from grfnet.reporting import write_run_report

    root = fake_eegmmidb(monkeypatch, tmp_path)
    config = tiny_config(root)
    config.model.name = "GRFNetSeq"
    config.model.kwargs = {"dilations": [1, 2], "spatial_components": 4, "window": 32}
    run_experiment(config, tmp_path / "run", folds=[0], log=lambda *_: None)
    summary = write_run_report(tmp_path / "run", tmp_path / "report", progress=False, log=lambda *_: None)
    assert {"time_course.png", "stream_gates.png", "kernel_responses.png", "spatial_patterns.png"} <= set(
        summary["figures"])
    assert "embeddings.png" not in summary["figures"]
    assert 0.0 <= summary["analyses"]["time_course"]["final_accuracy"] <= 1.0


def test_config_loads():
    from pathlib import Path

    from grfnet.experiments.config import load_config

    path = Path(__file__).resolve().parents[1] / "configs" / "grfnetseq_eegmmidb_imagery_lr.toml"
    config = load_config(path)
    assert config.model.name == "GRFNetSeq" and config.run_name == path.stem
