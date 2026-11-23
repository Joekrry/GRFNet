from __future__ import annotations

import numpy as np
import pytest
import torch

from grfnet import FastGRFNet
from grfnet.experiments import episodes

from conftest import fake_eegmmidb, tiny_config


def _model(mode="unsupervised") -> FastGRFNet:
    torch.manual_seed(0)
    model = FastGRFNet(n_channels=4, n_times=64, n_classes=2, dilations=[1, 2], spatial_components=3,
                       write_mode=mode)
    if model.writer is not None:
        with torch.no_grad():
            model.writer.value.weight.normal_(0, 1.0)
    return model


def test_sequences_follow_run_then_data_order():
    subjects = np.array([2, 1, 2, 1, 2])
    runs = np.array([8, 4, 4, 4, 4])
    sequences = episodes.subject_sequences(subjects, runs)
    assert sequences[1].tolist() == [1, 3]
    assert sequences[2].tolist() == [2, 4, 0]
    assert episodes.trial_positions(subjects, runs).tolist() == [2, 0, 0, 1, 1]


def test_state_restarts_for_every_subject():
    model = _model().eval()
    X = np.random.default_rng(0).standard_normal((6, 4, 64)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1])
    subjects = np.array([1, 1, 1, 2, 2, 2])
    probs = episodes.predict_sequences(model, X, y, subjects, np.full(6, 4))
    with torch.no_grad():
        for first in (0, 3):  # each subject's first trial sees a zero state
            logits, _ = model(torch.from_numpy(X[first:first + 1]))
            np.testing.assert_allclose(probs[first], torch.softmax(logits, 1)[0].numpy(), rtol=1e-5, atol=1e-6)
        later, _ = model(torch.from_numpy(X[1:2]))
    assert not np.allclose(probs[1], torch.softmax(later, 1)[0].numpy())  # later trials carry state


def test_state_is_detached_between_chunks_only():
    model = _model()
    incoming = []
    forward = model.forward

    def recording(x, state=None, label=None):
        incoming.append(state is not None and state.grad_fn is not None)
        return forward(x, state, label)

    model.forward = recording  # type: ignore[method-assign]
    X, y = torch.randn(4, 4, 64), torch.tensor([0, 1, 0, 1])
    episodes.unroll(model, X, y, [np.arange(4)], device="cpu", chunk_length=2,
                    on_chunk=lambda loss, n: loss.backward())
    # Step 0 starts from the zero state; step 1 continues the chunk; step 2 starts a new chunk.
    assert incoming == [False, True, False, True]


def test_unroll_drops_finished_sequences():
    model = _model().eval()
    X, y = torch.randn(5, 4, 64), torch.tensor([0, 1, 0, 1, 0])
    with torch.no_grad():
        logits = episodes.unroll(model, X, y, [np.array([0, 1, 2]), np.array([3, 4])], device="cpu")
        alone = episodes.unroll(model, X, y, [np.array([3, 4])], device="cpu")
    torch.testing.assert_close(logits[3:], alone[3:])


def test_label_feedback_uses_labels_only_after_prediction():
    model = _model("label_feedback").eval()
    X = np.random.default_rng(1).standard_normal((4, 4, 64)).astype(np.float32)
    subjects, runs = np.ones(4, dtype=np.int64), np.full(4, 4)
    a = episodes.predict_sequences(model, X, np.array([0, 0, 0, 0]), subjects, runs)
    b = episodes.predict_sequences(model, X, np.array([1, 1, 1, 1]), subjects, runs)
    np.testing.assert_allclose(a[0], b[0])  # the first prediction precedes every label
    assert not np.allclose(a[1:], b[1:])


@pytest.mark.parametrize("mode", ["off", "unsupervised", "label_feedback"])
def test_episodic_training_runs_and_reloads(mode, monkeypatch, tmp_path):
    from grfnet.experiments.runner import evaluate_checkpoint, run_experiment

    root = fake_eegmmidb(monkeypatch, tmp_path)
    config = tiny_config(root)
    config.training.mode, config.training.chunk_length, config.training.subjects_per_batch = "episodic", 4, 2
    config.model.name = "FastGRFNet"
    config.model.kwargs = {**config.model.kwargs, "write_mode": mode}
    summary = run_experiment(config.validate(), tmp_path / "run", folds=[0], log=lambda *_: None)
    result = evaluate_checkpoint(tmp_path / "run" / "fold_0" / "best.pt", device="cpu")
    assert result["test"] == summary["folds"][0]["test"]
    header = (tmp_path / "run" / "fold_0" / "predictions.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("subject,run,position,")


@pytest.mark.parametrize("change, message", [
    (dict(model="FastGRFNet", mode="standard"), "trains only with"),
    (dict(model="GRFNet", mode="episodic"), "needs a stateful model"),
    (dict(model="FastGRFNet", mode="episodic", amp=True), "amp"),
    (dict(model="FastGRFNet", mode="episodic", chunk_length=0), "chunk_length"),
])
def test_episodic_settings_are_validated(change, message):
    from grfnet.experiments.config import ExperimentConfig

    training = {k: v for k, v in change.items() if k != "model"}
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.from_dict({"model": {"name": change["model"]}, "training": training}).validate()


def test_fastgrfnet_config_loads():
    from pathlib import Path

    from grfnet.experiments.config import load_config

    path = Path(__file__).resolve().parents[1] / "configs" / "fastgrfnet_eegmmidb_imagery_lr.toml"
    config = load_config(path)
    assert config.training.mode == "episodic" and config.model.kwargs["write_mode"] == "unsupervised"
    assert config.run_name == path.stem
