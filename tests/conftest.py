"""Shared pytest helpers."""

from __future__ import annotations

import importlib.util

import pytest


def requires_mne():
    return pytest.mark.skipif(
        importlib.util.find_spec("mne") is None,
        reason="mne not installed",
    )


def requires_onnx():
    missing = [m for m in ("onnx", "onnxscript", "onnxruntime") if importlib.util.find_spec(m) is None]
    return pytest.mark.skipif(bool(missing), reason=f"not installed: {', '.join(missing)}")


def requires_matplotlib():
    return pytest.mark.skipif(
        importlib.util.find_spec("matplotlib") is None,
        reason="matplotlib not installed",
    )


def requires_eegmmidb():
    from grfnet.data.eegmmidb import default_root

    return pytest.mark.skipif(
        importlib.util.find_spec("mne") is None or not (default_root() / "S001" / "S001R04.edf").exists(),
        reason="EEGMMIDB not available (set GRFNET_EEGMMIDB) or mne not installed",
    )


def requires_torch():
    return pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="torch not installed",
    )


FAKE_CHANNELS = ["FC1", "C3", "Cz", "C4"]


def fake_eegmmidb(monkeypatch, tmp_path, n_subjects=6):
    """Patch the EDF reader with fake runs under tmp_path/data and return that root.

    Cues come every 4 s, and class 1 carries a 10 Hz rhythm at C3.
    """
    import numpy as np

    from grfnet.data import eegmmidb

    for subject in range(1, n_subjects + 1):
        for run in (4, 8, 12):
            path = eegmmidb.run_path(tmp_path / "data", subject, run)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def read(path, sfreq, l_freq, h_freq):
        subject, run = int(path.name[1:4]), int(path.name[5:7])
        rng = np.random.default_rng(subject * 100 + run)
        n = int(40 * sfreq)
        signal = rng.standard_normal((len(FAKE_CHANNELS), n)).astype(np.float32) * 10
        onsets = np.arange(0.0, 36.0, 4.0)
        codes = np.array(["T1", "T2"] * 5)[: len(onsets)]
        t = np.arange(int(2 * sfreq)) / sfreq
        for onset, code in zip(onsets, codes, strict=True):
            if code == "T2":
                start = int(onset * sfreq)
                signal[1, start:start + len(t)] += 20 * np.sin(2 * np.pi * 10 * t).astype(np.float32)
        return signal, list(FAKE_CHANNELS), onsets, codes

    monkeypatch.setattr(eegmmidb, "_read_run", read)
    return tmp_path / "data"


def tiny_config(root, **training):
    from grfnet.experiments.config import ExperimentConfig

    config = ExperimentConfig.from_dict({
        "name": "test",
        "data": {"root": str(root), "tmax": 2.0, "cache_dir": None},
        "split": {"n_folds": 3, "val_fraction": 0.25},
        "model": {"kwargs": {"dilations": [1, 2, 3, 5], "spatial_components": 4}},
        "training": {"epochs": 2, "batch_size": 16, "device": "cpu", "patience": None, **training},
    })
    return config.validate()
