from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from grfnet.data import eegmmidb
from grfnet.data.eegmmidb import (
    EpochsData,
    clean_channel_name,
    load_epochs,
    parse_subjects,
    task_classes,
)

from conftest import requires_eegmmidb, requires_matplotlib, requires_mne

RAW_NAMES = ["Fc5.", "Fcz.", "C3..", "Cz..", "C4..", "Fp1.", "Fpz.", "Afz.", "T10.", "Iz.."]
NAMES = ["FC5", "FCz", "C3", "Cz", "C4", "Fp1", "Fpz", "AFz", "T10", "Iz"]
SFREQ = 160.0


def _fake_root(tmp_path, subjects=(1, 2), runs=range(1, 15)):
    for subject in subjects:
        for run in runs:
            path = eegmmidb.run_path(tmp_path, subject, run)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    return tmp_path


def _fake_reader(calls=None):
    """Runs of 20 s with cues every 4 s cycling T0, T1, T2; the last cue overruns the recording."""

    def read(path, sfreq, l_freq, h_freq):
        if calls is not None:
            calls.append(path.name)
        subject, run = int(path.name[1:4]), int(path.name[5:7])
        n = int(20 * sfreq)
        signal = np.full((len(NAMES), n), subject * 100 + run, dtype=np.float32)
        signal += np.arange(n, dtype=np.float32) / sfreq
        onsets = np.array([0.0, 4.0, 8.0, 12.0, 18.0])
        codes = np.array(["T0", "T1", "T2", "T1", "T2"])
        return signal, list(NAMES), onsets, codes

    return read


@pytest.mark.parametrize("raw, clean", list(zip(RAW_NAMES, NAMES, strict=True)))
def test_clean_channel_name(raw, clean):
    assert clean_channel_name(raw) == clean


@requires_mne()
def test_every_edf_label_is_in_the_standard_template():
    import mne

    labels = ["Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.", "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..",
              "C6..", "Cp5.", "Cp3.", "Cp1.", "Cpz.", "Cp2.", "Cp4.", "Cp6.", "Fp1.", "Fpz.", "Fp2.", "Af7.", "Af3.",
              "Afz.", "Af4.", "Af8.", "F7..", "F5..", "F3..", "F1..", "Fz..", "F2..", "F4..", "F6..", "F8..", "Ft7.",
              "Ft8.", "T7..", "T8..", "T9..", "T10.", "Tp7.", "Tp8.", "P7..", "P5..", "P3..", "P1..", "Pz..", "P2..",
              "P4..", "P6..", "P8..", "Po7.", "Po3.", "Poz.", "Po4.", "Po8.", "O1..", "Oz..", "O2..", "Iz.."]
    template = set(mne.channels.make_standard_montage("standard_1020").ch_names)
    assert {clean_channel_name(label) for label in labels} <= template


def test_parse_subjects():
    assert parse_subjects("1-3, 7,3") == [1, 2, 3, 7]
    assert parse_subjects([5, 2]) == [2, 5]
    assert parse_subjects(None) is None
    with pytest.raises(ValueError, match="1..109"):
        parse_subjects("0-2")


def test_task_classes():
    names, events = task_classes("imagery")
    assert names == ("left_fist", "right_fist", "both_fists", "both_feet")
    assert sorted(events) == [4, 6, 8, 10, 12, 14]
    names, events = task_classes("execution_lr", include_rest=True)
    assert names == ("left_fist", "right_fist", "rest")
    assert events[3] == {"T1": "left_fist", "T2": "right_fist", "T0": "rest"}
    # the shared task tables are not modified
    assert "T0" not in eegmmidb.TASKS["execution_lr"][3]
    with pytest.raises(ValueError, match="unknown task"):
        task_classes("sleep")


def test_missing_root_explains_the_variable(tmp_path):
    with pytest.raises(FileNotFoundError, match="GRFNET_EEGMMIDB"):
        load_epochs(tmp_path / "absent")


def test_default_root_reads_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(eegmmidb.ROOT_ENV, str(tmp_path))
    assert eegmmidb.default_root() == tmp_path


def test_load_epochs_shapes_labels_and_bounds(monkeypatch, tmp_path):
    monkeypatch.setattr(eegmmidb, "_read_run", _fake_reader())
    data = load_epochs(_fake_root(tmp_path), "imagery_lr", tmin=-1.0, tmax=3.0)
    # per run: T1, T2, T1 fit; the T2 at 18 s overruns; T0 is not a class
    assert data.X.shape == (2 * 3 * 3, len(NAMES), 640)
    assert data.X.dtype == np.float32
    assert data.class_names == ("left_fist", "right_fist")
    assert data.y.tolist()[:3] == [0, 1, 0]
    assert sorted(set(data.runs.tolist())) == [4, 8, 12]
    assert data.subjects.tolist() == [1] * 9 + [2] * 9
    assert np.isclose(data.times[0], -1.0) and len(data.times) == 640
    # first epoch of subject 1, run 4 starts 1 s before the cue at 4 s
    assert np.isclose(data.X[0, 0, 0], 104 + 3.0)


def test_include_rest_and_subject_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(eegmmidb, "_read_run", _fake_reader())
    data = load_epochs(_fake_root(tmp_path), "imagery_ff", subjects=[2], include_rest=True)
    assert data.class_names == ("both_fists", "both_feet", "rest")
    assert set(data.subjects.tolist()) == {2}
    assert np.bincount(data.y).tolist() == [6, 3, 3]


def test_channel_selection_and_subsets(monkeypatch, tmp_path):
    monkeypatch.setattr(eegmmidb, "_read_run", _fake_reader())
    data = load_epochs(_fake_root(tmp_path), channels=["c4", "C3.."])
    assert data.ch_names == ["C4", "C3"]
    assert data.X.shape[1] == 2
    subset = data.select(data.subjects == 1)
    assert len(subset) == 9 and isinstance(subset, EpochsData)
    with pytest.raises(ValueError, match="not found"):
        data.pick_channels(["O1"])


def test_cache_skips_reading(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr(eegmmidb, "_read_run", _fake_reader(calls))
    root = _fake_root(tmp_path / "data")
    first = load_epochs(root, cache_dir=tmp_path / "cache")
    n_reads = len(calls)
    second = load_epochs(root, cache_dir=tmp_path / "cache")
    assert len(calls) == n_reads == 6
    assert np.array_equal(first.X, second.X) and first.ch_names == second.ch_names
    # a different window is a different cache entry
    load_epochs(root, tmax=2.0, cache_dir=tmp_path / "cache")
    assert len(calls) == 12


def test_rejects_bad_window(tmp_path):
    with pytest.raises(ValueError, match="tmax must exceed tmin"):
        load_epochs(tmp_path, tmin=1.0, tmax=1.0)


def test_loader_explains_missing_mne(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mne", None)
    with pytest.raises(ImportError, match=r"grfnet\[eeg\]"):
        load_epochs(_fake_root(tmp_path, subjects=(1,)))


def _synthetic_epochs(n_per_class=12, seed=0):
    import mne

    rng = np.random.default_rng(seed)
    names = mne.channels.make_standard_montage("standard_1020").ch_names[:32]
    names = [n for n in names if n not in ("A1", "A2", "M1", "M2")]
    for focus in ("C3", "Cz", "C4"):
        if focus not in names:
            names.append(focus)
    n = 2 * n_per_class
    times = -1.0 + np.arange(400) / 80.0
    X = rng.standard_normal((n, len(names), 400)).astype(np.float32)
    X += (np.sin(2 * np.pi * 10 * times) * (times < 0)).astype(np.float32)
    y = np.repeat([0, 1], n_per_class)
    return EpochsData(X, y, np.tile([1, 2], n_per_class), np.full(n, 4), names, 80.0, -1.0,
                      ("left_fist", "right_fist"), {"dataset": "eegmmidb", "task": "imagery_lr"})


@requires_mne()
@requires_matplotlib()
def test_report_writes_figures_and_summary(tmp_path):
    from grfnet.data.report import write_report

    recordings = [{"subject": 1, "run": 4, "sfreq": 160.0, "duration_s": 125.0, "n_channels": 64,
                   "T0": 15, "T1": 8, "T2": 7}]
    summary = write_report(_synthetic_epochs(), tmp_path, recordings=recordings, dpi=40)
    for name in summary["figures"]:
        assert (tmp_path / name).stat().st_size > 0
    assert set(summary["figures"]) == {"montage.png", "class_balance.png", "trial_rms.png", "example_trials.png",
                                       "psd.png", "erd_ers.png", "topomaps.png"}
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["n_trials"] == 24 and saved["trials_per_class"] == {"left_fist": 12, "right_fist": 12}
    assert (tmp_path / "recordings.csv").read_text(encoding="utf-8").startswith("subject,run,sfreq")
    assert "![" in (tmp_path / "report.md").read_text(encoding="utf-8")


@requires_mne()
@requires_matplotlib()
def test_report_without_baseline_skips_baseline_figures(tmp_path):
    from grfnet.data.report import write_report

    data = _synthetic_epochs()
    data = EpochsData(data.X[..., 80:], data.y, data.subjects, data.runs, data.ch_names, data.sfreq, 0.0,
                      data.class_names, data.info)
    summary = write_report(data, tmp_path, dpi=40)
    assert "erd_ers.png" not in summary["figures"] and "topomaps.png" not in summary["figures"]


@requires_eegmmidb()
def test_real_subject_loads():
    data = load_epochs(subjects=[1], tmin=0.0, tmax=4.0)
    assert data.X.shape[1:] == (64, 640)
    assert set(data.y.tolist()) == {0, 1}
    assert data.ch_names[:3] == ["FC5", "FC3", "FC1"]


@requires_eegmmidb()
def test_128_hz_subject_is_resampled():
    data = load_epochs(subjects=[88], tmin=0.0, tmax=4.0)
    assert data.sfreq == 160.0
    assert data.X.shape[1:] == (64, 640)
