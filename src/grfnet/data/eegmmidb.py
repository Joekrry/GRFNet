"""Loader for the PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .._progress import bar, use_bars

SFREQ = 160.0
N_SUBJECTS = 109
ROOT_ENV = "GRFNET_EEGMMIDB"
DEFAULT_ROOT = Path("src/dataset/files")
# Recorded at 128 Hz instead of 160 Hz.
RESAMPLED_SUBJECTS = (88, 92, 100)

RUN_DESCRIPTIONS = {
    1: "baseline, eyes open",
    2: "baseline, eyes closed",
    3: "executed left or right fist",
    4: "imagined left or right fist",
    5: "executed both fists or both feet",
    6: "imagined both fists or both feet",
    7: "executed left or right fist",
    8: "imagined left or right fist",
    9: "executed both fists or both feet",
    10: "imagined both fists or both feet",
    11: "executed left or right fist",
    12: "imagined left or right fist",
    13: "executed both fists or both feet",
    14: "imagined both fists or both feet",
}

_LR = {"T1": "left_fist", "T2": "right_fist"}
_FF = {"T1": "both_fists", "T2": "both_feet"}

# Task name -> {run: {annotation: class name}}.
TASKS: dict[str, dict[int, dict[str, str]]] = {
    "imagery_lr": {run: _LR for run in (4, 8, 12)},
    "imagery_ff": {run: _FF for run in (6, 10, 14)},
    "imagery": {**{run: _LR for run in (4, 8, 12)}, **{run: _FF for run in (6, 10, 14)}},
    "execution_lr": {run: _LR for run in (3, 7, 11)},
    "execution_ff": {run: _FF for run in (5, 9, 13)},
    "execution": {**{run: _LR for run in (3, 7, 11)}, **{run: _FF for run in (5, 9, 13)}},
}
REST = "rest"


@dataclass
class EpochsData:
    """Epoched trials with X of shape (n_trials, n_channels, n_times), in microvolts."""

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    runs: np.ndarray
    ch_names: list[str]
    sfreq: float
    tmin: float
    class_names: tuple[str, ...]
    info: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def times(self) -> np.ndarray:
        return self.tmin + np.arange(self.X.shape[-1]) / self.sfreq

    def select(self, mask: np.ndarray) -> EpochsData:
        return EpochsData(self.X[mask], self.y[mask], self.subjects[mask], self.runs[mask],
                          list(self.ch_names), self.sfreq, self.tmin, self.class_names, dict(self.info))

    def pick_channels(self, names: Sequence[str]) -> EpochsData:
        index = [_channel_index(self.ch_names, name) for name in names]
        return EpochsData(self.X[:, index], self.y, self.subjects, self.runs, [self.ch_names[i] for i in index],
                          self.sfreq, self.tmin, self.class_names, dict(self.info))


def clean_channel_name(name: str) -> str:
    """Convert an EDF label to 10-10 spelling, e.g. 'Fc5.' -> 'FC5', 'Fpz.' -> 'Fpz'."""
    label = name.strip().rstrip(".").upper()
    if label.endswith("Z"):
        label = label[:-1] + "z"
    if label.startswith("FP"):
        label = "Fp" + label[2:]
    return label


def default_root() -> Path:
    return Path(os.environ.get(ROOT_ENV, DEFAULT_ROOT))


def parse_subjects(spec: str | int | Iterable[int] | None) -> list[int] | None:
    """Parse '1-10,12', a number or an iterable into sorted subject numbers. ``None`` means all."""
    if spec is None:
        return None
    if isinstance(spec, int):
        spec = [spec]
    if isinstance(spec, str):
        subjects: list[int] = []
        for part in spec.replace(" ", "").split(","):
            if not part:
                continue
            first, _, last = part.partition("-")
            subjects.extend(range(int(first), int(last or first) + 1))
    else:
        subjects = [int(s) for s in spec]
    bad = [s for s in subjects if not 1 <= s <= N_SUBJECTS]
    if bad:
        raise ValueError(f"subjects must be in 1..{N_SUBJECTS}; got {bad}")
    return sorted(set(subjects))


def run_path(root: str | Path, subject: int, run: int) -> Path:
    return Path(root) / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf"


def list_subjects(root: str | Path | None = None) -> list[int]:
    root = _check_root(root)
    return sorted(int(p.name[1:]) for p in root.iterdir() if p.is_dir() and p.name[:1] == "S" and p.name[1:].isdigit())


def task_classes(task: str, include_rest: bool = False) -> tuple[tuple[str, ...], dict[int, dict[str, str]]]:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {sorted(TASKS)}")
    events = {run: dict(mapping) for run, mapping in TASKS[task].items()}
    names: list[str] = []
    for mapping in events.values():
        names.extend(n for n in mapping.values() if n not in names)
    if include_rest:
        for mapping in events.values():
            mapping["T0"] = REST
        names.append(REST)
    return tuple(names), events


def load_epochs(
    root: str | Path | None = None,
    task: str = "imagery_lr",
    subjects: str | Iterable[int] | None = None,
    *,
    tmin: float = 0.0,
    tmax: float = 4.0,
    sfreq: float = SFREQ,
    l_freq: float | None = None,
    h_freq: float | None = None,
    include_rest: bool = False,
    channels: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
    progress: bool | None = False,
) -> EpochsData:
    """Load cue-locked epochs for a task.

    Parameters
    ----------
    root : path, optional
        Folder holding S001 to S109. Defaults to ``$GRFNET_EEGMMIDB`` or ``src/dataset/files``.
    task : str
        A key of ``TASKS``, such as "imagery_lr".
    subjects : str or iterable of int, optional
        For example "1-10,12". Defaults to every subject found.
    tmin, tmax : float
        Window in seconds relative to the cue.
    sfreq : float
        Output sampling rate. Other rates are resampled.
    l_freq, h_freq : float, optional
        Band-pass edges, applied to the continuous run before resampling.
    include_rest : bool
        Add the T0 periods as a final "rest" class.
    channels : list of str, optional
        Channels to keep, in order.
    cache_dir : path, optional
        Per-subject cache, keyed on the options above.
    progress : bool, optional
        Show a progress bar. ``None`` shows one only in a terminal.

    Epochs that run past the end of a recording are dropped.
    """
    if tmax <= tmin:
        raise ValueError(f"tmax must exceed tmin; got tmin={tmin}, tmax={tmax}")
    root = _check_root(root)
    class_names, events = task_classes(task, include_rest)
    chosen = parse_subjects(subjects) or list_subjects(root)
    params = {"task": task, "tmin": tmin, "tmax": tmax, "sfreq": sfreq, "l_freq": l_freq,
              "h_freq": h_freq, "include_rest": include_rest}

    parts, ch_names = [], None
    bars = progress is not False and use_bars(progress)
    subject_bar = bar(chosen, enabled=bars, desc="loading subjects", unit="subject", leave=False)
    for i, subject in enumerate(subject_bar):
        part = _load_subject(root, subject, events, class_names, params, cache_dir)
        if ch_names is None:
            ch_names = part["ch_names"]
        elif part["ch_names"] != ch_names:
            raise ValueError(f"subject {subject} has different channels from subject {chosen[0]}")
        parts.append(part)
        subject_bar.set_postfix(subject=subject, trials=sum(len(p["y"]) for p in parts), refresh=False)
        if progress is not False and not bars:
            print(f"loaded subject {subject} ({i + 1}/{len(chosen)}): {len(part['y'])} trials", file=sys.stderr)

    n_times = int(round((tmax - tmin) * sfreq))
    data = EpochsData(
        X=np.concatenate([p["X"] for p in parts]) if parts else np.zeros((0, 0, n_times), np.float32),
        y=np.concatenate([p["y"] for p in parts]).astype(np.int64) if parts else np.zeros(0, np.int64),
        subjects=np.concatenate([p["subjects"] for p in parts]) if parts else np.zeros(0, np.int64),
        runs=np.concatenate([p["runs"] for p in parts]) if parts else np.zeros(0, np.int64),
        ch_names=list(ch_names or []),
        sfreq=float(sfreq),
        tmin=float(tmin),
        class_names=class_names,
        info={**params, "dataset": "eegmmidb"},
    )
    return data.pick_channels(channels) if channels is not None else data


def survey_recordings(root: str | Path | None = None, subjects: str | Iterable[int] | None = None, *,
                      progress: bool | None = False) -> list[dict]:
    """Sampling rate, duration and annotation counts of every EDF file, from the headers."""
    mne = _import_mne()
    root = _check_root(root)
    rows = []
    chosen = parse_subjects(subjects) or list_subjects(root)
    enabled = progress is not False and use_bars(progress)
    for subject in bar(chosen, enabled=enabled, desc="surveying recordings", unit="subject", leave=False):
        for run in RUN_DESCRIPTIONS:
            path = run_path(root, subject, run)
            if not path.exists():
                continue
            raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
            counts = {code: int(np.sum(raw.annotations.description == code)) for code in ("T0", "T1", "T2")}
            rows.append({"subject": subject, "run": run, "sfreq": float(raw.info["sfreq"]),
                         "duration_s": raw.n_times / raw.info["sfreq"], "n_channels": len(raw.ch_names), **counts})
    return rows


def _load_subject(root, subject, events, class_names, params, cache_dir) -> dict:
    cache_file = None
    if cache_dir is not None:
        key = hashlib.sha256(json.dumps({**params, "events": {str(k): v for k, v in events.items()}},
                                        sort_keys=True).encode()).hexdigest()[:16]
        cache_file = Path(cache_dir) / f"eegmmidb_S{subject:03d}_{key}.npz"
        if cache_file.exists():
            with np.load(cache_file) as f:
                return {"X": f["X"], "y": f["y"], "subjects": f["subjects"], "runs": f["runs"],
                        "ch_names": [str(n) for n in f["ch_names"]]}

    sfreq = params["sfreq"]
    n_times = int(round((params["tmax"] - params["tmin"]) * sfreq))
    X, y, runs, ch_names = [], [], [], None
    for run, mapping in sorted(events.items()):
        path = run_path(root, subject, run)
        if not path.exists():
            continue
        signal, names, onsets, descriptions = _read_run(path, sfreq, params["l_freq"], params["h_freq"])
        if ch_names is None:
            ch_names = names
        for onset, code in zip(onsets, descriptions, strict=True):
            if code not in mapping:
                continue
            start = int(round((onset + params["tmin"]) * sfreq))
            if start < 0 or start + n_times > signal.shape[1]:
                continue
            X.append(signal[:, start:start + n_times])
            y.append(class_names.index(mapping[code]))
            runs.append(run)
    if ch_names is None:
        raise FileNotFoundError(f"no EDF files for subject {subject} under {root}")
    part = {
        "X": np.stack(X).astype(np.float32) if X else np.zeros((0, len(ch_names), n_times), np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "subjects": np.full(len(y), subject, dtype=np.int64),
        "runs": np.asarray(runs, dtype=np.int64),
        "ch_names": ch_names,
    }
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, **{**part, "ch_names": np.asarray(ch_names)})
    return part


def _read_run(path: Path, sfreq: float, l_freq: float | None,
              h_freq: float | None) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Return the signal in microvolts, channel names, cue onsets in seconds and their codes."""
    mne = _import_mne()
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq, h_freq, verbose="error")
    if raw.info["sfreq"] != sfreq:
        raw.resample(sfreq, verbose="error")
    signal = (np.asarray(raw.get_data()) * 1e6).astype(np.float32)
    names = [clean_channel_name(n) for n in raw.ch_names]
    return signal, names, np.asarray(raw.annotations.onset), np.asarray(raw.annotations.description)


def _channel_index(ch_names: Sequence[str], name: str) -> int:
    lower = [n.lower() for n in ch_names]
    key = clean_channel_name(name).lower()
    if key not in lower:
        raise ValueError(f"channel {name!r} not found; available: {list(ch_names)}")
    return lower.index(key)


def _check_root(root: str | Path | None) -> Path:
    path = Path(root) if root is not None else default_root()
    if not path.is_dir():
        raise FileNotFoundError(
            f"EEGMMIDB folder not found: {path}. Pass root= or set {ROOT_ENV} to the folder holding S001 to S109."
        )
    return path


def _import_mne():
    try:
        import mne
    except ImportError:
        raise ImportError("the EEGMMIDB loader requires MNE-Python; install it with: pip install grfnet[eeg]") from None
    return mne
