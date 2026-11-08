"""Dataset report: figures and summary tables for epoched EEG. Requires the viz and eeg extras."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .eegmmidb import RESAMPLED_SUBJECTS, EpochsData

# Categorical slots, text and grid colours.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BANDS = {"mu (8-13 Hz)": (8.0, 13.0), "beta (13-30 Hz)": (13.0, 30.0)}
FOCUS_CHANNELS = ("C3", "Cz", "C4")
_CHUNK = 256


def write_report(epochs: EpochsData, out_dir: str | Path, *, recordings: Sequence[dict] | None = None,
                 dpi: int = 150) -> dict:
    """Write figures, summary.json, recordings.csv and report.md to out_dir. Returns the summary.

    epochs needs a pre-cue baseline (tmin < 0) for the ERD/ERS and topomap figures.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if len(epochs) == 0:
        raise ValueError("no trials to report on")
    focus = [name for name in FOCUS_CHANNELS if name.lower() in {n.lower() for n in epochs.ch_names}]

    figures: dict[str, str] = {}
    figures["montage.png"] = "Electrode positions on the standard 10-20 template."
    _save(_montage(epochs), out / "montage.png", dpi)
    figures["class_balance.png"] = "Trials per class for each subject."
    _save(_class_balance(epochs), out / "class_balance.png", dpi)
    figures["trial_rms.png"] = "Median trial RMS amplitude per subject."
    _save(_trial_rms(epochs), out / "trial_rms.png", dpi)
    if focus:
        figures["example_trials.png"] = f"One trial per class at {', '.join(focus)}."
        _save(_example_trials(epochs, focus), out / "example_trials.png", dpi)
        figures["psd.png"] = f"Mean power spectral density per class after the cue at {', '.join(focus)}."
        _save(_psd(epochs, focus), out / "psd.png", dpi)
    if epochs.tmin < 0 and focus:
        figures["erd_ers.png"] = "Band power relative to the pre-cue baseline, per class (ERD below 0, ERS above)."
        _save(_erd_ers(epochs, focus), out / "erd_ers.png", dpi)
    if epochs.tmin < 0:
        figures["topomaps.png"] = "Mu-band power change from the pre-cue baseline, per class."
        _save(_topomaps(epochs), out / "topomaps.png", dpi)

    summary = _summary(epochs, recordings)
    summary["figures"] = list(figures)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if recordings:
        with open(out / "recordings.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(recordings[0]))
            writer.writeheader()
            writer.writerows(recordings)
    (out / "report.md").write_text(_markdown(summary, figures), encoding="utf-8")
    return summary


def _summary(epochs: EpochsData, recordings: Sequence[dict] | None) -> dict:
    counts = {name: int(np.sum(epochs.y == i)) for i, name in enumerate(epochs.class_names)}
    subjects = sorted(int(s) for s in np.unique(epochs.subjects))
    per_subject = [int(np.sum(epochs.subjects == s)) for s in subjects]
    summary = {
        "dataset": epochs.info.get("dataset"),
        "task": epochs.info.get("task"),
        "n_subjects": len(subjects),
        "n_trials": len(epochs),
        "trials_per_class": counts,
        "trials_per_subject": {"min": min(per_subject), "median": float(np.median(per_subject)),
                               "max": max(per_subject)},
        "n_channels": len(epochs.ch_names),
        "n_times": int(epochs.X.shape[-1]),
        "sfreq": epochs.sfreq,
        "window_s": [epochs.tmin, epochs.tmin + epochs.X.shape[-1] / epochs.sfreq],
        "filter_hz": [epochs.info.get("l_freq"), epochs.info.get("h_freq")],
        "resampled_subjects": [s for s in subjects if s in RESAMPLED_SUBJECTS],
        "ch_names": list(epochs.ch_names),
    }
    if recordings:
        rates: dict[str, int] = {}
        for row in recordings:
            rates[str(row["sfreq"])] = rates.get(str(row["sfreq"]), 0) + 1
        summary["n_recordings"] = len(recordings)
        summary["recordings_per_sfreq"] = rates
        summary["recording_hours"] = round(sum(row["duration_s"] for row in recordings) / 3600, 2)
    return summary


def _markdown(summary: dict, figures: dict[str, str]) -> str:
    lines = [
        f"# Dataset report: {summary['dataset']} / {summary['task']}",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Subjects | {summary['n_subjects']} |",
        f"| Trials | {summary['n_trials']} |",
        *(f"| Trials: {name} | {n} |" for name, n in summary["trials_per_class"].items()),
        f"| Trials per subject (min / median / max) | {summary['trials_per_subject']['min']} / "
        f"{summary['trials_per_subject']['median']:g} / {summary['trials_per_subject']['max']} |",
        f"| Channels | {summary['n_channels']} |",
        f"| Sampling rate | {summary['sfreq']:g} Hz |",
        f"| Epoch window | {summary['window_s'][0]:g} s to {summary['window_s'][1]:g} s ({summary['n_times']} samples) |",
        f"| Resampled subjects | {', '.join(map(str, summary['resampled_subjects'])) or 'none'} |",
    ]
    if "n_recordings" in summary:
        lines.append(f"| EDF recordings | {summary['n_recordings']} ({summary['recording_hours']:g} h) |")
    lines.append("")
    for name, caption in figures.items():
        lines += [f"## {caption}", "", f"![{caption}]({name})", ""]
    return "\n".join(lines)


def _figure(nrows=1, ncols=1, width=8.0, height=4.0, **kwargs):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False, **kwargs)
    for ax in axes.flat:
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    return fig, axes


def _save(fig, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, facecolor="white")


def _title(ax, text: str) -> None:
    ax.set_title(text, color=TEXT, fontsize=10, loc="left")


def _label(ax, x: str | None = None, y: str | None = None) -> None:
    if x:
        ax.set_xlabel(x, color=MUTED, fontsize=9)
    if y:
        ax.set_ylabel(y, color=MUTED, fontsize=9)


def _legend(ax, **kwargs) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, **kwargs)


def _outside_legend(fig, ax) -> None:
    fig.legend(*ax.get_legend_handles_labels(), loc="outside right upper", frameon=False, fontsize=8,
               labelcolor=TEXT)


def _class_label(name: str) -> str:
    return name.replace("_", " ")


def _channel(epochs: EpochsData, name: str) -> int:
    return [n.lower() for n in epochs.ch_names].index(name.lower())


def _info(epochs: EpochsData):
    import mne

    info = mne.create_info(list(epochs.ch_names), epochs.sfreq, "eeg")
    info.set_montage("standard_1020", match_case=False)
    return info


def _montage(epochs: EpochsData):
    import mne

    fig, axes = _figure(width=6.0, height=6.0)
    ax = axes[0, 0]
    ax.grid(False)
    # Azimuthal projection keeps electrodes below the head's equator apart.
    mne.viz.plot_sensors(_info(epochs), kind="topomap", show_names=True, axes=ax, show=False,
                         pointsize=40, linewidth=0)
    for text in ax.texts:
        text.set_fontsize(6.5)
        text.set_color(TEXT)
    for collection in ax.collections:
        collection.set_color(SERIES[0])
    _title(ax, f"{len(epochs.ch_names)} electrodes (nose up)")
    return fig


def _class_balance(epochs: EpochsData):
    subjects = np.unique(epochs.subjects)
    fig, axes = _figure(width=max(8.0, 0.09 * len(subjects) + 2.0), height=3.6)
    ax = axes[0, 0]
    ax.grid(axis="x", visible=False)
    bottom = np.zeros(len(subjects))
    for i, name in enumerate(epochs.class_names):
        counts = np.array([np.sum((epochs.subjects == s) & (epochs.y == i)) for s in subjects])
        ax.bar(subjects, counts, bottom=bottom, width=0.8, color=SERIES[i % len(SERIES)],
               edgecolor="white", linewidth=0.5, label=_class_label(name))
        bottom += counts
    _label(ax, "Subject", "Trials")
    _title(ax, f"{len(epochs)} trials from {len(subjects)} subjects")
    _outside_legend(fig, ax)
    return fig


def _trial_rms(epochs: EpochsData):
    subjects = np.unique(epochs.subjects)
    rms = np.concatenate([np.sqrt(np.mean(epochs.X[i:i + _CHUNK].astype(np.float64) ** 2, axis=(1, 2)))
                          for i in range(0, len(epochs), _CHUNK)])
    median = np.array([np.median(rms[epochs.subjects == s]) for s in subjects])
    fig, axes = _figure(width=max(8.0, 0.09 * len(subjects) + 2.0), height=3.2)
    ax = axes[0, 0]
    ax.grid(axis="x", visible=False)
    resampled = np.isin(subjects, RESAMPLED_SUBJECTS)
    ax.bar(subjects[~resampled], median[~resampled], width=0.8, color=SERIES[0], label="recorded at 160 Hz")
    if resampled.any():
        ax.bar(subjects[resampled], median[resampled], width=0.8, color=SERIES[1], label="resampled from 128 Hz")
        _outside_legend(fig, ax)
    _label(ax, "Subject", "Median trial RMS (µV)")
    _title(ax, "Signal amplitude per subject")
    return fig


def _example_trials(epochs: EpochsData, focus: Sequence[str]):
    fig, axes = _figure(1, epochs.n_classes, width=3.2 * epochs.n_classes + 1.0, height=3.4, sharey=True)
    index = [_channel(epochs, name) for name in focus]
    times = epochs.times
    for c, ax in enumerate(axes[0]):
        trials = np.flatnonzero(epochs.y == c)
        _title(ax, _class_label(epochs.class_names[c]))
        if len(trials) == 0:
            continue
        trial = epochs.X[trials[0], index].astype(np.float64)
        spacing = 4.0 * float(np.std(epochs.X[:, index]) or 1.0)
        for k, name in enumerate(focus):
            ax.plot(times, trial[k] - k * spacing, color=SERIES[k], linewidth=1.0, label=name)
        ax.set_yticks([-k * spacing for k in range(len(focus))], list(focus))
        if epochs.tmin < 0:
            ax.axvline(0.0, color=MUTED, linewidth=0.8, linestyle="--")
        _label(ax, "Time from cue (s)")
    return fig


def _psd(epochs: EpochsData, focus: Sequence[str]):
    from scipy.signal import welch

    post = epochs.times >= 0
    index = [_channel(epochs, name) for name in focus]
    nperseg = int(min(epochs.sfreq, post.sum()))
    freqs, power = welch(epochs.X[:, index][:, :, post], fs=epochs.sfreq, nperseg=nperseg, axis=-1)
    keep = (freqs >= 1.0) & (freqs <= 40.0)
    db = 10 * np.log10(power[..., keep] + 1e-12)
    fig, axes = _figure(1, len(focus), width=3.4 * len(focus) + 0.5, height=3.4, sharey=True)
    for k, ax in enumerate(axes[0]):
        for c, name in enumerate(epochs.class_names):
            sel = db[epochs.y == c, k]
            if len(sel) == 0:
                continue
            mean, sem = sel.mean(0), sel.std(0) / np.sqrt(len(sel))
            colour = SERIES[c % len(SERIES)]
            ax.plot(freqs[keep], mean, color=colour, linewidth=1.5, label=_class_label(name))
            ax.fill_between(freqs[keep], mean - sem, mean + sem, color=colour, alpha=0.2, linewidth=0)
        for low, high in BANDS.values():
            ax.axvspan(low, high, color=GRID, alpha=0.35, linewidth=0)
        _title(ax, focus[k])
        _label(ax, "Frequency (Hz)", "Power (dB µV²/Hz)" if k == 0 else None)
    _legend(axes[0, -1], loc="upper right")
    return fig


def _band_power(X: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    return np.concatenate([sosfiltfilt(sos, X[i:i + _CHUNK].astype(np.float64), axis=-1) ** 2
                           for i in range(0, len(X), _CHUNK)])


def _relative_change(power: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Percent change from the mean over finite pre-cue samples."""
    baseline = np.nanmean(power[..., times < 0], axis=-1, keepdims=True)
    return 100.0 * (power - baseline) / np.maximum(baseline, 1e-12)


def _erd_ers(epochs: EpochsData, focus: Sequence[str]):
    index = [_channel(epochs, name) for name in focus]
    smooth = max(1, int(round(0.25 * epochs.sfreq)))
    kernel = np.ones(smooth) / smooth
    fig, axes = _figure(len(BANDS), len(focus), width=3.4 * len(focus) + 0.5, height=2.8 * len(BANDS),
                        sharex=True, sharey="row")
    for b, (band_name, band) in enumerate(BANDS.items()):
        power = _band_power(epochs.X[:, index], epochs.sfreq, band)
        for k, ax in enumerate(axes[b]):
            for c, name in enumerate(epochs.class_names):
                sel = power[epochs.y == c, k]
                if len(sel) == 0:
                    continue
                curve = np.convolve(sel.mean(0), kernel, mode="same")
                # Drop samples where the window or filter runs past the epoch.
                curve[:smooth] = curve[-smooth:] = np.nan
                change = _relative_change(curve, epochs.times)
                ax.plot(epochs.times, change, color=SERIES[c % len(SERIES)], linewidth=1.5,
                        label=_class_label(name))
            ax.axhline(0.0, color=MUTED, linewidth=0.8)
            ax.axvline(0.0, color=MUTED, linewidth=0.8, linestyle="--")
            _title(ax, f"{focus[k]}, {band_name}")
            _label(ax, "Time from cue (s)" if b == len(BANDS) - 1 else None,
                   "Change from baseline (%)" if k == 0 else None)
    _legend(axes[0, -1], loc="lower right")
    return fig


def _topomaps(epochs: EpochsData):
    import mne

    info = _info(epochs)
    # Per-class mean power, accumulated in chunks to bound memory.
    totals = np.zeros((epochs.n_classes, *epochs.X.shape[1:]))
    for i in range(0, len(epochs), _CHUNK):
        power = _band_power(epochs.X[i:i + _CHUNK], epochs.sfreq, BANDS["mu (8-13 Hz)"])
        np.add.at(totals, epochs.y[i:i + _CHUNK], power)
    counts = np.bincount(epochs.y, minlength=epochs.n_classes)
    # Skip 0.25 s at each epoch edge, where the band-pass filter is unreliable.
    times, edge = epochs.times, 0.25
    pre = (times < 0) & (times >= times[0] + edge)
    post = (times >= 0) & (times <= times[-1] - edge)
    maps = []
    for c in range(epochs.n_classes):
        if counts[c] == 0:
            maps.append(np.zeros(len(epochs.ch_names)))
            continue
        mean = totals[c] / counts[c]
        baseline = mean[:, pre if pre.any() else times < 0].mean(-1)
        maps.append(100.0 * (mean[:, post].mean(-1) - baseline) / np.maximum(baseline, 1e-12))
    limit = float(np.max(np.abs(maps))) or 1.0
    fig, axes = _figure(1, epochs.n_classes, width=2.8 * epochs.n_classes + 1.2, height=3.2)
    image = None
    for c, ax in enumerate(axes[0]):
        ax.grid(False)
        image, _ = mne.viz.plot_topomap(maps[c], info, axes=ax, show=False, cmap="RdBu_r",
                                        vlim=(-limit, limit), contours=0, sensors=True)
        _title(ax, _class_label(epochs.class_names[c]))
    bar = fig.colorbar(image, ax=list(axes[0]), shrink=0.8)
    bar.set_label("Mu power change (%)", color=MUTED, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    return fig
