"""Figures for a trained run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from .._plotting import GRID, MUTED, NEUTRAL, SERIES, TEXT, class_label, eeg_info, figure, label, legend, title

BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}


def learning_curves(histories: Mapping[int, Sequence[dict]], best_epochs: Mapping[int, int]):
    panels = [("train_loss", "Training loss"), ("val_loss", "Validation loss"),
              ("train_accuracy", "Training accuracy"), ("val_balanced_accuracy", "Validation balanced accuracy")]
    fig, axes = figure(2, 2, width=10.0, height=6.5, sharex=True)
    for ax, (key, name) in zip(axes.flat, panels, strict=True):
        for fold, rows in histories.items():
            values = [r.get(key) for r in rows]
            if all(v is None for v in values):
                continue
            epochs = [r["epoch"] for r in rows]
            colour = SERIES[fold % len(SERIES)]
            ax.plot(epochs, values, color=colour, linewidth=1.5, label=f"fold {fold}")
            best = best_epochs.get(fold)
            if best is not None and 1 <= best <= len(values) and values[best - 1] is not None:
                ax.plot([best], [values[best - 1]], "o", color=colour, markersize=6, markeredgecolor="white",
                        markeredgewidth=1.5, zorder=3)
        title(ax, name)
    for ax in axes[1]:
        label(ax, "Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside right upper", frameon=False, fontsize=8, labelcolor=TEXT,
                   title="Dots: selected epoch", title_fontsize=8)
    return fig


def confusion_matrix(matrix: Sequence[Sequence[int]], class_names: Sequence[str]):
    counts = np.asarray(matrix, dtype=np.int64)
    recall = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    n = len(class_names)
    fig, axes = figure(width=1.4 * n + 2.6, height=1.3 * n + 1.6)
    ax = axes[0, 0]
    ax.grid(False)
    image = ax.imshow(recall, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(n):
        for j in range(n):
            colour = "white" if recall[i, j] > 0.55 else TEXT
            ax.text(j, i, f"{counts[i, j]}\n{recall[i, j]:.1%}", ha="center", va="center", fontsize=8, color=colour)
    names = [class_label(c) for c in class_names]
    ax.set_xticks(range(n), names)
    ax.set_yticks(range(n), names)
    label(ax, "Predicted class", "True class")
    title(ax, "Pooled test predictions")
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.set_label("Share of true class", color=MUTED, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    return fig


def subject_scores(per_subject: Mapping[str, dict], subject_fold: Mapping[int, int], fold_scores: Mapping[int, float],
                   n_classes: int):
    subjects = sorted(per_subject, key=lambda s: per_subject[s]["balanced_accuracy"])
    scores = np.array([per_subject[s]["balanced_accuracy"] for s in subjects])
    chance = 1.0 / n_classes
    fig, axes = figure(1, 2, width=12.0, height=3.8, gridspec_kw={"width_ratios": [4, 1]}, sharey=True)
    ax = axes[0, 0]
    ax.grid(axis="x", visible=False)
    folds = sorted(set(subject_fold.values()))
    for fold in folds:
        index = [i for i, s in enumerate(subjects) if subject_fold.get(int(s)) == fold]
        ax.bar(index, scores[index], width=0.8, color=SERIES[fold % len(SERIES)], label=f"fold {fold}")
    ax.axhline(scores.mean(), color=TEXT, linewidth=0.8, label=f"mean {scores.mean():.3f}")
    ax.axhline(chance, color=MUTED, linewidth=0.8, linestyle="--", label=f"chance {chance:.2f}")
    ax.set_xticks([])
    ax.set_xlim(-1, len(scores))
    label(ax, f"{len(scores)} subjects, sorted", "Test balanced accuracy")
    title(ax, "Per-subject test scores")
    legend(ax, loc="upper left", ncols=min(len(folds) + 2, 7))

    side = axes[0, 1]
    side.grid(axis="x", visible=False)
    for fold in sorted(fold_scores):
        side.bar(fold, fold_scores[fold], width=0.7, color=SERIES[fold % len(SERIES)])
        side.annotate(f"{fold_scores[fold]:.3f}", (fold, fold_scores[fold]), xytext=(0, 3),
                      textcoords="offset points", ha="center", fontsize=7, color=TEXT)
    side.axhline(chance, color=MUTED, linewidth=0.8, linestyle="--")
    side.set_xticks(sorted(fold_scores))
    label(side, "Fold")
    title(side, "Per fold")
    side.set_ylim(min(0.0, float(scores.min()) - 0.05) if len(scores) else 0.0, 1.0)
    return fig


def spatial_weights(model) -> np.ndarray:
    return model.spatial.conv.weight.detach().cpu().numpy()[:, :, 0].astype(np.float64)


def spatial_patterns(weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Activation patterns A = C W^T (W C W^T)^-1 of spatial filters W, one row per filter.

    Unlike the filter weights, patterns show where each component's signal appears on the scalp
    (Haufe et al., 2014).
    """
    centred = X - X.mean(axis=-1, keepdims=True)
    cov = np.einsum("nct,ndt->cd", centred, centred, dtype=np.float64) / max(centred.shape[0] * centred.shape[2], 1)
    source_cov = weights @ cov @ weights.T
    patterns = cov @ weights.T @ np.linalg.pinv(source_cov)
    return patterns.T


def topomap_grid(maps: np.ndarray, ch_names: Sequence[str], sfreq: float, heading: str, *, ncols: int = 8,
                 names: Sequence[str] | None = None):
    import mne

    info = eeg_info(ch_names, sfreq)
    n = len(maps)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = figure(nrows, ncols, width=1.5 * ncols + 0.4, height=1.6 * nrows + 0.6)
    for k, ax in enumerate(axes.flat):
        ax.grid(False)
        if k >= n:
            ax.axis("off")
            continue
        values = maps[k] / (np.abs(maps[k]).max() or 1.0)
        mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-1.0, 1.0), contours=0,
                             sensors=False)
        ax.set_title(names[k] if names else str(k + 1), color=TEXT, fontsize=8)
    fig.suptitle(heading, color=TEXT, fontsize=10, x=0.01, ha="left")
    return fig


def _dilated(kernel: np.ndarray, dilation: int) -> np.ndarray:
    taps = np.zeros((len(kernel) - 1) * dilation + 1)
    taps[::dilation] = kernel
    return taps


def kernel_responses(model, sfreq: float, n_fft: int = 1024):
    """Median and interquartile range of the kernels' magnitude responses, per dilation, at unit energy."""
    bank = model.bank
    groups = list(zip(bank.dilations, bank.fixed, bank.learnable if bank.learnable is not None else [None] * len(
        bank.dilations), strict=True))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
    fig, axes = figure(1, len(groups), width=3.0 * len(groups) + 0.6, height=3.2, sharey=True)
    for ax, (dilation, fixed, learnable) in zip(axes[0], groups, strict=True):
        for convs, colour, name in ((fixed, NEUTRAL, "frozen"), (learnable, SERIES[0], "learnable")):
            if convs is None:
                continue
            kernels = convs.weight.detach().cpu().numpy()[:, 0, :].astype(np.float64)
            gains = []
            for kernel in kernels:
                taps = _dilated(kernel, dilation)
                gains.append(np.abs(np.fft.rfft(taps / (np.linalg.norm(taps) or 1.0), n_fft)))
            gains = np.array(gains)
            low, mid, high = np.percentile(gains, [25, 50, 75], axis=0)
            ax.fill_between(freqs, low, high, color=colour, alpha=0.25, linewidth=0)
            ax.plot(freqs, mid, color=colour, linewidth=1.5, label=f"{name} ({len(kernels)})")
        for low_f, high_f in BANDS.values():
            ax.axvspan(low_f, high_f, color=GRID, alpha=0.4, linewidth=0)
        title(ax, f"dilation {dilation}")
        label(ax, "Frequency (Hz)")
        ax.set_xlim(0, sfreq / 2)
    label(axes[0, 0], y="Gain (unit-energy kernel)")
    legend(axes[0, -1], loc="upper right")
    fig.suptitle("Bank kernel frequency responses (lines: median, bands: 25th to 75th percentile; grey: mu and beta)",
                 color=TEXT, fontsize=10, x=0.01, ha="left")
    return fig


def gate_weights(gates: Mapping[int, np.ndarray], statistics: Sequence[str]):
    values = np.array([gates[f] for f in sorted(gates)])
    x = np.arange(len(statistics))
    fig, axes = figure(width=1.3 * len(statistics) + 3.0, height=3.4)
    ax = axes[0, 0]
    ax.grid(axis="x", visible=False)
    ax.bar(x, values.mean(axis=0), width=0.6, color=GRID, label="mean over folds")
    for i, fold in enumerate(sorted(gates)):
        offset = (i - (len(gates) - 1) / 2) * 0.08
        ax.plot(x + offset, values[i], "o", color=SERIES[fold % len(SERIES)], markersize=6, markeredgecolor="white",
                markeredgewidth=1.0, label=f"fold {fold}")
    ax.axhline(1.0 / len(statistics), color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xticks(x, list(statistics))
    label(ax, "Pooling statistic", "Gate weight")
    title(ax, f"Stream gates (dashed: uniform 1/{len(statistics)})")
    legend(ax, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return fig


def gates_of(model) -> np.ndarray:
    return torch.softmax(model.stream_gate.detach().cpu().float(), dim=0).numpy()
