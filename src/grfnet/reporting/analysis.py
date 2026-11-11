"""Embedding projections and input saliency."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence

import numpy as np
import torch

from .._plotting import MUTED, SERIES, TEXT, class_label, eeg_info, figure, label, legend, title


@torch.no_grad()
def embeddings(model, X: np.ndarray, *, scale: float = 1.0, batch_size: int = 256,
               device: torch.device | str = "cpu") -> np.ndarray:
    model.eval().to(device)
    out = [model.forward_features(torch.from_numpy(X[i:i + batch_size] / scale).float().to(device)).cpu().numpy()
           for i in range(0, len(X), batch_size)]
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def project_2d(features: np.ndarray, *, seed: int = 0) -> tuple[np.ndarray, str]:
    """2-D t-SNE when scikit-learn is installed, PCA otherwise."""
    standardised = (features - features.mean(0)) / (features.std(0) + 1e-8)
    if importlib.util.find_spec("sklearn") is not None and len(features) > 5:
        from sklearn.manifold import TSNE

        perplexity = float(min(30.0, (len(features) - 1) / 3))
        points = TSNE(n_components=2, init="pca", perplexity=perplexity, random_state=seed).fit_transform(standardised)
        return np.asarray(points), "t-SNE"
    _, _, vt = np.linalg.svd(standardised - standardised.mean(0), full_matrices=False)
    return standardised @ vt[:2].T, "PCA"


def embedding_figure(points: np.ndarray, method: str, y: np.ndarray, pred: np.ndarray, class_names: Sequence[str]):
    fig, axes = figure(1, 2, width=10.0, height=4.6)
    for c, name in enumerate(class_names):
        mask = y == c
        axes[0, 0].scatter(points[mask, 0], points[mask, 1], s=7, color=SERIES[c % len(SERIES)], alpha=0.6,
                           linewidths=0, label=class_label(name))
    correct = y == pred
    axes[0, 1].scatter(points[correct, 0], points[correct, 1], s=7, color=SERIES[0], alpha=0.5, linewidths=0,
                       label=f"correct ({correct.mean():.1%})")
    axes[0, 1].scatter(points[~correct, 0], points[~correct, 1], s=7, color=SERIES[1], alpha=0.7, linewidths=0,
                       label="wrong")
    for ax, name in zip(axes[0], ("True class", "Prediction"), strict=True):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        title(ax, f"{name} ({method} of the embedding, {len(points)} test trials)")
        legend(ax, loc="best", markerscale=2.5)
    return fig


def saliency(model, X: np.ndarray, y: np.ndarray, *, scale: float = 1.0, batch_size: int = 128,
             device: torch.device | str = "cpu") -> np.ndarray:
    """Absolute gradient x input of the true-class score."""
    model.eval().to(device)
    out = []
    for i in range(0, len(X), batch_size):
        x = torch.from_numpy(X[i:i + batch_size] / scale).float().to(device).requires_grad_(True)
        target = torch.from_numpy(y[i:i + batch_size]).long().to(device)
        scores = model(x).gather(1, target[:, None]).sum()
        (grad,) = torch.autograd.grad(scores, x)
        out.append((grad * x).abs().detach().cpu().numpy())
    return np.concatenate(out) if out else np.zeros_like(X)


def saliency_figure(values: np.ndarray, y: np.ndarray, class_names: Sequence[str], ch_names: Sequence[str],
                    sfreq: float, tmin: float):
    import mne

    info = eeg_info(ch_names, sfreq)
    n = len(class_names)
    fig, axes = figure(1, n + 1, width=2.8 * n + 4.6, height=3.2,
                       gridspec_kw={"width_ratios": [1] * n + [1.8]})
    per_class = [values[y == c] for c in range(n)]
    maps = [v.mean(axis=(0, 2)) if len(v) else np.zeros(len(ch_names)) for v in per_class]
    top = float(max(m.max() for m in maps)) or 1.0
    image = None
    for c, ax in enumerate(axes[0, :n]):
        ax.grid(False)
        image, _ = mne.viz.plot_topomap(maps[c], info, axes=ax, show=False, cmap="Blues", vlim=(0.0, top),
                                        contours=0, sensors=True)
        title(ax, class_label(class_names[c]))
    if image is not None:
        bar = fig.colorbar(image, ax=list(axes[0, :n]), shrink=0.75, location="left")
        bar.set_label("Mean |gradient x input|", color=MUTED, fontsize=8)
        bar.ax.tick_params(colors=MUTED, labelsize=7)

    ax = axes[0, n]
    times = tmin + np.arange(values.shape[-1]) / sfreq
    width = max(1, int(round(0.1 * sfreq)))
    kernel = np.ones(width) / width
    for c, v in enumerate(per_class):
        if len(v):
            curve = np.convolve(v.mean(axis=(0, 1)), kernel, mode="same")
            curve[:width // 2] = curve[len(curve) - width // 2:] = np.nan
            ax.plot(times, curve, color=SERIES[c % len(SERIES)], linewidth=1.5, label=class_label(class_names[c]))
    if tmin < 0:
        ax.axvline(0.0, color=MUTED, linewidth=0.8, linestyle="--")
    label(ax, "Time from cue (s)", "Mean |gradient x input|")
    title(ax, "Over time (0.1 s moving mean)")
    legend(ax, loc="upper right")
    fig.suptitle("Input saliency for the true class", color=TEXT, fontsize=10, x=0.01, ha="left")
    return fig
