"""Classification metrics and paired tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Counts with true classes as rows and predicted classes as columns."""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return matrix


def balanced_accuracy(y_true, y_pred, n_classes: int) -> float:
    """Mean recall over the classes present in y_true."""
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    support = matrix.sum(axis=1)
    present = support > 0
    if not present.any():
        return float("nan")
    return float(np.mean(np.diag(matrix)[present] / support[present]))


def cohen_kappa(y_true, y_pred, n_classes: int) -> float:
    matrix = confusion_matrix(y_true, y_pred, n_classes).astype(np.float64)
    total = matrix.sum()
    if total == 0:
        return float("nan")
    observed = np.trace(matrix) / total
    expected = float(matrix.sum(0) @ matrix.sum(1)) / total**2
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 else float("nan")


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    matrix = confusion_matrix(y_true, y_pred, n_classes).astype(np.float64)
    tp = np.diag(matrix)
    denom = matrix.sum(0) + matrix.sum(1)
    present = matrix.sum(1) > 0
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1[present].mean()) if present.any() else float("nan")


def classification_metrics(y_true, y_pred, n_classes: int) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else float("nan"),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, n_classes),
        "kappa": cohen_kappa(y_true, y_pred, n_classes),
        "macro_f1": macro_f1(y_true, y_pred, n_classes),
        "confusion_matrix": confusion_matrix(y_true, y_pred, n_classes).tolist(),
    }


def per_subject_scores(subjects, y_true, y_pred, n_classes: int) -> dict[int, dict]:
    subjects, y_true, y_pred = np.asarray(subjects), np.asarray(y_true), np.asarray(y_pred)
    scores = {}
    for subject in np.unique(subjects):
        mask = subjects == subject
        scores[int(subject)] = {
            "n": int(mask.sum()),
            "accuracy": float(np.mean(y_true[mask] == y_pred[mask])),
            "balanced_accuracy": balanced_accuracy(y_true[mask], y_pred[mask], n_classes),
        }
    return scores


def paired_wilcoxon(a: ArrayLike, b: ArrayLike) -> dict:
    """Two-sided Wilcoxon signed-rank test on paired scores, such as per-subject accuracies. Requires the eval extra."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        raise ImportError("paired_wilcoxon requires SciPy; install it with: pip install grfnet[eval]") from None
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"a and b must be 1-D and the same length; got {x.shape} and {y.shape}")
    diff = x - y
    result = {"n": int(len(x)), "mean_a": float(x.mean()), "mean_b": float(y.mean()),
              "median_difference": float(np.median(diff)), "n_wins_a": int(np.sum(diff > 0)),
              "n_wins_b": int(np.sum(diff < 0)), "n_ties": int(np.sum(diff == 0))}
    if np.all(diff == 0):
        return {**result, "statistic": float("nan"), "p_value": 1.0}
    # SciPy's result type is untyped in its stubs; it unpacks as (statistic, p-value).
    statistic, p_value = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return {**result, "statistic": float(np.asarray(statistic)), "p_value": float(np.asarray(p_value))}
