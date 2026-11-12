"""Euclidean alignment of EEG trials across subjects."""

from __future__ import annotations

import numpy as np


def reference_matrix(X: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Inverse square root of the mean trial covariance of ``X`` with shape (n_trials, n_channels, n_times).

    Eigenvalues below ``eps`` times the largest are raised to that floor.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3 or len(X) == 0:
        raise ValueError(f"X must have shape (n_trials, n_channels, n_times) with n_trials > 0; got {X.shape}")
    X = X - X.mean(axis=-1, keepdims=True)
    covariance = np.einsum("nct,ndt->cd", X, X) / (X.shape[0] * X.shape[-1])
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, eps * values.max())
    return (vectors / np.sqrt(values)) @ vectors.T


def euclidean_alignment(X: np.ndarray, groups: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Whiten each group's trials by its own mean covariance [He and Wu, 2019].

    Every group, usually a subject, ends with an identity mean covariance. Labels are not used, and each group
    depends only on its own trials.

    Parameters
    ----------
    X : array of shape (n_trials, n_channels, n_times)
    groups : array of shape (n_trials,)
        Group of each trial.
    eps : float
        Relative eigenvalue floor of each covariance.

    Returns
    -------
    array of shape (n_trials, n_channels, n_times), float32
    """
    groups = np.asarray(groups)
    if groups.shape != (len(X),):
        raise ValueError(f"groups must have one entry per trial; got {groups.shape} for {len(X)} trials")
    out = np.empty(np.shape(X), dtype=np.float32)
    for group in np.unique(groups):
        mask = groups == group
        R = reference_matrix(X[mask], eps=eps)
        out[mask] = np.einsum("cd,ndt->nct", R, np.asarray(X[mask], dtype=np.float64))
    return out
