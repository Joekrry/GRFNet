from __future__ import annotations

import numpy as np
import pytest

from grfnet.data import euclidean_alignment
from grfnet.data.alignment import reference_matrix


def _subjects(n_subjects=3, n_trials=20, n_channels=5, n_times=200, seed=0):
    # Each subject gets its own channel mixing and gain.
    rng = np.random.default_rng(seed)
    X, groups = [], []
    for subject in range(n_subjects):
        mixing = rng.standard_normal((n_channels, n_channels)) * (subject + 1)
        X.append(np.einsum("cd,ndt->nct", mixing, rng.standard_normal((n_trials, n_channels, n_times))))
        groups += [subject + 1] * n_trials
    return np.concatenate(X).astype(np.float32), np.array(groups)


def _mean_covariance(X):
    X = X - X.mean(axis=-1, keepdims=True)
    return np.einsum("nct,ndt->cd", X, X) / (X.shape[0] * X.shape[-1])


def test_each_subject_ends_with_identity_mean_covariance():
    X, groups = _subjects()
    aligned = euclidean_alignment(X, groups)
    assert aligned.shape == X.shape and aligned.dtype == np.float32
    for subject in np.unique(groups):
        np.testing.assert_allclose(_mean_covariance(aligned[groups == subject]), np.eye(X.shape[1]), atol=1e-4)


def test_each_subject_is_aligned_independently_of_the_others():
    X, groups = _subjects()
    together = euclidean_alignment(X, groups)
    alone = euclidean_alignment(X[groups == 2], groups[groups == 2])
    np.testing.assert_allclose(together[groups == 2], alone, rtol=1e-5, atol=1e-5)


def test_reference_matrix_is_symmetric_and_survives_rank_deficiency():
    X, _ = _subjects(n_subjects=1)
    X = X - X.mean(axis=1, keepdims=True)  # common average reference: rank n_channels - 1
    R = reference_matrix(X)
    np.testing.assert_allclose(R, R.T, atol=1e-8)
    assert np.isfinite(R).all()


def test_alignment_rejects_bad_shapes():
    X, groups = _subjects()
    with pytest.raises(ValueError, match="one entry per trial"):
        euclidean_alignment(X, groups[:-1])
    with pytest.raises(ValueError, match="n_trials"):
        reference_matrix(X[0])
