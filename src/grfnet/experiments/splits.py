"""Subject-disjoint train, validation and test splits."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

SPLITS_FORMAT = "grfnet-splits-1"


def subject_kfold(subjects: Sequence[int], n_folds: int, *, val_fraction: float = 0.1, seed: int = 0) -> list[dict]:
    """Folds where each subject is in exactly one test set. Validation subjects come from the rest.

    Returns [{"fold", "train", "val", "test"}] with sorted subject lists.
    """
    subjects = sorted({int(s) for s in subjects})
    if n_folds < 2 or n_folds > len(subjects):
        raise ValueError(f"n_folds must be in 2..{len(subjects)} for {len(subjects)} subjects; got {n_folds}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(subjects).tolist()
    groups = [order[k::n_folds] for k in range(n_folds)]
    folds = []
    for k, test in enumerate(groups):
        rest = [s for s in order if s not in test]
        n_val = int(round(val_fraction * len(rest)))
        if val_fraction > 0:
            n_val = min(max(n_val, 1), len(rest) - 1)
        val = rng.permutation(rest)[:n_val].tolist()
        train = [s for s in rest if s not in val]
        folds.append({"fold": k, "train": sorted(train), "val": sorted(val), "test": sorted(test)})
    return folds


def check_folds(folds: Sequence[dict]) -> None:
    """Raise if any fold shares a subject between its train, val and test lists."""
    for fold in folds:
        sets = {name: set(fold.get(name, [])) for name in ("train", "val", "test")}
        if not sets["train"] or not sets["test"]:
            raise ValueError(f"fold {fold.get('fold')} needs train and test subjects")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = sets[a] & sets[b]
            if shared:
                raise ValueError(f"fold {fold.get('fold')} shares subjects between {a} and {b}: {sorted(shared)}")


def save_splits(path: str | Path, folds: Sequence[dict], **info) -> None:
    payload = {"format": SPLITS_FORMAT, **info, "folds": list(folds)}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_splits(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != SPLITS_FORMAT:
        raise ValueError(f"{path} is not a GRFNet splits file")
    folds = payload["folds"]
    check_folds(folds)
    return folds
