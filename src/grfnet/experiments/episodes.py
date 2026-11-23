"""Episodic training and sequential prediction for stateful models such as FastGRFNet.

An episode is one subject's trials in recording order. Episodes of several subjects run in
parallel, one trial per subject per step, and the state starts at zero for every subject.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn


def subject_sequences(subjects: np.ndarray, runs: np.ndarray) -> dict[int, np.ndarray]:
    """Trial indices of each subject, ordered by run and then by position in the data."""
    order = np.lexsort((np.arange(len(subjects)), runs, subjects))
    out: dict[int, list[int]] = {}
    for index in order:
        out.setdefault(int(subjects[index]), []).append(int(index))
    return {s: np.asarray(v, dtype=np.int64) for s, v in out.items()}


def trial_positions(subjects: np.ndarray, runs: np.ndarray) -> np.ndarray:
    """Position of each trial within its subject's sequence, from 0."""
    positions = np.zeros(len(subjects), dtype=np.int64)
    for indices in subject_sequences(subjects, runs).values():
        positions[indices] = np.arange(len(indices))
    return positions


def uses_labels(model: nn.Module) -> bool:
    return getattr(model, "write_mode", None) == "label_feedback"


def unroll(model: nn.Module, X: torch.Tensor, y: torch.Tensor, sequences: list[np.ndarray], *,
           device: torch.device | str, chunk_length: int | None = None,
           on_chunk: Callable[[torch.Tensor, int], None] | None = None) -> torch.Tensor:
    """Run sequences in parallel from a zero state and return the logits of every trial.

    Trials of finished sequences drop out of the batch. With ``on_chunk``, the summed loss of every
    ``chunk_length`` steps is passed to it with its trial count, and the state is detached afterwards.
    """
    feedback = uses_labels(model)
    length = max(len(s) for s in sequences)
    state = model.init_state(len(sequences), device)  # pyright: ignore[reportCallIssue]
    logits_all = torch.zeros(len(X), model.n_classes, device=device)  # pyright: ignore[reportArgumentType]
    chunk_loss, chunk_trials = torch.zeros((), device=device), 0
    for t in range(length):
        active = [i for i, s in enumerate(sequences) if t < len(s)]
        rows = torch.as_tensor([int(sequences[i][t]) for i in active])
        xb, yb = X[rows].to(device), y[rows].to(device)
        logits, new_state = model(xb, state[active], label=yb if feedback else None)
        state = state.clone()
        state[active] = new_state
        logits_all[rows.to(device)] = logits.detach().float()
        if on_chunk is not None:
            chunk_loss = chunk_loss + nn.functional.cross_entropy(logits.float(), yb, reduction="sum")
            chunk_trials += len(active)
            if (t + 1) % (chunk_length or length) == 0 or t == length - 1:
                on_chunk(chunk_loss, chunk_trials)
                chunk_loss, chunk_trials = torch.zeros((), device=device), 0
                state = state.detach()
    return logits_all


def train_epoch(model: nn.Module, X: torch.Tensor, y: torch.Tensor, sequences: dict[int, np.ndarray],
                optimiser: torch.optim.Optimizer, *, chunk_length: int, subjects_per_batch: int,
                generator: torch.Generator, device: torch.device | str) -> tuple[float, int, int]:
    """One pass over all training subjects with truncated back-propagation through the writes.

    Returns the summed loss, the number of trials and the number predicted correctly.
    """
    model.train()
    subjects = list(sequences)
    order = torch.randperm(len(subjects), generator=generator).tolist()
    total, seen, correct = 0.0, 0, 0

    def step(loss: torch.Tensor, n: int) -> None:
        nonlocal total
        optimiser.zero_grad(set_to_none=True)
        (loss / n).backward()
        optimiser.step()
        total += float(loss.detach())

    for start in range(0, len(order), subjects_per_batch):
        group = [sequences[subjects[i]] for i in order[start:start + subjects_per_batch]]
        logits = unroll(model, X, y, group, device=device, chunk_length=chunk_length, on_chunk=step)
        rows = torch.as_tensor(np.concatenate(group))
        seen += len(rows)
        correct += int((logits[rows.to(device)].argmax(1).cpu() == y[rows]).sum())
    return total, seen, correct


@torch.no_grad()
def predict_sequences(model: nn.Module, X: np.ndarray, y: np.ndarray, subjects: np.ndarray, runs: np.ndarray, *,
                      scale: float = 1.0, device: torch.device | str = "cpu") -> np.ndarray:
    """Class probabilities with each subject's trials streamed in order from a zero state.

    ``label_feedback`` models receive each trial's label after it has been predicted.
    """
    model.eval().to(device)
    sequences = list(subject_sequences(subjects, runs).values())
    if not sequences:
        return np.zeros((0, 0), np.float32)
    logits = unroll(model, torch.from_numpy(X / scale).float(), torch.from_numpy(y).long(), sequences,
                    device=device)
    return torch.softmax(logits, dim=1).cpu().numpy()
