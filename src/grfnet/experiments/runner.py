"""Cross-subject training and evaluation of GRF-Net models."""

from __future__ import annotations

import csv
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler

from .._progress import bar, use_bars, writer
from ..data.alignment import euclidean_alignment
from ..data.eegmmidb import EpochsData, load_epochs
from ..models import GRFNet
from .config import DataConfig, ExperimentConfig, ModelConfig, load_config, to_toml
from .metrics import classification_metrics, per_subject_scores
from .splits import check_folds, load_splits, save_splits, subject_kfold


class RunStopped(Exception):
    """A run ended between folds by a stop request. Finished folds are kept for resuming."""


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_dataset(config: DataConfig, subjects=None, *, progress: bool | None = False) -> EpochsData:
    if config.dataset != "eegmmidb":
        raise ValueError(f"unknown dataset {config.dataset!r}; available: ['eegmmidb']")
    data = load_epochs(config.root, config.task, config.subjects if subjects is None else subjects,
                       tmin=config.tmin, tmax=config.tmax, sfreq=config.sfreq, l_freq=config.l_freq,
                       h_freq=config.h_freq, include_rest=config.include_rest, channels=config.channels,
                       cache_dir=config.cache_dir, progress=progress)
    if config.normalize == "euclidean":
        # Per subject and label-free. Test subjects are aligned with their own unlabelled trials.
        data.X = euclidean_alignment(data.X, data.subjects)
    return data


def build_model(config: ModelConfig, data: EpochsData) -> GRFNet:
    models = {"GRFNet": GRFNet}
    if config.name not in models:
        raise ValueError(f"unknown model {config.name!r}; available: {sorted(models)}")
    kwargs = dict(config.kwargs)
    if config.use_channel_names:
        kwargs.setdefault("channel_names", list(data.ch_names))
    # Explicit dilations replace sfreq-based scaling.
    if "dilations" not in kwargs:
        kwargs.setdefault("sfreq", data.sfreq)
    return models[config.name](n_channels=len(data.ch_names), n_times=data.X.shape[-1], n_classes=data.n_classes,
                               **kwargs)


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, *, scale: float = 1.0, batch_size: int = 256,
                  device: torch.device | str = "cpu") -> np.ndarray:
    model.eval().to(device)
    out = []
    for i in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[i:i + batch_size] / scale).float().to(device)
        out.append(torch.softmax(model(batch).float(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def run_experiment(config: ExperimentConfig, out_dir: str | Path, *, folds: list[int] | None = None,
                   overwrite: bool = False, log=print, progress: bool | None = None,
                   stop: Callable[[], bool] | None = None) -> dict:
    """Train and test the selected folds, writing everything to ``out_dir``. Returns the pooled results.

    Running the same config into the same folder resumes it: finished folds are read back, not retrained.
    ``stop`` is checked between folds. Once it returns True, RunStopped is raised.
    """
    bars = use_bars(progress)
    log = writer(log, bars)
    out = Path(out_dir)
    resuming = False
    if out.exists() and any(out.iterdir()):
        if overwrite:
            shutil.rmtree(out)
        elif (out / "config.toml").exists() and load_config(out / "config.toml") == config:
            resuming = True
        else:
            raise FileExistsError(f"{out} holds a different run; pass overwrite=True (--overwrite) to replace it")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.toml").write_text(to_toml(config), encoding="utf-8")
    device = resolve_device(config.training.device)
    (out / "environment.json").write_text(dumps_json(environment(device)), encoding="utf-8")

    data = load_dataset(config.data, progress=progress)
    subjects = sorted(int(s) for s in np.unique(data.subjects))
    if config.split.scheme == "file":
        if config.split.file is None:
            raise ValueError("split.scheme = 'file' needs split.file")
        all_folds = load_splits(config.split.file)
    else:
        all_folds = subject_kfold(subjects, config.split.n_folds, val_fraction=config.split.val_fraction,
                                  seed=config.seed)
    check_folds(all_folds)
    if resuming and (out / "splits.json").exists() and load_splits(out / "splits.json") != all_folds:
        raise ValueError(f"{out} was split differently, e.g. over other subjects; pass --overwrite to start again")
    save_splits(out / "splits.json", all_folds, scheme=config.split.scheme, seed=config.seed,
                dataset=config.data.dataset, task=config.data.task)
    chosen = folds if folds is not None else config.split.folds
    selected = [f for f in all_folds if chosen is None or f["fold"] in chosen]
    if not selected:
        raise ValueError(f"no folds match {chosen}")

    results, predictions = [], []
    for fold in selected:
        finished = _read_fold(out / f"fold_{fold['fold']}")
        if finished is not None:
            results.append(finished[0])
            predictions.append(finished[1])
    if results:
        log(f"resuming: {len(results)}/{len(selected)} folds already completed")
    for fold in selected:
        if any(r["fold"] == fold["fold"] for r in results):
            continue
        if stop is not None and stop():
            raise RunStopped(f"stopped: {len(results)}/{len(selected)} folds completed; "
                             "run the same command again to resume")
        done = len(results)
        log(f"fold {fold['fold']}: {len(fold['train'])} train, {len(fold['val'])} val, "
            f"{len(fold['test'])} test subjects")
        fold_result, fold_predictions = train_fold(config, data, fold, out / f"fold_{fold['fold']}", device, log,
                                                   bars=bars, label=f"fold {fold['fold']} ({done}/{len(selected)} done)",
                                                   stop=stop)
        results.append(fold_result)
        predictions.append(fold_predictions)
        log(f"fold {fold['fold']}: best epoch {fold_result['best_epoch']} of {fold_result['epochs_run']}, "
            f"test balanced accuracy {fold_result['test']['balanced_accuracy']:.4f} "
            f"({done + 1}/{len(selected)} folds completed)")
    order = {f["fold"]: i for i, f in enumerate(selected)}
    paired = sorted(zip(results, predictions, strict=True), key=lambda rp: order[rp[0]["fold"]])
    results, predictions = [r for r, _ in paired], [p for _, p in paired]

    summary = summarise(config, data, results, predictions)
    (out / "results.json").write_text(dumps_json(summary), encoding="utf-8")
    _write_per_subject(out / "per_subject.csv", summary["per_subject"])
    log(f"test balanced accuracy over {len(results)} fold(s): "
        f"{summary['test']['balanced_accuracy']:.4f} (pooled)")
    return summary


def train_fold(config: ExperimentConfig, data: EpochsData, fold: dict, out: Path, device: torch.device,
               log=print, *, bars: bool = False, label: str | None = None,
               stop: Callable[[], bool] | None = None) -> tuple[dict, dict]:
    out.mkdir(parents=True, exist_ok=True)
    train_cfg = config.training
    seed_everything(config.seed + fold["fold"], train_cfg.deterministic)
    parts = {name: data.select(np.isin(data.subjects, fold[name])) for name in ("train", "val", "test")}
    if len(parts["train"]) == 0 or len(parts["test"]) == 0:
        raise ValueError(f"fold {fold['fold']} has no training or test trials")
    scale = float(parts["train"].X.std()) if config.data.normalize == "global" else 1.0
    scale = scale if scale > 0 else 1.0

    model = build_model(config.model, data).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=train_cfg.epochs)
                 if train_cfg.scheduler == "cosine" else None)
    amp = train_cfg.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp)
    X_train = torch.from_numpy(parts["train"].X / scale).float()
    y_train = torch.from_numpy(parts["train"].y)
    generator = torch.Generator().manual_seed(config.seed + fold["fold"])
    n_classes = data.n_classes

    history: list[dict] = []
    best_key: tuple | None = None
    best_epoch, best_state, stale = 0, _state_copy(model), 0
    desc = label or f"fold {fold['fold']}"
    epoch_bar = bar(total=train_cfg.epochs, enabled=bars, desc=desc, unit="epoch", leave=False)
    stopping = False
    for epoch in range(1, train_cfg.epochs + 1):
        if not stopping and stop is not None and stop():
            stopping = True
            epoch_bar.set_description(f"{desc}, stopping after this fold")
        start = time.perf_counter()
        model.train()
        total, seen, correct = 0.0, 0, 0
        for batch in torch.randperm(len(X_train), generator=generator).split(train_cfg.batch_size):
            xb, yb = X_train[batch].to(device), y_train[batch].to(device)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                logits = model(xb)
                loss = nn.functional.cross_entropy(logits.float(), yb)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            total += loss.item() * len(batch)
            seen += len(batch)
            correct += int((logits.argmax(1) == yb).sum())
        lr = optimiser.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        row = {"epoch": epoch, "lr": lr, "train_loss": total / seen, "train_accuracy": correct / seen}
        if len(parts["val"]):
            probs = predict_proba(model, parts["val"].X, scale=scale, device=device)
            row["val_loss"] = _nll(probs, parts["val"].y)
            row["val_balanced_accuracy"] = classification_metrics(
                parts["val"].y, probs.argmax(1), n_classes)["balanced_accuracy"]
        row["seconds"] = time.perf_counter() - start
        history.append(row)
        if not bars:
            log(_format_row(fold["fold"], row))

        # Highest validation balanced accuracy, ties broken by loss; the last epoch if there is no validation set.
        key = (row["val_balanced_accuracy"], -row["val_loss"]) if "val_loss" in row else (epoch,)
        if best_key is None or key > best_key:
            best_key, best_epoch, best_state, stale = key, epoch, _state_copy(model), 0
        else:
            stale += 1
        epoch_bar.set_postfix(_postfix(row, best_epoch), refresh=False)
        epoch_bar.update()
        if train_cfg.patience is not None and "val_loss" in row and stale >= train_cfg.patience:
            log(f"fold {fold['fold']}: stopping early at epoch {epoch}")
            break
    epoch_bar.close()

    _write_csv(out / "history.csv", history)
    model.load_state_dict(best_state)
    metadata = {
        "experiment": config.run_name,
        "fold": fold["fold"],
        "epoch": best_epoch,
        "scale": scale,
        "class_names": list(data.class_names),
        "ch_names": list(data.ch_names),
        "data": asdict(config.data),
        "subjects": {name: list(fold[name]) for name in ("train", "val", "test")},
        "seed": config.seed,
    }
    model.cpu().save(out / "best.pt", metadata=metadata)

    test = parts["test"]
    probs = predict_proba(model, test.X, scale=scale, device=device)
    pred = probs.argmax(1)
    result = {
        "fold": fold["fold"],
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "n_parameters": int(sum(p.numel() for p in params)),
        "scale": scale,
        "val": {k: history[best_epoch - 1][k] for k in ("val_loss", "val_balanced_accuracy")
                if k in history[best_epoch - 1]},
        "test": classification_metrics(test.y, pred, n_classes),
    }
    write_predictions(out / "predictions.csv", test, probs)
    # Written last: its presence marks the fold as finished for resuming.
    (out / "metrics.json").write_text(dumps_json(result), encoding="utf-8")
    return result, {"subjects": test.subjects, "y": test.y, "pred": pred}


def _read_fold(directory: Path) -> tuple[dict, dict] | None:
    if not (directory / "metrics.json").exists() or not (directory / "predictions.csv").exists():
        return None
    result = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    with open(directory / "predictions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    columns = {key: np.array([int(row[col]) for row in rows], dtype=np.int64)
               for key, col in (("subjects", "subject"), ("y", "y_true"), ("pred", "y_pred"))}
    return result, columns


def summarise(config: ExperimentConfig, data: EpochsData, results: list[dict], predictions: list[dict]) -> dict:
    subjects = np.concatenate([p["subjects"] for p in predictions])
    y = np.concatenate([p["y"] for p in predictions])
    pred = np.concatenate([p["pred"] for p in predictions])
    fold_scores = np.array([r["test"]["balanced_accuracy"] for r in results])
    per_subject = per_subject_scores(subjects, y, pred, data.n_classes)
    subject_scores = np.array([s["balanced_accuracy"] for s in per_subject.values()])
    return {
        "experiment": config.run_name,
        "dataset": config.data.dataset,
        "task": config.data.task,
        "class_names": list(data.class_names),
        "folds": results,
        "test": classification_metrics(y, pred, data.n_classes),
        "fold_balanced_accuracy": {"mean": float(fold_scores.mean()),
                                   "std": float(fold_scores.std(ddof=1)) if len(fold_scores) > 1 else 0.0},
        "subject_balanced_accuracy": {"mean": float(subject_scores.mean()),
                                      "std": float(subject_scores.std(ddof=1)) if len(subject_scores) > 1 else 0.0,
                                      "median": float(np.median(subject_scores))},
        "per_subject": {str(k): v for k, v in per_subject.items()},
    }


def evaluate_checkpoint(path: str | Path, *, subjects=None, root: str | None = None,
                        out_dir: str | Path | None = None, device: str = "auto") -> dict:
    """Re-test a checkpoint from ``run_experiment``, by default on its own test subjects."""
    from ..checkpoint import load_checkpoint

    meta = load_checkpoint(path)["metadata"]
    if "data" not in meta:
        raise ValueError(f"{path} has no experiment metadata; it was not written by grfnet.train")
    data_cfg = DataConfig(**meta["data"])
    if root is not None:
        data_cfg.root = root
    data = load_dataset(data_cfg, meta["subjects"]["test"] if subjects is None else subjects)
    if data.ch_names != meta["ch_names"] or list(data.class_names) != meta["class_names"]:
        raise ValueError("the loaded data does not match the checkpoint's channels or classes")
    model = GRFNet.load(path)
    probs = predict_proba(model, data.X, scale=meta["scale"], device=resolve_device(device))
    pred = probs.argmax(1)
    result = {
        "checkpoint": str(path),
        "experiment": meta.get("experiment"),
        "fold": meta.get("fold"),
        "subjects": sorted(int(s) for s in np.unique(data.subjects)),
        "test": classification_metrics(data.y, pred, data.n_classes),
        "per_subject": {str(k): v for k, v in per_subject_scores(data.subjects, data.y, pred,
                                                                 data.n_classes).items()},
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(dumps_json(result), encoding="utf-8")
        write_predictions(out / "predictions.csv", data, probs)
    return result


def compare_runs(run_a: str | Path, run_b: str | Path, metric: str = "balanced_accuracy") -> dict:
    """Wilcoxon test on the per-subject scores of two runs, over the subjects they share."""
    from .metrics import paired_wilcoxon

    scores = []
    for run in (run_a, run_b):
        summary = json.loads((Path(run) / "results.json").read_text(encoding="utf-8"))
        scores.append({s: v[metric] for s, v in summary["per_subject"].items()})
    shared = sorted(set(scores[0]) & set(scores[1]), key=int)
    if len(shared) < 2:
        raise ValueError(f"the runs share {len(shared)} subject(s); a paired test needs at least 2")
    test = paired_wilcoxon([scores[0][s] for s in shared], [scores[1][s] for s in shared])
    return {"run_a": str(run_a), "run_b": str(run_b), "metric": metric, **test}


def write_predictions(path: Path, data: EpochsData, probs: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "run", "y_true", "y_pred", *[f"p_{name}" for name in data.class_names]])
        for subject, run, y, p in zip(data.subjects, data.runs, data.y, probs, strict=True):
            writer.writerow([int(subject), int(run), int(y), int(p.argmax()), *[f"{v:.6f}" for v in p]])


def environment(device: torch.device) -> dict:
    from .. import __version__

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                                cwd=Path(__file__).resolve().parent).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "grfnet": __version__,
        "git_commit": commit,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def dumps_json(value) -> str:
    # NaN is not valid JSON.
    return json.dumps(_finite(value), indent=2)


def _finite(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


def _state_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None))))


def _postfix(row: dict, best_epoch: int) -> dict[str, str]:
    postfix = {"loss": f"{row['train_loss']:.3f}", "acc": f"{row['train_accuracy']:.3f}"}
    if "val_loss" in row:
        postfix["val_bacc"] = f"{row['val_balanced_accuracy']:.3f}"
    postfix["best"] = str(best_epoch)
    return postfix


def _format_row(fold: int, row: dict) -> str:
    text = f"fold {fold} epoch {row['epoch']:3d}  loss {row['train_loss']:.4f}  acc {row['train_accuracy']:.3f}"
    if "val_loss" in row:
        text += f"  val loss {row['val_loss']:.4f}  val bacc {row['val_balanced_accuracy']:.3f}"
    return text + f"  ({row['seconds']:.1f} s)"


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_per_subject(path: Path, per_subject: dict) -> None:
    rows = [{"subject": int(s), **v} for s, v in per_subject.items()]
    _write_csv(path, rows)
