"""Report for a run folder written by ``python -m grfnet.train``."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .._plotting import save
from .._progress import bar, use_bars, writer
from ..checkpoint import load_checkpoint
from ..experiments.config import ExperimentConfig, load_config
from ..experiments.runner import dumps_json, load_dataset, predict_proba, resolve_device
from ..models import GRFNet
from . import analysis, benchmark, figures


@dataclass
class RunFiles:

    path: Path
    config: ExperimentConfig
    results: dict
    splits: list[dict]
    histories: dict[int, list[dict]]
    checkpoints: dict[int, Path]


def read_run(run_dir: str | Path) -> RunFiles:
    path = Path(run_dir)
    for name in ("config.toml", "results.json"):
        if not (path / name).exists():
            raise FileNotFoundError(f"{path / name} not found; is {path} a folder written by python -m grfnet.train?")
    results = json.loads((path / "results.json").read_text(encoding="utf-8"))
    splits = json.loads((path / "splits.json").read_text(encoding="utf-8"))["folds"] if (
        path / "splits.json").exists() else []
    folds = [int(f["fold"]) for f in results["folds"]]
    histories = {k: read_history(path / f"fold_{k}" / "history.csv") for k in folds
                 if (path / f"fold_{k}" / "history.csv").exists()}
    checkpoints = {k: path / f"fold_{k}" / "best.pt" for k in folds if (path / f"fold_{k}" / "best.pt").exists()}
    return RunFiles(path, load_config(path / "config.toml"), results, splits, histories, checkpoints)


def read_history(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{k: (int(v) if k == "epoch" else float(v)) if v not in ("", None) else None for k, v in row.items()}
            for row in rows]


def write_run_report(run_dir: str | Path, out_dir: str | Path | None = None, *, fold: int | None = None,
                     root: str | None = None, data: bool = True, bench: bool = True, max_trials: int = 3000,
                     dpi: int = 150, device: str = "auto", progress: bool | None = None, log=print) -> dict:
    """Write figures, ``report.md``, ``report.json`` and ``efficiency.json`` for a run.

    Model figures use the checkpoint of ``fold``, or the lowest fold. ``data=False`` skips the analyses
    that read the dataset; ``bench=False`` skips the timing benchmark.
    """
    run = read_run(run_dir)
    out = Path(out_dir) if out_dir is not None else run.path / "report"
    out.mkdir(parents=True, exist_ok=True)
    bars = use_bars(progress)
    log = writer(log, bars)
    results = run.results
    class_names = results["class_names"]
    folds = sorted(int(f["fold"]) for f in results["folds"])
    chosen = fold if fold is not None else folds[0]
    if chosen not in run.checkpoints:
        raise ValueError(f"fold {chosen} has no checkpoint in {run.path}; available: {sorted(run.checkpoints)}")
    figure_notes: dict[str, str] = {}
    skipped: dict[str, str] = {}
    steps = bar(total=8, enabled=bars, desc="report", unit="step", leave=False)

    def emit(name: str, fig, caption: str) -> None:
        save(fig, out / name, dpi)
        figure_notes[name] = caption
        steps.update()

    best = {int(f["fold"]): int(f["best_epoch"]) for f in results["folds"]}
    emit("learning_curves.png", figures.learning_curves(run.histories, best),
         "Training and validation curves per fold. Dots mark the epoch kept for testing.")
    emit("confusion_matrix.png", figures.confusion_matrix(results["test"]["confusion_matrix"], class_names),
         "Pooled test confusion matrix over all folds, with counts and the share of each true class.")
    subject_fold = {int(s): int(f["fold"]) for f in run.splits for s in f["test"] if int(f["fold"]) in folds}
    fold_scores = {int(f["fold"]): float(f["test"]["balanced_accuracy"]) for f in results["folds"]}
    emit("subject_scores.png", figures.subject_scores(results["per_subject"], subject_fold, fold_scores,
                                                      len(class_names)),
         "Test balanced accuracy for every subject, coloured by the fold that tested it, and for each fold.")

    models, metas = {}, {}
    for k, path in sorted(run.checkpoints.items()):
        models[k] = GRFNet.load(path)
        metas[k] = load_checkpoint(path)["metadata"]
    model, meta = models[chosen], metas[chosen]
    sfreq = float(meta["data"]["sfreq"])
    emit("stream_gates.png", figures.gate_weights({k: figures.gates_of(m) for k, m in models.items()},
                                                  list(model.statistics)),
         "Softmax weight of each pooling statistic in every fold's model.")
    emit("spatial_filters.png", figures.topomap_grid(figures.spatial_weights(model), meta["ch_names"], sfreq,
                                                     f"Spatial filter weights, fold {chosen} (each map scaled to its own maximum)"),
         f"Spatial filter weights of fold {chosen}'s model. Weights show how channels are combined, not where the signal comes from.")
    emit("kernel_responses.png", figures.kernel_responses(model, sfreq),
         f"Frequency responses of fold {chosen}'s frozen and learnable bank kernels at each dilation.")

    analyses: dict = {}
    if data:
        try:
            data_cfg = replace(run.config.data, root=root) if root is not None else run.config.data
            log(f"loading the {len(meta['subjects']['test'])} test subjects of fold {chosen}")
            test = load_dataset(data_cfg, meta["subjects"]["test"], progress=progress)
        except (FileNotFoundError, ImportError) as error:
            skipped["data analyses"] = str(error)
            log(f"skipping data analyses: {error}")
            test = None
        if test is not None:
            analyses = _data_analyses(model, meta, test, out, emit, chosen, max_trials, dpi, device,
                                      run.config.seed)
    else:
        skipped["data analyses"] = "disabled (--no-data)"
    steps.close()

    efficiency = None
    if bench:
        log("benchmarking")
        efficiency = benchmark.benchmark_model(model, len(meta["ch_names"]), int(model.n_times or round(
            (meta["data"]["tmax"] - meta["data"]["tmin"]) * sfreq)), checkpoint=run.checkpoints[chosen],
            histories=run.histories)
        (out / "efficiency.json").write_text(dumps_json(efficiency), encoding="utf-8")
    else:
        skipped["benchmark"] = "disabled (--no-benchmark)"

    summary = _summary(run, chosen, models[chosen], analyses, efficiency, figure_notes, skipped)
    (out / "report.json").write_text(dumps_json(summary), encoding="utf-8")
    (out / "report.md").write_text(_markdown(summary, figure_notes), encoding="utf-8")
    log(f"report written to {out}")
    return summary


def _data_analyses(model, meta, test, out, emit, fold, max_trials, dpi, device, seed) -> dict:
    scale = float(meta["scale"])
    dev = resolve_device(device)
    X = test.X
    patterns = figures.spatial_patterns(figures.spatial_weights(model), X / scale)
    emit("spatial_patterns.png", figures.topomap_grid(patterns, test.ch_names, test.sfreq,
                                                      f"Spatial activation patterns, fold {fold} (each map scaled to its own maximum)"),
         f"Activation patterns of fold {fold}'s spatial filters, estimated from its test data (Haufe et al. 2014). "
         "Patterns show which channels each component responds to.")

    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(len(X), size=min(max_trials, len(X)), replace=False))
    probs = predict_proba(model, X[index], scale=scale, device=dev)
    pred = probs.argmax(1)
    points, method = analysis.project_2d(analysis.embeddings(model, X[index], scale=scale, device=dev), seed=seed)
    emit("embeddings.png", analysis.embedding_figure(points, method, test.y[index], pred, test.class_names),
         f"{method} projection of fold {fold}'s embedding for {len(index)} test trials.")

    values = analysis.saliency(model, X[index], test.y[index], scale=scale, device=dev)
    emit("saliency.png", analysis.saliency_figure(values, test.y[index], test.class_names, test.ch_names,
                                                  test.sfreq, test.tmin),
         f"Mean absolute gradient x input of the true-class score, per channel and over time, for fold {fold}.")
    channel_saliency = values.mean(axis=(0, 2))
    top = np.argsort(channel_saliency)[::-1][:5]
    model.cpu()
    return {"fold": fold, "n_trials": int(len(index)), "projection": method,
            "top_saliency_channels": [test.ch_names[i] for i in top]}


def _summary(run: RunFiles, fold: int, model, analyses: dict, efficiency: dict | None, figure_notes: dict,
             skipped: dict) -> dict:
    results = run.results
    folds = []
    split_of = {int(f["fold"]): f for f in run.splits}
    for f in results["folds"]:
        split = split_of.get(int(f["fold"]), {})
        folds.append({
            "fold": int(f["fold"]),
            "subjects": {name: len(split.get(name, [])) for name in ("train", "val", "test")},
            "best_epoch": f["best_epoch"],
            "epochs_run": f["epochs_run"],
            "val_balanced_accuracy": f.get("val", {}).get("val_balanced_accuracy"),
            "test_balanced_accuracy": f["test"]["balanced_accuracy"],
            "test_accuracy": f["test"]["accuracy"],
            "test_trials": f["test"]["n"],
        })
    return {
        "run": str(run.path),
        "experiment": results.get("experiment"),
        "dataset": results.get("dataset"),
        "task": results.get("task"),
        "model": run.config.model.name,
        "class_names": results["class_names"],
        "chance": 1.0 / len(results["class_names"]),
        "n_subjects": len(results["per_subject"]),
        "test": {k: v for k, v in results["test"].items()},
        "fold_balanced_accuracy": results["fold_balanced_accuracy"],
        "subject_balanced_accuracy": results["subject_balanced_accuracy"],
        "folds": folds,
        "model_figures_fold": fold,
        "parameters": benchmark.count_parameters(model),
        "analyses": analyses,
        "efficiency": efficiency,
        "figures": figure_notes,
        "skipped": skipped,
    }


def _markdown(summary: dict, figure_notes: dict[str, str]) -> str:
    test = summary["test"]
    fold_b, subj_b = summary["fold_balanced_accuracy"], summary["subject_balanced_accuracy"]
    params = summary["parameters"]
    lines = [
        f"# Run report: {summary['experiment']}",
        "",
        f"{summary['model']} on {summary['dataset']} / {summary['task']}, {len(summary['folds'])} subject-disjoint "
        f"fold(s), {summary['n_subjects']} test subjects. All numbers are read from the run folder.",
        "",
        "## Test results",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Balanced accuracy (pooled) | {test['balanced_accuracy']:.4f} |",
        f"| Accuracy (pooled) | {test['accuracy']:.4f} |",
        f"| Cohen's kappa (pooled) | {test['kappa']:.4f} |",
        f"| Macro F1 (pooled) | {test['macro_f1']:.4f} |",
        f"| Balanced accuracy per fold, mean (SD) | {fold_b['mean']:.4f} ({fold_b['std']:.4f}) |",
        f"| Balanced accuracy per subject, mean (SD) | {subj_b['mean']:.4f} ({subj_b['std']:.4f}) |",
        f"| Balanced accuracy per subject, median | {subj_b['median']:.4f} |",
        f"| Chance level | {summary['chance']:.4f} |",
        f"| Test trials | {test['n']} |",
        f"| Trainable / frozen parameters | {params['trainable']:,} / {params['frozen']:,} |",
        "",
        "## Folds",
        "",
        "| Fold | Train / val / test subjects | Best epoch | Epochs run | Val balanced accuracy | Test balanced accuracy | Test accuracy |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f in summary["folds"]:
        val = f"{f['val_balanced_accuracy']:.4f}" if f["val_balanced_accuracy"] is not None else "n/a"
        s = f["subjects"]
        lines.append(f"| {f['fold']} | {s['train']} / {s['val']} / {s['test']} | {f['best_epoch']} | "
                     f"{f['epochs_run']} | {val} | {f['test_balanced_accuracy']:.4f} | {f['test_accuracy']:.4f} |")
    efficiency = summary["efficiency"]
    if efficiency:
        flops = efficiency["flops_per_trial"]
        lines += ["", "## Efficiency", "",
                  f"Measured on {efficiency['machine']['processor'] or efficiency['machine']['platform']}"
                  + (f" and {efficiency['machine']['cuda_device']}" if efficiency["machine"]["cuda_device"] else "")
                  + f" with PyTorch {efficiency['machine']['torch']}. Timings depend on the hardware and its load.",
                  "", "| Measure | Value |", "| --- | --- |",
                  f"| Parameters (trainable / frozen / total) | {params['trainable']:,} / {params['frozen']:,} / "
                  f"{params['total']:,} |",
                  f"| Operations per trial | {flops / 1e6:.2f} MFLOPs |" if flops is not None else
                  "| Operations per trial | n/a |",
                  f"| Checkpoint size | {efficiency['checkpoint_bytes'] / 1024:.1f} KiB |"]
        for t in efficiency["inference"]:
            lines.append(f"| Inference, {t['device']}, batch {t['batch_size']} | {t['median_ms']:.2f} ms "
                         f"({t['trials_per_second']:,.0f} trials/s) |")
        training = efficiency.get("training")
        if training and training["median_epoch_seconds"] is not None:
            lines.append(f"| Training, median epoch | {training['median_epoch_seconds']:.2f} s |")
            lines.append(f"| Training, all folds | {training['total_seconds'] / 60:.1f} min |")
    lines += ["", "## Figures", "",
              f"Model figures use fold {summary['model_figures_fold']}'s checkpoint.", ""]
    for name, caption in figure_notes.items():
        lines += [f"### {name.removesuffix('.png').replace('_', ' ').capitalize()}", "", caption, "",
                  f"![{caption}]({name})", ""]
    if summary["analyses"].get("top_saliency_channels"):
        lines += [f"Channels with the highest mean saliency: {', '.join(summary['analyses']['top_saliency_channels'])}.",
                  ""]
    if summary["skipped"]:
        lines += ["## Skipped", ""] + [f"- {k}: {v}" for k, v in summary["skipped"].items()] + [""]
    return "\n".join(lines)
