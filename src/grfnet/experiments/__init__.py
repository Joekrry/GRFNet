"""Experiment configuration, subject splits, metrics and the training runner."""

from .config import ExperimentConfig, load_config
from .metrics import balanced_accuracy, classification_metrics, confusion_matrix, paired_wilcoxon
from .runner import compare_runs, evaluate_checkpoint, run_experiment
from .splits import load_splits, subject_kfold

__all__ = [
    "ExperimentConfig",
    "balanced_accuracy",
    "classification_metrics",
    "compare_runs",
    "confusion_matrix",
    "evaluate_checkpoint",
    "load_config",
    "load_splits",
    "paired_wilcoxon",
    "run_experiment",
    "subject_kfold",
]
