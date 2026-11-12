"""Experiment configuration read from TOML files."""

from __future__ import annotations

import re
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# Lowercase words joined by underscores, e.g. grfnet_eegmmidb_imagery_lr_geometry.
_NAME = re.compile(r"[a-z0-9]+(_[a-z0-9]+)*")


def run_name(model: str, dataset: str, task: str, *, include_rest: bool = False, variant: str | None = None) -> str:
    """Standard run name: {model}_{dataset}_{task}, then _rest and _{variant} when they apply."""
    parts = [model, dataset, task] + (["rest"] if include_rest else []) + ([variant] if variant else [])
    return "_".join(p.lower() for p in parts)


@dataclass
class DataConfig:
    dataset: str = "eegmmidb"
    root: str | None = None
    task: str = "imagery_lr"
    subjects: str | list[int] | None = None
    tmin: float = 0.0
    tmax: float = 4.0
    sfreq: float = 160.0
    l_freq: float | None = None
    h_freq: float | None = None
    include_rest: bool = False
    channels: list[str] | None = None
    cache_dir: str | None = None
    # "global" divides by the training-set standard deviation; "euclidean" whitens each subject by its own mean
    # covariance; "none" leaves microvolts.
    normalize: str = "global"


@dataclass
class SplitConfig:
    # "kfold" or "file" (folds read from a splits.json)
    scheme: str = "kfold"
    n_folds: int = 5
    val_fraction: float = 0.1
    folds: list[int] | None = None
    file: str | None = None


@dataclass
class ModelConfig:
    name: str = "GRFNet"
    use_channel_names: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    patience: int | None = None
    device: str = "auto"
    amp: bool = False
    deterministic: bool = False


@dataclass
class ExperimentConfig:
    # Left out, the run is named by run_name(); variant tags a run that differs from its config.
    name: str | None = None
    variant: str | None = None
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def run_name(self) -> str:
        if self.name:
            return self.name
        return run_name(self.model.name, self.data.dataset, self.data.task, include_rest=self.data.include_rest,
                        variant=self.variant)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> ExperimentConfig:
        values = dict(values)
        sections = {"data": DataConfig, "split": SplitConfig, "model": ModelConfig, "training": TrainingConfig}
        built = {name: _build(section, values.pop(name, {}), name) for name, section in sections.items()}
        return _build(cls, {**values, **built}, "top level")

    def validate(self) -> ExperimentConfig:
        for key, value in (("name", self.name), ("variant", self.variant)):
            if value is not None and not _NAME.fullmatch(value):
                raise ValueError(f"{key} must be lowercase letters and digits joined by underscores; got {value!r}")
        if self.data.normalize not in ("global", "euclidean", "none"):
            raise ValueError(f"data.normalize must be 'global', 'euclidean' or 'none'; got {self.data.normalize!r}")
        if self.split.scheme not in ("kfold", "file"):
            raise ValueError(f"split.scheme must be 'kfold' or 'file'; got {self.split.scheme!r}")
        if self.split.scheme == "file" and not self.split.file:
            raise ValueError("split.scheme = 'file' needs split.file")
        if self.split.scheme == "kfold" and self.split.n_folds < 2:
            raise ValueError(f"split.n_folds must be at least 2; got {self.split.n_folds}")
        if not 0.0 <= self.split.val_fraction < 1.0:
            raise ValueError(f"split.val_fraction must be in [0, 1); got {self.split.val_fraction}")
        if self.training.scheduler not in ("cosine", "none"):
            raise ValueError(f"training.scheduler must be 'cosine' or 'none'; got {self.training.scheduler!r}")
        if self.training.epochs < 1 or self.training.batch_size < 1:
            raise ValueError("training.epochs and training.batch_size must be positive")
        return self


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Read a TOML config and apply ``section.key=value`` overrides."""
    with open(path, "rb") as f:
        values = tomllib.load(f)
    for item in overrides or []:
        apply_override(values, item)
    return ExperimentConfig.from_dict(values).validate()


def apply_override(values: dict, item: str) -> None:
    # Values parse as TOML where possible: "3" becomes an int, "1-10" stays a string.
    key, sep, raw = item.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"override must look like section.key=value; got {item!r}")
    try:
        value = tomllib.loads(f"v = {raw.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw.strip()
    *parents, leaf = key.strip().split(".")
    target = values
    for part in parents:
        target = target.setdefault(part, {})
    target[leaf] = value


def to_toml(config: ExperimentConfig) -> str:
    # TOML has no null. None fields are omitted and reload as their default, which is None for every optional field.
    values = config.to_dict()
    lines = [f"{key} = {_toml_value(value)}" for key, value in values.items()
             if not isinstance(value, dict) and value is not None]
    for section, body in values.items():
        if not isinstance(body, dict):
            continue
        lines += ["", f"[{section}]"]
        nested = []
        for key, value in body.items():
            if isinstance(value, dict):
                nested.append((key, value))
            elif value is not None:
                lines.append(f"{key} = {_toml_value(value)}")
        for key, table in nested:
            lines += ["", f"[{section}.{key}]"]
            lines += [f"{k} = {_toml_value(v)}" for k, v in table.items() if v is not None]
    return "\n".join(lines) + "\n"


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} to TOML")


def _build(cls, values: dict, where: str):
    if not isinstance(values, dict):
        raise ValueError(f"[{where}] must be a table")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown keys in {where}: {unknown}; expected some of {sorted(known)}")
    return cls(**values)
