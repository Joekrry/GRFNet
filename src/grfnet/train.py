"""Command line: python -m grfnet.train CONFIG [options]."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m grfnet.train",
                                     description="Train and test a model with subject-disjoint folds.")
    parser.add_argument("config", help="TOML experiment config, e.g. configs/eegmmidb_imagery_lr.toml")
    parser.add_argument("--out", default=None, help="run folder (default: runs/<name>)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set training.epochs=5 (repeatable)")
    parser.add_argument("--folds", default=None, help="fold numbers to run, e.g. 0 or 0,2 (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run folder")
    args = parser.parse_args(argv)

    from .experiments.config import load_config
    from .experiments.runner import run_experiment

    config = load_config(args.config, args.overrides)
    folds = [int(f) for f in args.folds.split(",")] if args.folds else None
    out = Path(args.out) if args.out else Path("runs") / config.name
    run_experiment(config, out, folds=folds, overwrite=args.overwrite)
    print(f"results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
