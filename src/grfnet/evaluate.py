"""Evaluate a checkpoint or compare two runs: python -m grfnet.evaluate."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m grfnet.evaluate",
                                     description="Test a trained checkpoint, or compare two runs per subject.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--checkpoint", help="best.pt written by python -m grfnet.train")
    action.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                        help="two run folders; paired Wilcoxon test on per-subject scores (eval extra)")
    parser.add_argument("--subjects", default=None, help="subjects to test, e.g. 1-10 (default: the fold's test subjects)")
    parser.add_argument("--root", default=None, help="dataset folder, if it has moved since training")
    parser.add_argument("--out", default=None, help="folder for metrics.json and predictions.csv")
    parser.add_argument("--metric", default="balanced_accuracy", choices=["balanced_accuracy", "accuracy"],
                        help="per-subject score for --compare")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    from .experiments.runner import compare_runs, dumps_json, evaluate_checkpoint

    if args.compare:
        result = compare_runs(*args.compare, metric=args.metric)
    else:
        result = evaluate_checkpoint(args.checkpoint, subjects=args.subjects, root=args.root, out_dir=args.out,
                                     device=args.device)
        result = {k: v for k, v in result.items() if k != "per_subject"}
    print(dumps_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
