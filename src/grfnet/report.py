"""Write a run report: python -m grfnet.report RUN_DIR."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m grfnet.report",
                                     description="Write figures, tables and an efficiency benchmark for a run.")
    parser.add_argument("run_dir", help="folder written by python -m grfnet.train")
    parser.add_argument("--out", default=None, help="output folder (default: RUN_DIR/report)")
    parser.add_argument("--fold", type=int, default=None, help="fold whose model the model figures use (default: lowest)")
    parser.add_argument("--root", default=None, help="dataset folder, if it has moved since training")
    parser.add_argument("--no-data", action="store_true", help="skip the analyses that read the dataset")
    parser.add_argument("--no-benchmark", action="store_true", help="skip the timing benchmark")
    parser.add_argument("--max-trials", type=int, default=3000, help="test trials for embeddings and saliency")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true", help="hide progress bars")
    args = parser.parse_args(argv)

    from .reporting.report import write_run_report

    write_run_report(args.run_dir, args.out, fold=args.fold, root=args.root, data=not args.no_data,
                     bench=not args.no_benchmark, max_trials=args.max_trials, dpi=args.dpi, device=args.device,
                     progress=False if args.no_progress else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
