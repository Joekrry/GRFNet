"""Dataset report: python -m grfnet.data report."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .eegmmidb import TASKS, default_root, load_epochs, parse_subjects, survey_recordings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m grfnet.data", description="EEGMMIDB dataset tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="write dataset figures and summary tables")
    report.add_argument("--root", default=None, help=f"dataset folder (default: $GRFNET_EEGMMIDB or {default_root()})")
    report.add_argument("--out", default=None, help="output folder (default: runs/dataset_eegmmidb_<task>)")
    report.add_argument("--task", default="imagery_lr", choices=sorted(TASKS))
    report.add_argument("--subjects", default=None, help="e.g. 1-10,12 (default: all)")
    report.add_argument("--include-rest", action="store_true", help="add T0 rest periods as a class")
    report.add_argument("--tmin", type=float, default=-1.0, help="epoch start from cue in s (default: -1)")
    report.add_argument("--tmax", type=float, default=4.0, help="epoch end from cue in s (default: 4)")
    report.add_argument("--l-freq", type=float, default=None, help="high-pass edge in Hz")
    report.add_argument("--h-freq", type=float, default=None, help="low-pass edge in Hz")
    report.add_argument("--cache-dir", default=None, help="folder for cached epochs (e.g. runs/cache)")
    report.add_argument("--dpi", type=int, default=150)
    report.add_argument("--no-progress", action="store_true", help="hide progress bars")
    args = parser.parse_args(argv)

    from ..experiments.config import run_name
    from .report import write_report

    out = args.out or f"runs/{run_name('dataset', 'eegmmidb', args.task, include_rest=args.include_rest)}"

    subjects = parse_subjects(args.subjects)
    progress = False if args.no_progress else None
    print("surveying recordings", file=sys.stderr)
    recordings = survey_recordings(args.root, subjects, progress=progress)
    epochs = load_epochs(args.root, args.task, subjects, tmin=args.tmin, tmax=args.tmax, l_freq=args.l_freq,
                         h_freq=args.h_freq, include_rest=args.include_rest, cache_dir=args.cache_dir,
                         progress=progress)
    summary = write_report(epochs, out, recordings=recordings, dpi=args.dpi)
    print(f"{summary['n_trials']} trials from {summary['n_subjects']} subjects; report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
