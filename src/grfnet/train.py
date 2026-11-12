"""Train a model: python -m grfnet.train CONFIG."""

from __future__ import annotations

import argparse
import signal
from collections.abc import Callable, Sequence
from pathlib import Path

# Exit status after a requested stop, as for a process ended by Ctrl+C.
STOPPED = 130


def install_stop_handler(notify: Callable[[str], object] = print) -> Callable[[], bool]:
    """First Ctrl+C finishes the current fold and stops; a second aborts at once. Returns the stop check."""
    state = {"requested": False}

    def handler(signum, frame):
        if state["requested"]:
            signal.signal(signal.SIGINT, signal.default_int_handler)
            raise KeyboardInterrupt
        state["requested"] = True
        notify("stop requested: finishing the current fold, then exiting (press Ctrl+C again to abort now)")

    signal.signal(signal.SIGINT, handler)
    return lambda: state["requested"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m grfnet.train",
                                     description="Train and test a model with subject-disjoint folds. "
                                                 "Ctrl+C stops after the current fold; running the same "
                                                 "command again resumes.")
    parser.add_argument("config", help="TOML experiment config, e.g. configs/grfnet_eegmmidb_imagery_lr.toml")
    parser.add_argument("--out", default=None, help="run folder (default: runs/<run name>)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set training.epochs=5 (repeatable)")
    parser.add_argument("--folds", default=None, help="fold numbers to run, e.g. 0 or 0,2 (default: all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing run folder instead of resuming it")
    parser.add_argument("--no-progress", action="store_true",
                        help="hide progress bars and print one line per epoch instead")
    args = parser.parse_args(argv)

    from ._progress import use_bars, writer
    from .experiments.config import load_config
    from .experiments.runner import RunStopped, run_experiment

    config = load_config(args.config, args.overrides)
    folds = [int(f) for f in args.folds.split(",")] if args.folds else None
    out = Path(args.out) if args.out else Path("runs") / config.run_name
    progress = False if args.no_progress else None
    previous = signal.getsignal(signal.SIGINT)
    stop = install_stop_handler(writer(print, use_bars(progress)))
    try:
        run_experiment(config, out, folds=folds, overwrite=args.overwrite, progress=progress, stop=stop)
    except RunStopped as stopped:
        print(stopped)
        return STOPPED
    finally:
        signal.signal(signal.SIGINT, previous)
    print(f"results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
