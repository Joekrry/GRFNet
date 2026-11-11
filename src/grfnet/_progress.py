"""Optional tqdm progress bars."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Iterable


def use_bars(progress: bool | None) -> bool:
    """``None`` means draw bars only when tqdm is installed and stderr is a terminal."""
    if progress is False or importlib.util.find_spec("tqdm") is None:
        return False
    return progress is True or sys.stderr.isatty()


def bar(iterable: Iterable | None = None, *, enabled: bool, **kwargs):
    if enabled:
        from tqdm.auto import tqdm

        return tqdm(iterable, dynamic_ncols=True, **kwargs)
    return _NoBar(iterable)


def writer(log: Callable[[str], object], enabled: bool) -> Callable[[str], object]:
    # print() would break an active bar.
    if enabled and log is print:
        from tqdm.auto import tqdm

        return tqdm.write
    return log


class _NoBar:
    def __init__(self, iterable: Iterable | None):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable if self._iterable is not None else ())

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def set_description(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None
