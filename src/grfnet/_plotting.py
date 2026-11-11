"""Figure style shared by the dataset and run reports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

# Categorical slots in fixed order, then text and grid colours.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#a9a8a3"


def figure(nrows=1, ncols=1, width=8.0, height=4.0, **kwargs):
    # Figure rather than pyplot: no global state, no GUI backend.
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False, **kwargs)
    for ax in axes.flat:
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    return fig, axes


def save(fig, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, facecolor="white")


def title(ax, text: str) -> None:
    ax.set_title(text, color=TEXT, fontsize=10, loc="left")


def label(ax, x: str | None = None, y: str | None = None) -> None:
    if x:
        ax.set_xlabel(x, color=MUTED, fontsize=9)
    if y:
        ax.set_ylabel(y, color=MUTED, fontsize=9)


def legend(ax, **kwargs) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, **kwargs)


def outside_legend(fig, ax) -> None:
    fig.legend(*ax.get_legend_handles_labels(), loc="outside right upper", frameon=False, fontsize=8,
               labelcolor=TEXT)


def class_label(name: str) -> str:
    return name.replace("_", " ")


def eeg_info(ch_names: Sequence[str], sfreq: float):
    import mne

    info = mne.create_info(list(ch_names), sfreq, "eeg")
    info.set_montage("standard_1020", match_case=False)
    return info
