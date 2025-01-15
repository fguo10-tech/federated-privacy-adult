"""Matplotlib helpers with one consistent, colour-blind-safe visual language.

Colour carries *role*, never rank:

* ``BEFORE_COLOR`` (blue) — every "Before" / baseline method, in every track.
* ``AFTER_COLOR`` (orange) — every "After" method (the method under study).
* ``REFERENCE_COLOR`` (grey) — reference lines, ablations and anything not run.
* ``SPLIT_COLORS`` — the two federated client splits (workclass, IID control).

Method identity is carried by axis labels, markers and direct labels, so no
chart relies on telling many hues apart. The blue / orange pair and the violet /
aqua pair were checked for colour-vision-deficiency separation (protan, deutan,
tritan) and contrast against the light chart surface.

Bars always start at zero, every bar carries its value, error bars are the
sample standard deviation across seeds, and no chart uses two y-scales.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

BEFORE_COLOR = "#2a78d6"
AFTER_COLOR = "#eb6834"
REFERENCE_COLOR = "#898781"
SPLIT_COLORS = {"workclass": "#6250d6", "iid": "#1baf7a"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def use_headless_backend() -> None:
    """Select the non-interactive Agg backend. Call from scripts, not from notebooks."""
    matplotlib.use("Agg", force=True)


def apply_style() -> None:
    """Apply the repository chart style to matplotlib's global rcParams."""
    matplotlib.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.5,
            "figure.dpi": 100,
        }
    )


def save_figure(fig: Figure, path: str | Path, dpi: int = 160) -> Path:
    """Save ``fig`` as PNG (creating parent folders), close it and return the path."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def format_number(value: float, digits: int = 3) -> str:
    """Compact value label: fixed decimals, no scientific notation for ordinary magnitudes."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "TBD"
    if math.isinf(value):
        return "∞"
    return f"{value:.{digits}f}"


def bar_panel(
    ax: Axes,
    labels: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float | None],
    colors: Sequence[str],
    ylabel: str,
    title: str,
    digits: int = 3,
) -> None:
    """Draw one bar per label with error bars and a value label on every bar.

    A missing mean (``nan``) draws no bar and prints ``TBD`` at the baseline, so an
    experiment that has not been run can never look like a measured zero. An infinite
    mean (for example a privacy budget of ``∞`` for a run without differential privacy)
    draws no bar either and prints ``∞``.
    """
    positions = np.arange(len(labels))
    heights = [0.0 if _undrawable(m) else float(m) for m in means]
    errors = [0.0 if (s is None or _missing(s) or _undrawable(m)) else float(s)
              for m, s in zip(means, stds, strict=True)]  # fmt: skip
    ax.bar(positions, heights, width=0.62, color=list(colors), linewidth=0)
    drawn = [i for i, m in enumerate(means) if not _undrawable(m) and errors[i] > 0]
    if drawn:
        ax.errorbar(
            positions[drawn],
            [heights[i] for i in drawn],
            yerr=[errors[i] for i in drawn],
            fmt="none",
            ecolor=INK_SECONDARY,
            elinewidth=1.0,
            capsize=3,
        )
    top = max([h + e for h, e in zip(heights, errors, strict=True)] + [0.0])
    ax.set_ylim(0, top * 1.22 if top > 0 else 1.0)
    for i, mean in enumerate(means):
        text = format_number(float("nan") if _missing(mean) else float(mean), digits)
        ax.annotate(
            text,
            (positions[i], heights[i] + errors[i]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=28 if max(map(len, labels), default=0) > 7 else 0,
                       ha="right" if max(map(len, labels), default=0) > 7 else "center")  # fmt: skip
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.tick_params(axis="x", length=0)


def line_with_band(
    ax: Axes,
    x: Sequence[float],
    mean: Sequence[float],
    std: Sequence[float] | None,
    label: str,
    color: str,
    marker: str = "o",
    linestyle: str = "-",
) -> None:
    """Plot a mean line with markers and a translucent ± one standard deviation band."""
    x_array = np.asarray(x, dtype=float)
    mean_array = np.asarray(mean, dtype=float)
    ax.plot(x_array, mean_array, color=color, marker=marker, linestyle=linestyle, label=label,
            markeredgecolor=SURFACE, markeredgewidth=1.2)  # fmt: skip
    if std is not None:
        std_array = np.nan_to_num(np.asarray(std, dtype=float), nan=0.0)
        ax.fill_between(x_array, mean_array - std_array, mean_array + std_array,
                        color=color, alpha=0.15, linewidth=0)  # fmt: skip


@dataclass(frozen=True)
class Bar:
    """One bar of a Before / After panel."""

    method: str
    role: str  # "before" | "after" | "reference"
    mean: float | None
    std: float | None = None


@dataclass(frozen=True)
class BeforeAfterPanel:
    """One metric of a Before / After chart: its bars share one unit and one axis."""

    metric: str
    unit: str
    direction: str
    bars: tuple[Bar, ...]
    digits: int = 3


ROLE_COLORS = {"before": BEFORE_COLOR, "after": AFTER_COLOR, "reference": REFERENCE_COLOR}


def before_after_figure(
    panels: Sequence[BeforeAfterPanel],
    path: str | Path,
    title: str,
    source_label: str,
    caption: str = "",
    columns: int = 3,
) -> Path:
    """Grouped Before / After chart as small multiples: one panel per metric.

    Metrics have different units and scales, so each gets its own axis instead of
    sharing one misleading axis. Before bars are blue and After bars are orange in
    every panel of every track. The source label is part of the figure title.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    apply_style()
    count = len(panels)
    if count == 0:
        raise ValueError("before_after_figure needs at least one panel")
    ncols = min(columns, count)
    nrows = math.ceil(count / ncols)
    widest = max(len(panel.bars) for panel in panels)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(max(3.4, 0.62 * widest + 1.4) * ncols, 3.3 * nrows + 0.9),
        squeeze=False,
    )  # fmt: skip
    for index, panel in enumerate(panels):
        ax = axes[index // ncols][index % ncols]
        bar_panel(
            ax,
            [bar.method for bar in panel.bars],
            [math.nan if bar.mean is None else bar.mean for bar in panel.bars],
            [bar.std for bar in panel.bars],
            [ROLE_COLORS[bar.role] for bar in panel.bars],
            ylabel=f"{panel.metric} ({panel.unit})" if panel.unit else panel.metric,
            title=f"{panel.metric} — {panel.direction.lower()}",
            digits=panel.digits,
        )
    for index in range(count, nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)
    handles = [Patch(color=BEFORE_COLOR, label="Before"), Patch(color=AFTER_COLOR, label="After")]
    if any(bar.role == "reference" for panel in panels for bar in panel.bars):
        handles.append(Patch(color=REFERENCE_COLOR, label="Reference"))
    fig.legend(
        handles=handles, loc="upper right", ncols=len(handles), bbox_to_anchor=(0.995, 0.995)
    )
    fig.suptitle(f"{title} {source_label}", x=0.01, ha="left", fontsize=11.5, fontweight="bold",
                 color=INK)  # fmt: skip
    if caption:
        fig.text(0.01, 0.005, caption, ha="left", va="bottom", fontsize=8, color=INK_SECONDARY,
                 wrap=True)  # fmt: skip
    fig.tight_layout(rect=(0, 0.04 if caption else 0, 1, 0.95))
    return save_figure(fig, path)


def _missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _undrawable(value: float | None) -> bool:
    """True when no bar can be drawn: the value is missing or infinite."""
    return _missing(value) or math.isinf(float(value))  # type: ignore[arg-type]
