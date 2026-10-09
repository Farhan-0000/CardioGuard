"""
Figure helpers for the report (PNG 300 dpi + SVG).

One consistent style: system sans, recessive hairline grid, thin bars with a
surface-coloured gap, a legend whenever two or more series are shown, text in
ink colours (never the series colour). Categorical colours come from a
colour-blind-validated palette and are assigned in fixed slot order; condition
identity is carried by the x axis, not by colour. Sequential data (confusion
matrices) use a single blue ramp; polarity (waterfall deltas) uses the
blue ↔ red diverging pair.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# Categorical slots (validated: adjacent CVD ΔE ≥ 9.1, normal-vision ΔE ≥ 19.6 on the light surface).
SERIES: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL = "#b9b8b1"
POSITIVE_DELTA = "#2a78d6"  # reduction (good) — diverging cool pole
NEGATIVE_DELTA = "#e34948"  # increase (bad) — diverging warm pole
BLUE_RAMP = ("#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

_STYLE: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlepad": 8,
    "axes.labelsize": 9,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "text.color": INK,
    "svg.fonttype": "none",
}


def new_figure(
    width: float = 6.4, height: float = 3.6, ncols: int = 1, nrows: int = 1
) -> tuple[Figure, Any]:
    """Create a figure with the house style applied (the style stays active for later artists)."""
    plt.style.use(_STYLE)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=True)
    return fig, axes


def save_figure(fig: Figure, out_dir: Path, name: str) -> list[Path]:
    """Write ``<name>.png`` (300 dpi) and ``<name>.svg``; closes the figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.svg"]
    fig.tight_layout()
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def legend_above(ax: Axes, ncols: int, handles: Sequence[Any] | None = None) -> None:
    """Legend in one row between the title and the plot area (never over the data)."""
    kwargs: dict[str, Any] = {
        "loc": "lower left",
        "bbox_to_anchor": (0.0, 1.0),
        "ncols": max(1, ncols),
        "borderaxespad": 0.3,
        "handlelength": 1.2,
        "columnspacing": 1.2,
    }
    if handles is not None:
        kwargs["handles"] = list(handles)
    ax.legend(**kwargs)


def set_title(ax: Axes, title: str) -> None:
    """Left-aligned title; padded when a legend sits above the axes."""
    ax.set_title(title, loc="left", pad=22 if ax.get_legend() is not None else 8)


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _num(value: Any) -> float | None:
    """``float(value)`` when it is a finite number, else ``None``."""
    return float(value) if _finite(value) else None


def grouped_bars(
    ax: Axes,
    categories: Sequence[str],
    series: Mapping[str, Sequence[float | None]],
    errors: Mapping[str, Sequence[tuple[float | None, float | None]]] | None = None,
    *,
    percent: bool = False,
    value_labels: bool = True,
) -> None:
    """Grouped vertical bars, one colour per series (fixed slot order), optional CI whiskers."""
    n_series = max(1, len(series))
    group_width = 0.78
    width = min(group_width / n_series, 0.32)
    scale = 100.0 if percent else 1.0
    for s_index, (name, raw_values) in enumerate(series.items()):
        offset = (s_index - (n_series - 1) / 2) * width
        xs = [i + offset for i in range(len(categories))]
        values = [_num(v) for v in raw_values]
        bounds = (
            [(_num(lo), _num(hi)) for lo, hi in errors[name]] if errors and name in errors else []
        )
        ax.bar(
            xs,
            [v * scale if v is not None else 0.0 for v in values],
            width=width,
            color=SERIES[s_index % len(SERIES)],
            edgecolor=SURFACE,
            linewidth=1.0,
            label=name,
            zorder=2,
        )
        for x, value, (low, high) in zip(xs, values, bounds, strict=False):
            if value is not None and low is not None and high is not None:
                ax.errorbar(
                    x,
                    value * scale,
                    yerr=[[max(0.0, value - low) * scale], [max(0.0, high - value) * scale]],
                    fmt="none",
                    ecolor=INK_SECONDARY,
                    elinewidth=0.9,
                    capsize=2.5,
                    zorder=3,
                )
        if not value_labels:
            continue
        for i, (x, value) in enumerate(zip(xs, values, strict=True)):
            if value is None:
                ax.annotate(
                    "n/a",
                    (x, 0),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=INK_MUTED,
                )
                continue
            high = bounds[i][1] if bounds else None
            top = high if high is not None else value
            ax.annotate(
                f"{value * scale:.0f}" if percent else f"{value:.2f}",
                (x, top * scale),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=INK_SECONDARY,
                zorder=4,
            )
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_axisbelow(True)
    if n_series > 1:
        legend_above(ax, min(n_series, 4))


def rate_figure(
    categories: Sequence[str],
    series: Mapping[str, Sequence[Mapping[str, Any] | None]],
    *,
    title: str,
    ylabel: str,
    note: str = "",
) -> Figure:
    """Rates (Proportion dicts with rate/low/high) per condition with Wilson CI error bars, in %."""
    fig, ax = new_figure(width=max(5.0, 0.62 * len(categories) + 2.0), height=3.6)
    values = {k: [(p or {}).get("rate") for p in v] for k, v in series.items()}
    errors = {
        k: [((p or {}).get("low"), (p or {}).get("high")) for p in v] for k, v in series.items()
    }
    grouped_bars(ax, categories, values, errors, percent=True)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 110)
    set_title(ax, title)
    if note:
        fig.text(0.01, 0.005, note, fontsize=6.5, color=INK_MUTED, ha="left", va="bottom")
    return fig


def confusion_figure(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    title: str,
    xlabel: str = "Predicted",
    ylabel: str = "Gold",
) -> Figure:
    """Confusion matrix: colour = row-normalised share (blue ramp), cell text = count."""
    fig, ax = new_figure(width=5.2, height=4.2)
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    shares = []
    for row in matrix:
        total = sum(row)
        shares.append([c / total if total else 0.0 for c in row])
    ax.imshow(shares, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.grid(False)
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            ink = "#ffffff" if shares[i][j] > 0.55 else INK
            ax.text(j, i, str(count), ha="center", va="center", fontsize=8, color=ink)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    set_title(ax, title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def stacked_share_figure(
    categories: Sequence[str],
    shares: Mapping[str, Sequence[float]],
    title: str,
    ylabel: str,
    panels: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> Figure:
    """100 % stacked columns (segments separated by a surface gap); optional side-by-side panels."""
    groups = panels or {"": shares}
    fig, axes = new_figure(
        width=max(5.0, 0.55 * len(categories) * len(groups) + 2.5), height=3.6, ncols=len(groups)
    )
    axes_list = list(axes) if len(groups) > 1 else [axes]
    for ax, (panel_title, panel) in zip(axes_list, groups.items(), strict=True):
        bottoms = [0.0] * len(categories)
        for s_index, (label, values) in enumerate(panel.items()):
            heights = [100.0 * float(v) if _finite(v) else 0.0 for v in values]
            ax.bar(
                range(len(categories)),
                heights,
                bottom=bottoms,
                width=0.6,
                color=SERIES[s_index % len(SERIES)],
                edgecolor=SURFACE,
                linewidth=1.0,
                label=label,
                zorder=2,
            )
            bottoms = [b + h for b, h in zip(bottoms, heights, strict=True)]
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        if panel_title:
            ax.set_title(panel_title, loc="left", fontsize=9)
    axes_list[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
    return fig


def box_figure(
    groups: Mapping[str, Sequence[float]], title: str, ylabel: str, log: bool = False
) -> Figure:
    """Box plots (median line, IQR box, 1.5·IQR whiskers, outliers as small dots)."""
    names = [k for k, v in groups.items() if v]
    fig, ax = new_figure(width=max(5.0, 0.6 * len(names) + 2.0), height=3.6)
    if names:
        ax.boxplot(
            [list(groups[n]) for n in names],
            tick_labels=names,
            widths=0.45,
            patch_artist=True,
            boxprops={"facecolor": "#cde2fb", "edgecolor": SERIES[0], "linewidth": 1.0},
            medianprops={"color": INK, "linewidth": 1.4},
            whiskerprops={"color": INK_SECONDARY, "linewidth": 0.9},
            capprops={"color": INK_SECONDARY, "linewidth": 0.9},
            flierprops={
                "marker": "o",
                "markersize": 2.5,
                "markerfacecolor": INK_MUTED,
                "markeredgecolor": "none",
                "alpha": 0.7,
            },
        )
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    set_title(ax, title)
    ax.set_axisbelow(True)
    return fig


def waterfall_figure(
    start_label: str,
    start_value: float,
    steps: Sequence[tuple[str, float]],
    title: str,
    ylabel: str,
    end_label: str = "",
    extra: Sequence[tuple[str, float, float]] = (),
) -> Figure:
    """Waterfall in %: a start bar, cumulative delta bars (blue = reduction, red = increase),
    an optional end-level bar, then ``extra`` (label, base, delta) bars drawn from their own base."""
    from matplotlib.patches import Patch

    labels = [start_label]
    fig, ax = new_figure(width=max(5.5, 0.85 * (len(steps) + len(extra) + 2) + 1.5), height=3.9)
    _level_bar(ax, 0, start_value)
    level = start_value
    position = 1
    for label, delta in steps:
        _delta_bar(ax, position, level, delta)
        level += delta
        labels.append(label)
        position += 1
    if end_label:
        _level_bar(ax, position, level)
        labels.append(end_label)
        position += 1
    for label, base, delta in extra:
        _delta_bar(ax, position, base, delta)
        labels.append(label)
        position += 1
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.axhline(0, color=BASELINE, linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    legend_above(
        ax,
        3,
        [
            Patch(color=POSITIVE_DELTA, label="reduction"),
            Patch(color=NEGATIVE_DELTA, label="increase"),
            Patch(color=NEUTRAL, label="level"),
        ],
    )
    set_title(ax, title)
    return fig


def _level_bar(ax: Axes, x: int, value: float) -> None:
    ax.bar(x, value * 100, width=0.55, color=NEUTRAL, edgecolor=SURFACE, zorder=2)
    ax.annotate(
        f"{value * 100:.1f}",
        (x, value * 100),
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        fontsize=7,
        color=INK_SECONDARY,
    )


def _delta_bar(ax: Axes, x: int, base: float, delta: float) -> None:
    bottom = (base + min(delta, 0.0)) * 100
    height = abs(delta) * 100
    colour = POSITIVE_DELTA if delta < 0 else NEGATIVE_DELTA
    ax.bar(
        x,
        height if height > 0 else 0.3,
        bottom=bottom,
        width=0.55,
        color=colour,
        edgecolor=SURFACE,
        zorder=2,
    )
    ax.annotate(
        f"{delta * 100:+.1f}",
        (x, (base + max(delta, 0.0)) * 100),
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        fontsize=7,
        color=INK_SECONDARY,
    )


def scatter_curves_figure(
    points: Sequence[tuple[str, float, float]],
    curves: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    title: str,
    xlabel: str,
    ylabel: str,
) -> Figure:
    """Risk–coverage style plot: labelled points (one per condition) + optional curves (2 px)."""
    fig, ax = new_figure(width=5.6, height=4.0)
    for s_index, (name, (xs, ys)) in enumerate(curves.items()):
        ax.plot(
            [x * 100 for x in xs],
            [y * 100 for y in ys],
            color=SERIES[s_index % len(SERIES)],
            linewidth=2,
            solid_capstyle="round",
            label=name,
            zorder=2,
        )
    for label, x, y in points:
        if _finite(x) and _finite(y):
            ax.scatter(
                [x * 100], [y * 100], s=36, color=INK, edgecolors=SURFACE, linewidths=1.5, zorder=3
            )
            ax.annotate(
                label,
                (x * 100, y * 100),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7.5,
                color=INK_SECONDARY,
            )
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="both")
    if curves:
        legend_above(ax, len(curves))
    set_title(ax, title)
    return fig
