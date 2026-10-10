#!/usr/bin/env python3
"""
Figures drawn specifically for the IEEE paper (single-column width, 300 dpi PNG + SVG).

The report's architecture diagram (``report/figures/architecture_pipeline.png``) is a
full-page figure; a conference column is 3.5 in wide, so the paper uses this compact
redraw of the same pipeline. It is a schematic, not a result: no number in it comes
from an experiment (temperatures, k and the loop-breaker limit are configuration
values from ``medbridge/config.py`` and ``medbridge/retrieval/hybrid_retriever.py``).

Usage::

    python report/paper/figures/make_figures.py            # -> report/paper/figures/gate_pipeline.{png,svg}
    python report/paper/figures/make_figures.py --out DIR

Every label is measured against its box after drawing; the script raises if a label
overflows, so layout regressions are caught when the figure is rebuilt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parent

# category -> (fill, edge)
COLORS: dict[str, tuple[str, str]] = {
    "io": ("#FFFFFF", "#222222"),
    "rule": ("#ECEFF3", "#5A6372"),
    "emergency": ("#F8D7D7", "#B83232"),
    "llm": ("#DCE8F8", "#2B5A9E"),
    "gate": ("#FDE9B5", "#A86A12"),
    "template": ("#DDF1E1", "#2F7D45"),
    "optional": ("#FFF7E0", "#A86A12"),
}
FONT = 6.0
SMALL = 5.1
LINE_H = 0.094  # inches per text line at FONT


@dataclass
class Node:
    key: str
    text: str
    kind: str
    cx: float
    cy: float
    w: float
    h: float
    dashed: bool = False

    @property
    def left(self) -> float:
        return self.cx - self.w / 2

    @property
    def right(self) -> float:
        return self.cx + self.w / 2

    @property
    def top(self) -> float:
        return self.cy - self.h / 2

    @property
    def bottom(self) -> float:
        return self.cy + self.h / 2


def _height(text: str) -> float:
    return 0.10 + LINE_H * (text.count("\n") + 1)


def gate_pipeline(out_dir: Path = FIG_DIR) -> list[Path]:
    """Compact MedBridge routing pipeline for one column (3.45 in wide)."""
    width = 3.45
    main_x, main_w = 1.00, 1.88
    side_x, side_w = 2.84, 1.10
    gap = 0.13

    main_specs = [
        ("msg", "Patient message", "io"),
        ("redact", "Privacy redaction (deterministic rules)", "rule"),
        ("emerg", "Emergency fast-path\nred-flag rules on raw text, no LLM", "emergency"),
        ("extract", "Context extraction (LLM)\ntyped events, state = fold(events)", "llm"),
        ("gate1", "Gate 1: Context Gate (T = 0)\nSOFT-ASK  |  PROCEED", "gate"),
        ("retr", "Hybrid retrieval: dense + BM25\nRRF (k = 60), rerank to top-5", "rule"),
        (
            "gate2",
            "Gate 2: Evidence Gate (T = 0)\nANSWER | GENERALIZE |\nABSTAIN | ESCALATE",
            "gate",
        ),
        ("gen", "Grounded generation (T = 0.3)\n[n] citations, citation check", "llm"),
        ("out", "Response + immutable audit log", "io"),
    ]
    nodes: dict[str, Node] = {}
    y = 0.08
    for key, text, kind in main_specs:
        h = _height(text)
        nodes[key] = Node(key, text, kind, main_x, y + h / 2, main_w, h)
        y += h + gap
    height = y - gap + 0.08

    side_specs = [
        ("t_esc", "emerg", "ESCALATE template\n(instant, no LLM)", "template", False),
        ("ask", "gate1", "Clarifying question;\nafter 2 in a row:\nGENERALIZE", "llm", False),
        (
            "t_abs",
            "gate2",
            "ABSTAIN / ESCALATE\ntemplates\n(generator bypassed)",
            "template",
            False,
        ),
        ("g3", "gen", "Gate 3 (optional):\nNLI grounding\nverifier", "optional", True),
    ]
    for key, anchor, text, kind, dashed in side_specs:
        h = _height(text)
        nodes[key] = Node(key, text, kind, side_x, nodes[anchor].cy, side_w, h, dashed)

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    texts = []
    for node in nodes.values():
        fill, edge = COLORS[node.kind]
        ax.add_patch(
            FancyBboxPatch(
                (node.left, node.top),
                node.w,
                node.h,
                boxstyle="round,pad=0,rounding_size=0.04",
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.8,
                linestyle=(0, (3, 1.6)) if node.dashed else "solid",
            )
        )
        weight = "bold" if node.kind in {"gate", "emergency"} else "normal"
        t = ax.text(
            node.cx,
            node.cy,
            node.text,
            ha="center",
            va="center",
            fontsize=FONT,
            fontweight=weight,
            linespacing=1.15,
            color="#111111",
        )
        texts.append((t, node))

    def arrow(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        color: str = "#333333",
        dashed: bool = False,
        label: str = "",
        label_xy: tuple[float, float] | None = None,
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=6,
                linewidth=0.8,
                color=color,
                linestyle=(0, (3, 1.6)) if dashed else "solid",
                shrinkA=0,
                shrinkB=0,
            )
        )
        if label:
            lx, ly = label_xy or ((x0 + x1) / 2, (y0 + y1) / 2 - 0.05)
            ax.text(lx, ly, label, fontsize=SMALL, ha="center", va="bottom", color=color)

    order = [k for k, _, _ in main_specs]
    for a, b in pairwise(order):
        na, nb = nodes[a], nodes[b]
        label = {"emerg": "no red flag", "gate1": "PROCEED", "gate2": "ANSWER / GENERALIZE"}.get(
            a, ""
        )
        arrow(
            main_x,
            na.bottom,
            main_x,
            nb.top,
            label=label,
            label_xy=(main_x + 0.06, (na.bottom + nb.top) / 2 + 0.035) if label else None,
        )
    # the arrow labels sit to the right of the vertical arrows
    for t in ax.texts:
        if t.get_text() in {"no red flag", "PROCEED", "ANSWER / GENERALIZE"}:
            t.set_horizontalalignment("left")

    side_arrows = [
        ("emerg", "t_esc", "red flag", "#B83232", False),
        ("gate1", "ask", "SOFT-ASK", "#A86A12", False),
        ("gate2", "t_abs", "ABSTAIN/\nESCALATE", "#A86A12", False),
        ("gen", "g3", "draft", "#A86A12", True),
    ]
    for a, b, label, color, dashed in side_arrows:
        na, nb = nodes[a], nodes[b]
        arrow(
            na.right,
            na.cy,
            nb.left,
            nb.cy,
            color=color,
            dashed=dashed,
            label=label,
            label_xy=((na.right + nb.left) / 2, na.cy - 0.025),
        )
    # Gate 3 returns the verified (or downgraded) draft to the output
    g3, out = nodes["g3"], nodes["out"]
    arrow(g3.cx, g3.bottom, out.right, out.cy, color="#A86A12", dashed=True)
    # speculative retrieval note
    ax.text(
        side_x,
        nodes["retr"].cy,
        "retrieval starts with\nGate 1 (speculative);\ndiscarded on SOFT-ASK",
        fontsize=SMALL,
        ha="center",
        va="center",
        style="italic",
        color="#2B5A9E",
        linespacing=1.1,
    )

    # overflow check: every label must fit inside its box
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    inv = ax.transData.inverted()
    for t, node in texts:
        bb = t.get_window_extent(renderer=renderer)
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        if (x1 - x0) > node.w - 0.04 or abs(y1 - y0) > node.h - 0.02:
            raise ValueError(f"label of '{node.key}' overflows its box")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "gate_pipeline.png", out_dir / "gate_pipeline.svg"]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=FIG_DIR, help="output directory")
    args = parser.parse_args(argv)
    for path in gate_pipeline(args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
