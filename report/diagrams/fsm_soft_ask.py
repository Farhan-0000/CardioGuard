"""
State machine of the SOFT-ASK loop-breaker counter (``soft_ask_count``).

Source of truth: ``PipelineOrchestrator._stage_context_gate`` and
``_stage_load_state`` in ``medbridge/core/orchestrator.py``, ``persist_turn`` in
``medbridge/state/turn_store.py`` and ``SOFT_ASK_MAX_COUNT = 2`` in
``medbridge/config.py``:

* Gate 1 PROCEED (also the Gate 1 failure fallback and the C0/C1/C1G/C2/C4 ablation
  bypass) → counter reset to 0, the Evidence Gate decides;
* Gate 1 SOFT-ASK with ``count < 2`` → SOFT-ASK reply, counter + 1;
* Gate 1 SOFT-ASK with ``count ≥ 2`` → loop-breaker: forced GENERALIZE, Evidence
  Gate skipped (retrieval still runs), counter reset to 0;
* turns that never reach Gate 1 (emergency fast-path, small talk) do not touch
  the counter (``TurnRecord.update_soft_ask_count = False``);
* regenerate starts from ``soft_ask_count_before`` recorded in the replaced
  turn's trace, so a regenerate never double-counts.

Run: ``python report/diagrams/fsm_soft_ask.py`` → ``report/figures/fsm_soft_ask.{png,svg}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import PALETTE, SUBTEXT, save_matplotlib
from _mpl import Box, Canvas
from matplotlib.patches import Circle, FancyArrowPatch

NAME = "fsm_soft_ask"
AMBER = PALETTE["gate"].line
GREEN = PALETTE["safe"].line
GREY = "#4A5568"

W, H = 7.4, 4.3
R = 0.47
Y = 1.62


def build() -> list[Path]:
    c = Canvas(W, H)
    ax = c.ax
    k = PALETTE["det"]
    centres = {0: (1.55, Y), 1: (3.95, Y), 2: (6.35, Y)}
    subs = {0: "no open question", 1: "1 SOFT-ASK asked", 2: "2 SOFT-ASKs asked"}
    for n, (x, y) in centres.items():
        ax.add_patch(Circle((x, y), R, facecolor=k.fill, edgecolor=k.line, lw=1.5, zorder=3))
        t1 = ax.text(
            x,
            y - 0.06,
            f"count = {n}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=k.text,
            zorder=4,
        )
        t2 = ax.text(
            x, y + 0.13, subs[n], ha="center", va="center", fontsize=6.6, color=SUBTEXT, zorder=4
        )
        inner = Box(x, y, 2 * R * 0.92, 2 * R * 0.7)
        c.texts += [(t1, inner, f"count={n}"), (t2, inner, f"count={n}")]

    def curve(
        p0: tuple[float, float], p1: tuple[float, float], rad: float, color: str, lw: float = 1.2
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=10,
                color=color,
                lw=lw,
                zorder=2.5,
                shrinkA=0,
                shrinkB=0,
            )
        )

    # start
    ax.plot([0.35], [Y], marker="o", ms=7, color="#1A202C", zorder=3)
    c.arrow([(0.42, Y), (centres[0][0] - R, Y)], color=GREY)
    c.label(0.75, Y - 0.12, "new\nsession", size=6.8, color=GREY, bg=False)

    # SOFT-ASK increments (straight, left → right)
    for a, b in ((0, 1), (1, 2)):
        xa, xb = centres[a][0] + R, centres[b][0] - R
        c.arrow([(xa, Y), (xb, Y)], color=AMBER, lw=1.3)
        c.label(
            (xa + xb) / 2, Y - 0.16, "Gate 1: SOFT-ASK", color=AMBER, size=7.0, bold=True, bg=False
        )
        c.label(
            (xa + xb) / 2,
            Y + 0.13,
            "reply SOFT-ASK\ncount + 1",
            color=AMBER,
            size=6.8,
            bg=False,
            va="top",
        )

    # loop-breaker: count = 2, SOFT-ASK → forced GENERALIZE, reset (arc below)
    x2, x0 = centres[2][0], centres[0][0]
    curve((x2 - 0.12, Y + R - 0.03), (x0 + 0.12, Y + R - 0.03), rad=-0.32, color=AMBER, lw=1.9)
    c.label(
        (x0 + x2) / 2,
        Y + 1.32,
        "Gate 1: SOFT-ASK with count ≥ 2  →  loop-breaker",
        color=AMBER,
        size=7.2,
        bold=True,
    )
    c.label(
        (x0 + x2) / 2,
        Y + 1.50,
        "forced GENERALIZE · Evidence Gate skipped (retrieval still runs) · count → 0",
        color=AMBER,
        size=6.8,
    )

    # PROCEED resets (arcs above, right → left)
    x1 = centres[1][0]
    curve((x1 - 0.2, Y - R + 0.04), (x0 + 0.2, Y - R + 0.04), rad=0.42, color=GREEN)
    c.label((x0 + x1) / 2, Y - 0.63, "Gate 1: PROCEED*\ncount → 0", color=GREEN, size=6.9, bg=False)
    curve((x2 - 0.12, Y - R + 0.02), (x0 + 0.12, Y - R + 0.02), rad=0.36, color=GREEN)
    c.label(
        (x0 + x2) / 2 + 0.6,
        Y - 1.43,
        "Gate 1: PROCEED*  ·  count → 0  ·  Evidence Gate routes the turn",
        color=GREEN,
        size=6.9,
    )
    # self loop on count = 0 (PROCEED keeps it at 0)
    ang0, ang1 = 145, 100
    import math

    def on_circle(n: int, deg: float) -> tuple[float, float]:
        x, y = centres[n]
        return (x + R * math.cos(math.radians(deg)), y - R * math.sin(math.radians(deg)))

    curve(on_circle(0, ang0), on_circle(0, ang1), rad=-1.9, color=GREEN)
    c.label(0.66, Y - 0.93, "Gate 1: PROCEED*", color=GREEN, size=6.9, bg=False)

    # notes
    notes = [
        (
            "Counter unchanged",
            [
                "Turns that never reach Gate 1 keep the",
                "stored value: emergency fast-path",
                "ESCALATE, small-talk template.",
            ],
        ),
        (
            "Regenerate",
            [
                "Re-runs the last user message from",
                "soft_ask_count_before (value recorded",
                "before that turn): restored, never +2.",
            ],
        ),
        (
            "* PROCEED also covers",
            [
                "the Gate 1 failure fallback and the",
                "Gate 1 bypass in C0, C1, C1G, C2, C4.",
                "Threshold SOFT_ASK_MAX_COUNT = 2.",
            ],
        ),
    ]
    nw = 2.3
    for i, (title, lines) in enumerate(notes):
        c.box(
            0.2 + nw / 2 + i * (nw + 0.15),
            3.85,
            nw,
            0.62,
            title,
            lines,
            kind="note",
            title_size=7.6,
            sub_size=6.7,
        )
    c.check()
    return save_matplotlib(c.fig, NAME)


if __name__ == "__main__":
    for p in build():
        print(p.relative_to(Path(__file__).resolve().parents[2]))
