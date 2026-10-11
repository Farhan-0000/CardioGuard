#!/usr/bin/env python3
"""
Results figure for the IEEE paper: what the judged responses of each condition contain.

Each bar splits one condition's LLM-judged test responses into three parts:

- responses with at least one UNSUPPORTED or CONTRADICTED claim (= the response-level
  hallucination rate, anchored at the baseline so the lengths compare directly);
- responses whose clinical claims were all SUPPORTED;
- responses with no clinical claim (templates, clarifying questions, small talk).

The number to the right of each bar is the hallucination rate among responses that
contain claims. Every value is read from ``evaluation/results/summary.json`` (written by
``python -m evaluation.analyze``): ``per_condition.<C>.llm.n_judged``,
``.llm.hallucination.k`` and ``.llm.hallucination_given_claims.{k,n}``. Only the LLM judge
appears: it scored all ten conditions, the NLI judge (stopped early) did not.

The figure exists because ``reports/figures/hallucination_rate_by_condition.png`` shows the
partial C2 NLI subset (n = 76) as if it were a full-condition result, and because the
split by claim content is the finding that Section V has to make visible.

Usage::

    python report/paper/figures/make_results_figure.py     # -> report/paper/figures/response_composition.{png,svg}
    python report/paper/figures/make_results_figure.py --summary PATH --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIG_DIR.parents[2]
SUMMARY = REPO_ROOT / "evaluation" / "results" / "summary.json"

ORDER = ["C0", "C1", "C1G", "C2", "C3", "C4", "C5", "V1", "V2", "V12"]
LABELS = {
    "C0": "C0 direct LLM",
    "C1": "C1 vanilla RAG",
    "C1G": "C1G RAG, prod. prompt",
    "C2": "C2 RAG + safety prompt",
    "C3": "C3 Context Gate only",
    "C4": "C4 Evidence Gate only",
    "C5": "C5 MedBridge",
    "V1": "V1 calibrated Gate 2",
    "V2": "V2 + Gate 3 verifier",
    "V12": "V12 V1 + V2",
}
# Semantic roles (reference palette of the dataviz method): hallucinated = orange,
# fully supported = blue, no claims = recessive neutral.
HALLUCINATED = "#eb6834"
SUPPORTED = "#2a78d6"
NO_CLAIMS = "#bcbbb4"
INK = "#1d1d1b"
MUTED = "#52514e"
FONT = 6.2


def composition(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Per condition: shares of hallucinated / fully supported / claim-free responses."""
    rows: list[dict[str, Any]] = []
    per = summary.get("per_condition") or {}
    for cond in ORDER:
        llm = (per.get(cond) or {}).get("llm") or {}
        if not llm:
            continue
        n = int(llm["n_judged"])
        halluc = int(llm["hallucination"]["k"])
        given = llm["hallucination_given_claims"]
        with_claims = int(given["n"])
        if int(given["k"]) != halluc or not 0 <= halluc <= with_claims <= n:
            raise ValueError(f"{cond}: inconsistent counts in summary.json")
        rows.append(
            {
                "condition": cond,
                "n": n,
                "hallucinated": halluc / n,
                "supported": (with_claims - halluc) / n,
                "no_claims": (n - with_claims) / n,
                "given_claims": float(given["rate"]),
                "with_claims": with_claims,
            }
        )
    if not rows:
        raise ValueError("summary.json has no per-condition LLM-judge results")
    return rows


def response_composition(summary_path: Path = SUMMARY, out_dir: Path = FIG_DIR) -> list[Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = composition(summary)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "hatch.color": "white",
            "hatch.linewidth": 0.4,
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    fig.subplots_adjust(left=0.315, right=0.85, top=0.83, bottom=0.145)

    ys = list(range(len(rows)))[::-1]
    height = 0.66
    for y, r in zip(ys, rows, strict=True):
        left = 0.0
        for key, color, hatch in (
            ("hallucinated", HALLUCINATED, "/////"),
            ("supported", SUPPORTED, None),
            ("no_claims", NO_CLAIMS, None),
        ):
            width = 100 * r[key]
            ax.barh(
                y,
                width,
                left=left,
                height=height,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                hatch=hatch,
            )
            left += width
        ax.text(
            100 * r["hallucinated"] - 1.2,
            y,
            f"{100 * r['hallucinated']:.1f}",
            ha="right",
            va="center",
            fontsize=FONT - 0.6,
            color="white",
            fontweight="bold",
            bbox={"boxstyle": "square,pad=0.08", "fc": HALLUCINATED, "ec": "none"},
        )
        ax.text(
            103,
            y,
            f"{100 * r['given_claims']:.1f}% of {r['with_claims']}",
            ha="left",
            va="center",
            fontsize=FONT - 0.4,
            color=INK,
            clip_on=False,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS.get(r["condition"], r["condition"]) for r in rows], fontsize=FONT)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=FONT, colors=MUTED, length=2)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlabel("Share of judged test responses (%)", fontsize=FONT, color=MUTED, labelpad=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9a9992")
    ax.spines["bottom"].set_linewidth(0.5)
    ax.text(
        103,
        len(rows) - 0.35,
        "halluc. among\nclaim-bearing",
        ha="left",
        va="bottom",
        fontsize=FONT - 0.6,
        color=MUTED,
        clip_on=False,
    )
    handles = [
        Patch(
            facecolor=HALLUCINATED, edgecolor="white", hatch="/////", label="≥1 unsupported claim"
        ),
        Patch(facecolor=SUPPORTED, label="all claims supported"),
        Patch(facecolor=NO_CLAIMS, label="no clinical claim"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.995),
        ncol=3,
        fontsize=FONT,
        frameon=False,
        handlelength=1.3,
        handleheight=0.9,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "response_composition.png", out_dir / "response_composition.svg"]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--summary", type=Path, default=SUMMARY, help="summary.json from evaluation.analyze"
    )
    parser.add_argument("--out", type=Path, default=FIG_DIR, help="output directory")
    args = parser.parse_args(argv)
    for path in response_composition(args.summary, args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
