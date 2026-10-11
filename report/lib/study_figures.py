"""
Report-specific versions of hallucination-study figures.

``evaluation/analyze.py`` draws every NLI rate it finds in ``summary.json``. The NLI judge was
concluded early: it scored C0, C1 and C1G completely but C2 only on a non-random subset of
answerable responses. That partial rate must not appear as a condition result, so this module
redraws the hallucination-rate figure from ``summary.json`` with the analysis plotting code,
showing NLI bars only for conditions the NLI judge scored completely and labelling partial
coverage instead of drawing it. Nothing is typed in by hand: every bar comes from the file.

Usage (from the repository root)::

    .venv-mac/bin/python report/lib/study_figures.py            # writes report/figures/study_*
    .venv-mac/bin/python report/lib/study_figures.py --summary path/to/summary.json --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = REPO_ROOT / "evaluation" / "results" / "summary.json"
DEFAULT_OUT = REPO_ROOT / "report" / "figures"
HALLUCINATION_FIGURE = "study_hallucination_rate_by_condition"

CONDITION_ORDER = ("C0", "C1", "C1G", "C2", "C3", "C4", "C5", "V1", "V2", "V12")


def nli_coverage(m: dict[str, Any] | None) -> tuple[str, int]:
    """Coverage of the NLI judge for one condition of ``summary.per_condition``.

    ``("complete", n)`` when the NLI judge scored every response the LLM judge scored,
    ``("partial", n)`` when it scored only a subset (the NLI run was concluded early), and
    ``("none", 0)`` when it did not score the condition on this split. A partial NLI rate
    comes from a non-random subset and must never be shown as a condition result.

    ``judge_coverage.nli`` (written by ``evaluation/analyze.py`` since a judge's rates are
    reported only above 95 per cent coverage) takes precedence: ``reported`` means complete,
    otherwise ``judged`` responses form a partial subset whose rates are blank.
    """
    cov = ((m or {}).get("judge_coverage") or {}).get("nli")
    if isinstance(cov, dict) and "judged" in cov:
        judged = int(cov.get("judged") or 0)
        if judged <= 0:
            return "none", 0
        return ("complete" if cov.get("reported") else "partial"), judged
    nli = (m or {}).get("nli")
    if not isinstance(nli, dict) or not nli.get("n_judged"):
        return "none", 0
    n_nli = int(nli.get("n_judged") or 0)
    n_llm = int(((m or {}).get("llm") or {}).get("n_judged") or 0)
    return ("complete" if n_nli >= n_llm else "partial"), n_nli


def hallucination_series(
    summary: dict[str, Any],
) -> tuple[list[str], dict[str, list[dict[str, Any] | None]], dict[str, int]]:
    """Conditions, the LLM / NLI series (partial NLI rates masked) and the partial NLI sizes."""
    per = summary.get("per_condition") or {}
    conds = [c for c in CONDITION_ORDER if c in per] + [c for c in per if c not in CONDITION_ORDER]
    llm = [((per[c].get("llm") or {}).get("hallucination")) for c in conds]
    nli: list[dict[str, Any] | None] = []
    partial: dict[str, int] = {}
    for c in conds:
        status, n = nli_coverage(per[c])
        if status == "complete":
            nli.append((per[c].get("nli") or {}).get("hallucination"))
        else:
            nli.append(None)
            if status == "partial":
                partial[c] = n
    series: dict[str, list[dict[str, Any] | None]] = {"LLM judge": llm}
    if any(v is not None for v in nli) or partial:
        series["NLI judge (complete conditions only)"] = nli
    return conds, series, partial


def build_hallucination_figure(summary: dict[str, Any], out_dir: Path) -> list[Path]:
    """Write ``study_hallucination_rate_by_condition.{png,svg}``; returns the paths."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from evaluation import plots  # analysis plotting style (matplotlib, Agg backend)

    conds, series, partial = hallucination_series(summary)
    note = f"{summary.get('split', 'test')} split · 95% Wilson CIs · unit: item-turn"
    if partial:
        note += " · NLI judge concluded early: " + ", ".join(
            f"{c} partial subset (n = {n}), not shown" for c, n in partial.items()
        )
    fig = plots.rate_figure(
        conds,
        series,
        title="Response-level hallucination rate by condition",
        ylabel="Responses with ≥1 unsupported claim (%)",
        note=note,
    )
    ax = fig.axes[0]
    n_series = len(series)
    width = min(0.78 / max(1, n_series), 0.32)
    offset = (n_series - 1 - (n_series - 1) / 2) * width  # slot of the last (NLI) series
    for c, n in partial.items():
        x = conds.index(c) + offset
        ax.annotate(
            f"partial subset, n = {n}",
            (x, 9),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=6.0,
            color=plots.INK_MUTED,
        )
    return plots.save_figure(fig, out_dir, HALLUCINATION_FIGURE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    for path in build_hallucination_figure(summary, args.out):
        print(f"[study_figures] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
