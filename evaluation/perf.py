"""
Before/after latency table from ``reports/performance/*.json`` (files written
by ``scripts/latency_probe.py``: ``{label, cold: [...], warm: [...], warm_summary}``).

Rows are matched by scenario turn (``S4-t2``) because the action of a turn can
differ between the two versions; both actions are shown. Outputs
``reports/performance/latency_table.md`` (+ ``.csv``) and the figure
``latency_before_after.png/.svg`` (p50 bars, whiskers up to p95).

Usage::

    python -m evaluation.perf                         # newest 'before' vs newest 'after' file
    python -m evaluation.perf --before reports/performance/before_gemma3-4b.json \\
                              --after reports/performance/after_qwen3-8b.json
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evaluation import plots
from evaluation.io import read_json, write_table
from evaluation.paths import FIGURES_DIR, PERFORMANCE_DIR

SCENARIO_NAMES: dict[str, str] = {
    "S1": "Emergency fast-path",
    "S2": "Out-of-domain",
    "S3": "Clarification gate",
    "S4": "Clinical guidance (2 turns)",
    "S5": "Lifestyle",
}


def latest_with_label(perf_dir: Path, label: str) -> Path | None:
    """Most recently modified probe file whose ``label`` field equals ``label``."""
    matches = []
    for path in perf_dir.glob("*.json"):
        try:
            data = read_json(path)
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("label") == label and "warm_summary" in data:
            matches.append(path)
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def by_turn(probe: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{"S4-t2": {"actions", "p50", "p95", "n", "cold"}}`` from one probe file."""
    out: dict[str, dict[str, Any]] = {}
    for key, stats in (probe.get("warm_summary") or {}).items():
        turn, _, action = key.partition(":")
        row = out.setdefault(turn, {"actions": [], "p50": [], "p95": [], "n": 0, "cold": None})
        row["actions"].append(action)
        row["p50"].append(float(stats["p50"]))
        row["p95"].append(float(stats["p95"]))
        row["n"] += int(stats.get("n", 0))
    for row in out.values():
        row["p50"] = statistics.median(row["p50"])
        row["p95"] = max(row["p95"])
    for cold in probe.get("cold") or []:
        turn = f"{cold['scenario']}-t{cold['turn']}"
        if turn in out:
            out[turn]["cold"] = float(cold["latency_s"])
    return out


def build_table(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows of the before/after table (seconds; speed-up = before p50 / after p50)."""
    b, a = by_turn(before), by_turn(after)
    rows = []
    for turn in sorted(set(b) | set(a)):
        rb, ra = b.get(turn), a.get(turn)
        speedup = (rb["p50"] / ra["p50"]) if rb and ra and ra["p50"] > 0 else None
        rows.append(
            {
                "Scenario turn": turn,
                "Scenario": SCENARIO_NAMES.get(turn.split("-")[0], ""),
                "Action before": "/".join(rb["actions"]) if rb else "—",
                "Action after": "/".join(ra["actions"]) if ra else "—",
                "Cold before s": _fmt(rb["cold"] if rb else None),
                "Cold after s": _fmt(ra["cold"] if ra else None),
                "Warm p50 before s": _fmt(rb["p50"] if rb else None),
                "Warm p95 before s": _fmt(rb["p95"] if rb else None),
                "Warm p50 after s": _fmt(ra["p50"] if ra else None),
                "Warm p95 after s": _fmt(ra["p95"] if ra else None),
                "Speed-up (p50)": f"{speedup:.1f}×" if speedup else "—",
            }
        )
    return rows


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2f}"


def write_outputs(
    before_path: Path, after_path: Path, out_dir: Path, figures_dir: Path
) -> list[Path]:
    """Table (md + csv) and figure."""
    before, after = read_json(before_path), read_json(after_path)
    rows = build_table(before, after)
    paths = write_table(
        rows,
        out_dir,
        "latency_table",
        f"Latency before vs after ({before_path.name} → {after_path.name}, warm runs)",
    )
    b, a = by_turn(before), by_turn(after)
    turns = sorted(set(b) | set(a))
    fig, ax = plots.new_figure(width=max(5.5, 0.9 * len(turns) + 2), height=3.6)
    series = {
        "before": [b[t]["p50"] if t in b else None for t in turns],
        "after": [a[t]["p50"] if t in a else None for t in turns],
    }
    errors = {
        "before": [(b[t]["p50"], b[t]["p95"]) if t in b else (None, None) for t in turns],
        "after": [(a[t]["p50"], a[t]["p95"]) if t in a else (None, None) for t in turns],
    }
    plots.grouped_bars(ax, turns, series, errors)
    ax.set_ylabel("Seconds (bar = warm p50, whisker = p95)")
    plots.set_title(ax, "End-to-end latency per scenario turn, before vs after optimisation")
    paths += plots.save_figure(fig, figures_dir, "latency_before_after")
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Before/after latency table and figure")
    parser.add_argument("--perf-dir", type=Path, default=PERFORMANCE_DIR)
    parser.add_argument("--before", type=Path, default=None)
    parser.add_argument("--after", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)
    before = args.before or latest_with_label(args.perf_dir, "before")
    after = args.after or latest_with_label(args.perf_dir, "after")
    if before is None or after is None:
        print(
            f"[perf] need a 'before' and an 'after' probe file in {args.perf_dir} "
            "(scripts/latency_probe.py --label before|after)"
        )
        return 2
    for path in write_outputs(before, after, args.perf_dir, args.figures_dir):
        print(f"[perf] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
