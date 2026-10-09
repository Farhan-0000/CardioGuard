"""
Model-selection experiment (CLAUDE_AGENT_PROMPT §2): which local model should
run the application (extractor + gates + generator)?

For every candidate model the full MedBridge pipeline (condition C5) is run on
the DEV split with ``python -m evaluation.run --model <m>`` in a separate
process (clean settings and singletons per model; the request cache keys
include the model, so namespaces never mix). Reported per model:

* JSON-schema validity per LLM role (``llm_calls``, uncached calls only — a
  memoised call does not reveal how many attempts it originally took): valid
  on the first attempt and valid at all (after the wrapper's repair/retry);
* routing accuracy (strict / lenient) vs gold and macro-F1;
* latency per turn (p50 / p95, cache-imputed) and per LLM role (uncached calls);
* effective generation speed (completion tokens / whole-call latency incl. prompt
  processing, median over uncached calls);
* model size, parameter count and quantisation (Ollama ``/api/tags``).

Usage::

    python -m evaluation.model_selection --models qwen3:8b,qwen3.5:4b,gemma3:4b --split dev --limit 40
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evaluation import metrics as mx
from evaluation import plots
from evaluation.datasets.schema import ACTIONS
from evaluation.degraded import record_degraded
from evaluation.io import load_run_records, write_json, write_table
from evaluation.manifest import ollama_models, utc_now
from evaluation.outcomes import impute_latency, uncached_call_medians
from evaluation.paths import DEFAULT_DATASET, FIGURES_DIR, MODEL_SELECTION_DIR

DEFAULT_MODELS: tuple[str, ...] = ("qwen3:8b", "qwen3.5:4b", "gemma3:4b")


def safe_name(model: str) -> str:
    """Filesystem-safe directory name for a model tag (``qwen3:8b`` → ``qwen3_8b``)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def run_model(
    model: str, split: str, limit: int | None, runs_root: Path, dataset: Path, fresh: bool = False
) -> int:
    """Run condition C5 for one model in a subprocess; returns its exit code."""
    cmd = [
        sys.executable,
        "-m",
        "evaluation.run",
        "--conditions",
        "C5",
        "--split",
        split,
        "--model",
        model,
        "--dataset",
        str(dataset),
        "--out",
        str(runs_root / safe_name(model)),
        "--fresh" if fresh else "--resume",
        # LLM failures are what this experiment measures: never trip the outage breaker
        "--max-consecutive-degraded",
        "0",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    print(f"[model_selection] {' '.join(cmd)}", file=sys.stderr, flush=True)
    return subprocess.run(cmd, check=False).returncode


def role_stats(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Schema validity, latency and speed per LLM role (``call_name``)."""
    calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        for call in rec.get("llm_calls") or []:
            calls[str(call.get("call_name"))].append(call)
    out: dict[str, dict[str, Any]] = {}
    for role, items in sorted(calls.items()):
        uncached = [c for c in items if not c.get("cached")]
        first = sum(1 for c in uncached if c.get("success") and int(c.get("attempts") or 0) == 1)
        valid = sum(1 for c in uncached if c.get("success"))
        fresh = [c for c in uncached if c.get("success") and c.get("latency_ms")]
        speeds = [
            float(c["completion_tokens"]) / (float(c["latency_ms"]) / 1000.0)
            for c in fresh
            if c.get("completion_tokens")
        ]
        latencies = [float(c["latency_ms"]) for c in fresh]
        out[role] = {
            "n_calls": len(items),
            "n_cached": len(items) - len(uncached),
            "valid_first_attempt": mx.proportion(first, len(uncached)).as_dict(),
            "valid_any_attempt": mx.proportion(valid, len(uncached)).as_dict(),
            "latency_p50_ms": mx.percentile(latencies, 50),
            "latency_p95_ms": mx.percentile(latencies, 95),
            "tokens_per_s_median": statistics.median(speeds) if speeds else None,
            "n_uncached": len(fresh),
        }
    return out


def summarize_model(
    model: str, records: Sequence[dict[str, Any]], info: dict[str, Any] | None
) -> dict[str, Any]:
    """All model-selection metrics for one model's C5 records."""
    valid = [r for r in records if not r.get("error") and r.get("action")]
    medians = uncached_call_medians(valid)
    latencies = [lat for lat, _ in (impute_latency(r, medians) for r in valid) if lat is not None]
    routing = (
        mx.routing_metrics(
            [str(r["gold_action"]) for r in valid],
            [str(r["action"]) for r in valid],
            [r.get("acceptable_actions") or [r["gold_action"]] for r in valid],
            labels=ACTIONS,
        )
        if valid
        else None
    )
    return {
        "model": model,
        "n_turns": len(records),
        "n_valid": len(valid),
        "n_errors": len(records) - len(valid),
        "n_degraded": sum(1 for r in records if record_degraded(r)),
        "routing": routing,
        "roles": role_stats(valid),
        "latency_turn_p50_ms": mx.percentile(latencies, 50),
        "latency_turn_p95_ms": mx.percentile(latencies, 95),
        "ollama": info or {},
    }


def write_outputs(summary: dict[str, Any], out_dir: Path, figures_dir: Path) -> list[Path]:
    """summary.json, comparison table and the three-panel figure."""
    paths = [out_dir / "summary.json"]
    write_json(paths[0], summary)
    rows = []
    models = summary["models"]
    for m in models:
        s = summary["results"][m]
        routing = s.get("routing") or {}
        roles = s.get("roles") or {}
        info = s.get("ollama") or {}
        rows.append(
            {
                "Model": m,
                "Params": info.get("parameter_size", "—"),
                "Quant": info.get("quantization_level", "—"),
                "Size GB": f"{info['size_bytes'] / 1e9:.1f}" if info.get("size_bytes") else "—",
                "Turns": s["n_valid"],
                "Routing strict %": _pct(routing.get("accuracy_strict")),
                "Routing lenient %": _pct(routing.get("accuracy_lenient")),
                "Macro-F1": f"{routing['macro_f1']:.2f}"
                if routing.get("macro_f1") is not None
                else "—",
                **{
                    f"Valid 1st try {role} %": _pct(v["valid_first_attempt"])
                    for role, v in roles.items()
                },
                "Turn p50 s": _seconds(s["latency_turn_p50_ms"]),
                "Turn p95 s": _seconds(s["latency_turn_p95_ms"]),
                "Gen tok/s (effective)": _num(
                    (roles.get("response_generator") or {}).get("tokens_per_s_median")
                ),
            }
        )
    paths += write_table(
        rows, out_dir, "model_selection", f"Model selection (C5, {summary['split']} split)"
    )

    fig, axes = plots.new_figure(width=10.5, height=3.6, ncols=3)
    acc: dict[str, list[float | None]] = {"strict": [], "lenient": []}
    for m in models:
        routing = summary["results"][m].get("routing") or {}
        acc["strict"].append((routing.get("accuracy_strict") or {}).get("rate"))
        acc["lenient"].append((routing.get("accuracy_lenient") or {}).get("rate"))
    plots.grouped_bars(axes[0], models, acc, percent=True)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Routing accuracy (%)")
    plots.set_title(axes[0], "Routing vs gold")
    role_names = sorted({r for m in models for r in summary["results"][m].get("roles") or {}})
    validity = {
        m: [
            ((summary["results"][m].get("roles") or {}).get(role) or {})
            .get("valid_first_attempt", {})
            .get("rate")
            for role in role_names
        ]
        for m in models
    }
    plots.grouped_bars(
        axes[1],
        [r.replace("_", " ") for r in role_names],
        validity,
        percent=True,
        value_labels=False,
    )
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Valid JSON on first attempt (%)")
    axes[1].tick_params(axis="x", labelrotation=20)
    plots.set_title(axes[1], "Schema validity per LLM role")
    lat = {
        "p50": [_s(summary["results"][m]["latency_turn_p50_ms"]) for m in models],
        "p95": [_s(summary["results"][m]["latency_turn_p95_ms"]) for m in models],
    }
    plots.grouped_bars(axes[2], models, lat)
    axes[2].set_ylabel("Seconds per turn")
    plots.set_title(axes[2], "Latency per turn")
    paths += plots.save_figure(fig, figures_dir, "model_selection")
    return paths


def _s(ms: float | None) -> float | None:
    return None if ms is None or ms != ms else ms / 1000.0


def _seconds(ms: float | None) -> str:
    value = _s(ms)
    return "—" if value is None else f"{value:.1f}"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def _pct(p: dict[str, Any] | None) -> str:
    if not p or p.get("rate") is None or p["rate"] != p["rate"]:
        return "—"
    return f"{100 * p['rate']:.1f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare candidate application models on C5 (dev split)"
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=MODEL_SELECTION_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--fresh", action="store_true", help="discard previous per-model runs")
    parser.add_argument("--skip-run", action="store_true", help="only summarise existing runs")
    parser.add_argument("--ollama-url", default=None, help="default: settings.OLLAMA_BASE_URL")
    args = parser.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    runs_root = args.out / "runs"
    exit_codes: dict[str, int] = {}
    if not args.skip_run:
        for model in models:
            exit_codes[model] = run_model(
                model, args.split, args.limit, runs_root, args.dataset, args.fresh
            )
    if args.ollama_url is None:
        from medbridge.config import get_settings

        args.ollama_url = get_settings().OLLAMA_BASE_URL
    info = ollama_models(args.ollama_url, models)
    results = {}
    for model in models:
        records = load_run_records(runs_root / safe_name(model), ["C5"]).get("C5", [])
        records = [r for r in records if args.split == "all" or r.get("split") == args.split]
        results[model] = summarize_model(
            model, records, info.get(model) if "error" not in info else None
        )
    summary = {
        "generated_at": utc_now(),
        "split": args.split,
        "limit": args.limit,
        "condition": "C5",
        "models": models,
        "exit_codes": exit_codes,
        "results": results,
    }
    for path in write_outputs(summary, args.out, args.figures_dir):
        print(f"[model_selection] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
