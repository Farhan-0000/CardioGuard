"""
CLI for the MedBridge-AQ routing benchmark.

Examples::

    # Harness self-test (routing simulator, hermetic, seconds):
    python -m tests.benchmark.benchmark_runner --mode mock --all

    # Live: production pipeline, one new eval-DB session per vignette, resumable:
    python -m tests.benchmark.benchmark_runner --mode live --all --resume
    python -m tests.benchmark.benchmark_runner --mode live --subset emergency_escalate
    python -m tests.benchmark.benchmark_runner --mode live --all --limit 12   # round-robin

Live mode uses ``--database-url``, else ``$EVAL_DATABASE_URL``, else
``EVAL_DATABASE_URL`` from ``.env``; it refuses the development database.
Every evaluated vignette is appended to a JSONL checkpoint in
``reports/benchmark_runs/`` so ``--resume`` continues an interrupted run (and re-renders
the report of a finished run without any LLM call).

Exit codes: 0 = acceptance targets met, 1 = targets not met, 2 = configuration error.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from tests.benchmark.evaluator import (
    SUBSET_FILES,
    BenchmarkEvaluator,
    load_checkpoint,
    load_vignettes,
    rotate_checkpoint,
    select_round_robin,
)
from tests.benchmark.report_generator import ReportGenerator
from tests.benchmark.schemas import VignetteEvaluationResult

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "reports" / "benchmark_runs"
# One report path for both modes; the report states its mode prominently (mock reports carry
# the "routing simulator - not evidence of model behaviour" banner). A finished live run can
# be re-rendered without LLM calls: ``--mode live --all --resume`` reads every row from the
# checkpoint.
DEFAULT_REPORT_PATH = REPO_ROOT / "reports" / "medbridge_aq_benchmark_report.md"


def checkpoint_path(mode: str, subset: str, run_name: str | None = None) -> Path:
    return RUNS_DIR / f"{run_name or f'aq_{mode}_{subset}'}.jsonl"


def _progress(index: int, total: int, result: VignetteEvaluationResult) -> None:
    mark = "ok " if result.action_match else "MISS"
    extra = f" error={result.error}" if result.error else ""
    print(
        f"  [{index:>3}/{total}] {mark} {result.vignette_id:<11} "
        f"expected={result.expected_action.value:<10} "
        f"actual={result.actual_action.value:<10} {result.latency_ms:>9.0f} ms{extra}",
        flush=True,
    )


async def run_benchmark_cli(
    subset: str = "sample",
    mode: str = "mock",
    output_path: Path | None = None,
    quiet: bool = False,
    *,
    limit: int | None = None,
    resume: bool = False,
    database_url: str | None = None,
    run_name: str | None = None,
    use_checkpoint: bool | None = None,
    executor: Any | None = None,
) -> int:
    """Run the benchmark; returns the process exit code."""
    output = output_path or DEFAULT_REPORT_PATH
    try:
        vignettes = select_round_robin(load_vignettes(None if subset == "all" else subset), limit)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    if not vignettes:
        print(f"[ERROR] No vignettes found for subset '{subset}'", file=sys.stderr)
        return 2

    metadata: dict[str, Any] = {"subset": subset, "limit": limit if limit is not None else "none"}
    if mode == "live" and executor is None:
        from tests.benchmark.live import (
            LiveConfigError,
            configure_eval_environment,
            resolve_eval_database_url,
            run_metadata,
        )

        try:
            url = resolve_eval_database_url(database_url)
        except LiveConfigError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2
        configure_eval_environment(url)
        metadata.update(run_metadata(url))

    checkpointing = (mode == "live") if use_checkpoint is None else use_checkpoint
    ckpt = checkpoint_path(mode, subset, run_name) if checkpointing else None
    completed: dict[str, VignetteEvaluationResult] = {}
    if ckpt is not None:
        if resume:
            completed = load_checkpoint(ckpt)
        else:
            moved = rotate_checkpoint(ckpt)
            if moved and not quiet:
                print(f"Previous checkpoint moved to {moved}")
        metadata["checkpoint"] = (
            str(ckpt.relative_to(REPO_ROOT)) if ckpt.is_relative_to(REPO_ROOT) else str(ckpt)
        )

    if not quiet:
        print("=" * 78)
        print(
            f"MedBridge-AQ routing benchmark — mode={mode} subset={subset} "
            f"vignettes={len(vignettes)} resume_hits={sum(v.id in completed for v in vignettes)}"
        )
        if mode == "mock":
            print("MOCK MODE: routing simulator — not evidence of model behaviour.")
        print("=" * 78)

    evaluator = BenchmarkEvaluator(mode=mode, executor=executor)
    try:
        results, metrics = await evaluator.run_benchmark(
            vignettes,
            checkpoint=ckpt,
            completed=completed,
            progress=None if quiet else _progress,
        )
    finally:
        await evaluator.close()

    report = ReportGenerator.generate_markdown_report(
        metrics=metrics, results=results, eval_mode=mode, metadata=metadata
    )
    ReportGenerator.save_report(report, output)
    json_path = output.with_suffix(".json")
    ReportGenerator.save_json(ReportGenerator.build_json(metrics, results, metadata), json_path)

    if not quiet:
        print("-" * 78)
        print(
            f"Routing accuracy:      {metrics.overall_routing_accuracy}% "
            f"({metrics.total_correct_routings}/{metrics.total_vignettes})"
        )
        print(
            f"Citation precision:    {metrics.overall_citation_precision} "
            f"(n={metrics.citation_evaluated})"
        )
        print(
            f"Emergency recall:      {metrics.emergency_detection_recall}% "
            f"({metrics.emergency_detected_count}/{metrics.emergency_count})"
        )
        print(f"Out-of-domain ANSWER:  {metrics.hallucination_rate}%")
        print(f"Latency p50/p95:       {metrics.latency_p50_ms} / {metrics.latency_p95_ms} ms")
        print(f"Errors:                {metrics.errors}")
        print(f"Report: {output}  (+ {json_path.name})")
    return 0 if metrics.passed_acceptance_criteria else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MedBridge-AQ routing benchmark")
    parser.add_argument("--all", action="store_true", help="all vignettes of all 6 subsets")
    parser.add_argument(
        "--subset",
        default="sample",
        help="sample | all | " + " | ".join(SUBSET_FILES),
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="mock = routing simulator (hermetic); live = production pipeline",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N vignettes (round-robin across subsets)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="skip vignettes already in the JSONL checkpoint"
    )
    parser.add_argument(
        "--database-url", default=None, help="evaluation database (default: EVAL_DATABASE_URL)"
    )
    parser.add_argument("--run-name", default=None, help="checkpoint file stem")
    parser.add_argument(
        "--output", type=Path, default=None, help="Markdown report path (JSON written next to it)"
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    subset = "all" if args.all else args.subset
    return asyncio.run(
        run_benchmark_cli(
            subset=subset,
            mode=args.mode,
            output_path=args.output,
            quiet=args.quiet,
            limit=args.limit,
            resume=args.resume,
            database_url=args.database_url,
            run_name=args.run_name,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
