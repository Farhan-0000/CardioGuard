"""
Markdown + JSON report for the MedBridge-AQ routing benchmark.

Every number and statement in the report is computed from the per-vignette
results / ``BenchmarkMetrics`` passed in; nothing is hard-coded. Mock-mode
reports carry a prominent "routing simulator — not evidence of model
behaviour" banner.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tests.benchmark.evaluator import (
    ACCEPTANCE_CITATION_PRECISION,
    ACCEPTANCE_EMERGENCY_RECALL,
    ACCEPTANCE_OOD_ANSWER_RATE,
    ACCEPTANCE_ROUTING_ACCURACY,
    LOOP_BREAKER_THRESHOLD,
)
from tests.benchmark.schemas import BenchmarkMetrics, VignetteEvaluationResult

ACTIONS = ["ANSWER", "SOFT-ASK", "GENERALIZE", "ABSTAIN", "ESCALATE"]
MOCK_BANNER = (
    "> **MOCK MODE — routing simulator — not evidence of model behaviour.** Actions come from "
    "the real emergency classifier, the loop-breaker rule and the vignette's own subset label; "
    "citations are pseudo-citations built from the corpus manifest. Use these numbers only to "
    "check the harness. Run `--mode live` for pipeline results."
)


def _fmt(value: float | None, suffix: str = "%") -> str:
    return "n/a" if value is None else f"{value}{suffix}"


def _status(ok: bool | None) -> str:
    if ok is None:
        return "n/a"
    return "PASS" if ok else "FAIL"


class ReportGenerator:
    """Formats benchmark metrics into Markdown (and JSON) reports."""

    @staticmethod
    def generate_markdown_report(
        metrics: BenchmarkMetrics,
        results: list[VignetteEvaluationResult] | None = None,
        eval_mode: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        results = results or []
        mode = eval_mode or metrics.mode
        is_mock = mode.startswith("mock")
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        meta = metadata or {}
        lines: list[str] = ["# MedBridge-AQ Routing Benchmark Report", ""]
        if is_mock:
            lines += [MOCK_BANNER, ""]
        lines += [
            f"- **Generated:** {now}",
            f"- **Mode:** `{mode}`"
            + (" (routing simulator)" if is_mock else " (production pipeline)"),
            f"- **Vignettes evaluated:** {metrics.total_vignettes}",
            f"- **Errors (turns that raised in the harness):** {metrics.errors}",
        ]
        for key, value in meta.items():
            lines.append(f"- **{key}:** `{value}`")
        lines += ["", "## 1. Summary against acceptance targets", ""]
        acc = metrics.acceptance
        verdict = (
            "all acceptance targets met"
            if metrics.passed_acceptance_criteria
            else "acceptance targets not met"
        )
        lines += [
            "| Metric | Target | Measured | Status | Basis |",
            "| :--- | :---: | :---: | :---: | :--- |",
            f"| Routing accuracy | >= {ACCEPTANCE_ROUTING_ACCURACY}% | "
            f"{metrics.overall_routing_accuracy}% | {_status(acc.get('routing_accuracy'))} | "
            f"{metrics.total_correct_routings}/{metrics.total_vignettes} expected actions "
            "matched |",
            f"| Citation precision | >= {ACCEPTANCE_CITATION_PRECISION}% | "
            f"{_fmt(metrics.overall_citation_precision)} | "
            f"{_status(acc.get('citation_precision'))} | "
            f"macro-average over {metrics.citation_evaluated} vignettes that returned >= 1 "
            "citation and list expected sources |",
            f"| Emergency recall | {ACCEPTANCE_EMERGENCY_RECALL}% | "
            f"{metrics.emergency_detection_recall}% | {_status(acc.get('emergency_recall'))} | "
            f"{metrics.emergency_detected_count}/{metrics.emergency_count} expected-ESCALATE "
            f"vignettes escalated ({metrics.emergency_fastpath_count} by the deterministic "
            "fast-path) |",
            f"| Out-of-domain ANSWER rate | {ACCEPTANCE_OOD_ANSWER_RATE}% | "
            f"{metrics.hallucination_rate}% | {_status(acc.get('ood_answer_rate'))} | "
            f"{metrics.out_of_domain_count - metrics.out_of_domain_abstain_count} of "
            f"{metrics.out_of_domain_count} out-of-domain vignettes not ABSTAINed; "
            "ANSWER on out-of-domain is a routing proxy, not a hallucination judgement |",
            "",
            f"Overall: **{verdict}** ({sum(1 for v in acc.values() if v)}/{len(acc)} targets "
            "met; targets without applicable vignettes are not judged).",
            "",
            "## 2. Results by subset",
            "",
            "| Subset | Vignettes | Correct | Routing accuracy | Citation precision (n) "
            "| Mean latency | Errors |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for s in metrics.subsets.values():
            lines.append(
                f"| `{s.subset_name}` | {s.total_vignettes} | {s.correct_routings} | "
                f"{s.routing_accuracy}% | {_fmt(s.avg_citation_precision)} "
                f"({s.citation_evaluated}) | {s.avg_latency_ms} ms | {s.errors} |"
            )

        lines += ["", "## 3. Safety invariants (computed from this run)", ""]
        lines.append(
            f"- **Emergency triage:** {metrics.emergency_detected_count} of "
            f"{metrics.emergency_count} expected-ESCALATE vignettes were escalated; "
            f"{metrics.emergency_fastpath_count} by the deterministic fast-path before any LLM "
            "call."
        )
        if metrics.loop_breaker_count:
            other = metrics.loop_breaker_generalize_count - metrics.loop_breaker_forced_count
            text = (
                f"- **Loop-breaker (soft_ask_count >= {LOOP_BREAKER_THRESHOLD}):** "
                f"{metrics.loop_breaker_count} vignettes start at or above the threshold; "
                f"{metrics.loop_breaker_generalize_count} ended in GENERALIZE "
                f"({_fmt(metrics.loop_breaker_rate)}); {metrics.loop_breaker_forced_count} were "
                "forced by the loop-breaker (Gate 1 returned SOFT-ASK at the threshold)"
            )
            if other > 0:
                text += f" and {other} reached GENERALIZE through Gate 1 PROCEED and Gate 2"
            lines.append(text + ".")
        else:
            lines.append("- **Loop-breaker:** no vignette in this run starts at the threshold.")
        lines.append(
            f"- **Out-of-domain:** {metrics.out_of_domain_abstain_count} of "
            f"{metrics.out_of_domain_count} out-of-domain vignettes were ABSTAINed; "
            f"{metrics.out_of_domain_answer_count} received an ANSWER."
        )
        if metrics.expected_answer_count:
            lines.append(
                f"- **Citations on expected-ANSWER vignettes:** "
                f"{metrics.expected_answer_with_citations}/{metrics.expected_answer_count} "
                "returned "
                f">= 1 citation ({_fmt(metrics.citation_presence_rate)})."
            )

        lines += [
            "",
            "## 4. Action confusion matrix",
            "",
            "Rows: expected action; columns: actual action.",
            "",
            "| Expected \\ Actual | " + " | ".join(ACTIONS) + " |",
            "| :--- | " + " | ".join(":---:" for _ in ACTIONS) + " |",
        ]
        for exp in ACTIONS:
            row = [str(metrics.confusion_matrix.get(exp, {}).get(act, 0)) for act in ACTIONS]
            lines.append(f"| **{exp}** | " + " | ".join(row) + " |")

        lines += [
            "",
            "## 5. Who decided the final route",
            "",
            "| Decision source | Vignettes |",
            "| :--- | :---: |",
        ]
        for source, count in metrics.decision_sources.items():
            lines.append(f"| `{source}` | {count} |")

        lines += [
            "",
            "## 6. Latency",
            "",
            "- Client wall-clock per turn (excluding harness errors): mean "
            f"{metrics.mean_latency_ms} ms, "
            f"p50 {metrics.latency_p50_ms} ms, p95 {metrics.latency_p95_ms} ms, "
            f"p99 {metrics.latency_p99_ms} ms.",
        ]
        if metrics.server_latency_p50_ms is not None:
            lines.append(
                f"- Server-side `explanation.total_ms`: p50 {metrics.server_latency_p50_ms} ms, "
                f"p95 {metrics.server_latency_p95_ms} ms."
            )

        misses = [r for r in results if not r.action_match]
        lines += ["", f"## 7. Misrouted or failed vignettes ({len(misses)})", ""]
        if misses:
            lines += [
                "| Vignette | Subset | Expected | Actual | Decided by | Note |",
                "| :--- | :--- | :---: | :---: | :--- | :--- |",
            ]
            for r in misses:
                note = r.error or ""
                if not note and r.explanation and r.explanation.downgraded_to_generalize:
                    note = "ANSWER downgraded: no valid citation"
                lines.append(
                    f"| {r.vignette_id} | {r.subset} | {r.expected_action.value} | "
                    f"{r.actual_action.value} | {r.decision_source or 'n/a'} | {note} |"
                )
        else:
            lines.append("None.")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def build_json(
        metrics: BenchmarkMetrics,
        results: list[VignetteEvaluationResult],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "mode": metrics.mode,
            "note": (
                "routing simulator - not evidence of model behaviour"
                if metrics.mode == "mock"
                else "production pipeline"
            ),
            "metadata": metadata or {},
            "metrics": metrics.model_dump(mode="json"),
            "results": [r.model_dump(mode="json") for r in results],
        }

    @staticmethod
    def save_report(report_text: str, output_path: Path) -> Path:
        """Write ``report_text`` (creating parent directories)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")
        return output_path

    @staticmethod
    def save_json(data: dict[str, Any], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path
