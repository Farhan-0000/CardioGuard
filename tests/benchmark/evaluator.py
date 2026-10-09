"""
Evaluation engine for the MedBridge-AQ routing benchmark.

Two modes:

* ``mock`` — a deterministic *routing simulator* that applies the documented
  pipeline rules (real emergency classifier, loop-breaker at 2, then the
  vignette's subset as a stand-in for the LLM gates). It exercises the harness,
  the metrics and the citation matcher (pseudo-citations are built from the
  corpus manifest). It is **not evidence of model behaviour**.
* ``live`` — every vignette runs through the default production pipeline in a
  NEW session of the evaluation database, with its snapshot seeded as typed
  events and its soft-ask counter set first (see :mod:`tests.benchmark.live`).

Results can be checkpointed one JSONL row per vignette so long live runs can
be resumed.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from medbridge.api.schemas.enums import ActionEnum
from medbridge.core.emergency_classifier import classify as classify_emergency
from tests.benchmark.citations import citation_precision, corpus_documents_for
from tests.benchmark.schemas import (
    BenchmarkMetrics,
    CitationRecord,
    ClinicalVignette,
    ExplanationSummary,
    SubsetMetrics,
    VignetteEvaluationResult,
)

VIGNETTES_DIR = Path(__file__).parent / "vignettes"

SUBSET_FILES: dict[str, str] = {
    "emergency_escalate": "emergency_escalate.json",
    "complete_context_answer": "complete_context_answer.json",
    "incomplete_context_soft_ask": "incomplete_context_soft_ask.json",
    "lifestyle_and_generalize": "lifestyle_and_generalize.json",
    "out_of_domain_abstain": "out_of_domain_abstain.json",
    "complex_comorbid_contraindications": "complex_comorbid_contraindications.json",
}

ACCEPTANCE_ROUTING_ACCURACY = 90.0
ACCEPTANCE_CITATION_PRECISION = 90.0
ACCEPTANCE_EMERGENCY_RECALL = 100.0
ACCEPTANCE_OOD_ANSWER_RATE = 0.0
LOOP_BREAKER_THRESHOLD = 2

MOCK_TEMPLATES: dict[ActionEnum, str] = {
    ActionEnum.ESCALATE: "[simulated] emergency template",
    ActionEnum.GENERALIZE: "[simulated] general guidance",
    ActionEnum.ABSTAIN: "[simulated] abstain template",
    ActionEnum.SOFT_ASK: "[simulated] clarifying question",
    ActionEnum.ANSWER: "[simulated] grounded answer",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_vignettes(
    subset: str | None = None,
    directory: Path | None = None,
) -> list[ClinicalVignette]:
    """Load vignettes.

    ``subset`` is a subset name, ``"sample"`` (first 2 per subset) or ``None``/``"all"``.
    """
    base_dir = directory or VIGNETTES_DIR
    per_file = 2 if subset == "sample" else None
    if subset and subset not in ("sample", "all") and subset not in SUBSET_FILES:
        raise ValueError(f"Unknown subset '{subset}' (choose from {sorted(SUBSET_FILES)})")
    targets = {subset: SUBSET_FILES[subset]} if subset in SUBSET_FILES else SUBSET_FILES
    vignettes: list[ClinicalVignette] = []
    for filename in targets.values():
        path = base_dir / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data[:per_file] if per_file else data:
            vignettes.append(ClinicalVignette(**item))
    return vignettes


def select_round_robin(
    vignettes: list[ClinicalVignette], limit: int | None
) -> list[ClinicalVignette]:
    """First ``limit`` vignettes taken round-robin across subsets (deterministic)."""
    if limit is None or limit >= len(vignettes):
        return list(vignettes)
    buckets: dict[str, list[ClinicalVignette]] = {}
    for v in vignettes:
        buckets.setdefault(v.subset, []).append(v)
    order = [s for s in SUBSET_FILES if s in buckets] + [
        s for s in buckets if s not in SUBSET_FILES
    ]
    selected: list[ClinicalVignette] = []
    index = 0
    while len(selected) < limit:
        progressed = False
        for name in order:
            if index < len(buckets[name]) and len(selected) < limit:
                selected.append(buckets[name][index])
                progressed = True
        if not progressed:
            break
        index += 1
    return selected


# ---------------------------------------------------------------------------
# Checkpoints (one JSONL row per vignette)
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path) -> dict[str, VignetteEvaluationResult]:
    """Completed results keyed by vignette id (later rows win; unreadable rows skipped)."""
    done: dict[str, VignetteEvaluationResult] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = VignetteEvaluationResult.model_validate_json(line)
        except ValueError:
            continue
        done[row.vignette_id] = row
    return done


def append_checkpoint(path: Path, result: VignetteEvaluationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(result.model_dump_json() + "\n")


def rotate_checkpoint(path: Path) -> Path | None:
    """Move an existing checkpoint aside (fresh run without ``--resume``)."""
    if not path.exists():
        return None
    target = path.with_name(f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak{path.suffix}")
    path.rename(target)
    return target


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

LiveRunner = Callable[[ClinicalVignette], Awaitable[Any]]


class BenchmarkEvaluator:
    """Evaluates vignettes in ``mock`` (simulator) or ``live`` (production pipeline) mode."""

    def __init__(self, mode: str = "mock", executor: Any | None = None) -> None:
        if mode not in ("mock", "live"):
            raise ValueError("mode must be 'mock' or 'live'")
        self.mode = mode
        self._executor = executor

    # -- mock ---------------------------------------------------------------

    @staticmethod
    def _mock_citations(vignette: ClinicalVignette) -> list[CitationRecord]:
        labels = vignette.expected_citations or ["AHA/ACC 2025"]
        records: list[CitationRecord] = []
        for label in labels:
            docs = corpus_documents_for(label)
            if docs and docs[0].guideline_id not in {r.guideline_id for r in records}:
                records.append(docs[0])
        return records

    def simulate_pipeline_decision(
        self, vignette: ClinicalVignette
    ) -> tuple[ActionEnum, list[CitationRecord], str, float, ExplanationSummary]:
        """Routing simulator (mock mode).

        1. Real deterministic emergency classifier on the query (fast-path).
        2. Loop-breaker: ``soft_ask_count >= 2`` forces GENERALIZE.
        3. Otherwise the vignette's subset stands in for the LLM gates.
        """
        start = time.perf_counter()
        summary = ExplanationSummary()
        citations: list[CitationRecord] = []
        if classify_emergency(vignette.query) == ActionEnum.ESCALATE:
            action = ActionEnum.ESCALATE
            summary.gate2_action, summary.gate2_source = "ESCALATE", "fastpath"
        elif vignette.soft_ask_count >= LOOP_BREAKER_THRESHOLD:
            action = ActionEnum.GENERALIZE
            summary.gate1_action, summary.gate1_source = "SOFT-ASK", "simulated"
            summary.gate2_action, summary.gate2_source = "GENERALIZE", "loop_breaker"
        else:
            action = {
                "emergency_escalate": ActionEnum.ESCALATE,
                "out_of_domain_abstain": ActionEnum.ABSTAIN,
                "incomplete_context_soft_ask": ActionEnum.SOFT_ASK,
                "lifestyle_and_generalize": ActionEnum.GENERALIZE,
            }.get(vignette.subset, ActionEnum.ANSWER)
            if action == ActionEnum.SOFT_ASK:
                summary.gate1_action, summary.gate1_source = "SOFT-ASK", "simulated"
            else:
                summary.gate1_action, summary.gate1_source = "PROCEED", "simulated"
                summary.gate2_action, summary.gate2_source = action.value, "simulated"
            if action == ActionEnum.ANSWER:
                citations = self._mock_citations(vignette)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return action, citations, MOCK_TEMPLATES[action], elapsed_ms, summary

    # -- one vignette ---------------------------------------------------------

    async def _run_live(
        self, vignette: ClinicalVignette
    ) -> tuple[ActionEnum, list[CitationRecord], str, float, ExplanationSummary, str]:
        if self._executor is None:
            from tests.benchmark.live import LivePipelineExecutor

            self._executor = LivePipelineExecutor()
        turn = await self._executor.run(
            vignette.query, vignette.patient_snapshot, vignette.soft_ask_count
        )
        response = turn.response
        citations = [CitationRecord.from_response(c) for c in response.citations]
        summary = ExplanationSummary.from_explanation(response.explanation)
        return (
            ActionEnum(response.action),
            citations,
            response.response_text,
            turn.latency_ms,
            summary,
            str(turn.session_id),
        )

    async def evaluate_vignette(self, vignette: ClinicalVignette) -> VignetteEvaluationResult:
        """Run one vignette and score it (never raises; errors are recorded)."""
        session_id: str | None = None
        error: str | None = None
        try:
            if self.mode == "mock":
                action, citations, text, latency, summary = self.simulate_pipeline_decision(
                    vignette
                )
            else:
                action, citations, text, latency, summary, session_id = await self._run_live(
                    vignette
                )
        except Exception as exc:  # recorded as an error row, scored as a miss
            action, citations, text, latency = ActionEnum.GENERALIZE, [], "", 0.0
            summary = ExplanationSummary()
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
        return self.score(vignette, action, citations, text, latency, summary, session_id, error)

    def score(
        self,
        vignette: ClinicalVignette,
        action: ActionEnum,
        citations: list[CitationRecord],
        response_text: str,
        latency_ms: float,
        summary: ExplanationSummary | None = None,
        session_id: str | None = None,
        error: str | None = None,
    ) -> VignetteEvaluationResult:
        """Build the result row (precision only where citations and labels exist)."""
        is_emergency = vignette.expected_action == ActionEnum.ESCALATE
        is_ood = vignette.subset == "out_of_domain_abstain"
        precision, matched = citation_precision(vignette.expected_citations, citations)
        return VignetteEvaluationResult(
            vignette_id=vignette.id,
            subset=vignette.subset,
            query=vignette.query,
            mode=self.mode,
            expected_action=vignette.expected_action,
            actual_action=action,
            action_match=action == vignette.expected_action and error is None,
            soft_ask_count=vignette.soft_ask_count,
            expected_citations=list(vignette.expected_citations),
            actual_citations=citations,
            matched_citations=matched,
            citation_precision=precision,
            is_emergency=is_emergency,
            emergency_detected=is_emergency and action == ActionEnum.ESCALATE and error is None,
            is_out_of_domain=is_ood,
            hallucinated=is_ood and action == ActionEnum.ANSWER,
            response_text=response_text,
            latency_ms=round(latency_ms, 2),
            explanation=summary,
            session_id=session_id,
            error=error,
        )

    async def run_benchmark(
        self,
        vignettes: list[ClinicalVignette],
        *,
        checkpoint: Path | None = None,
        completed: dict[str, VignetteEvaluationResult] | None = None,
        progress: Callable[[int, int, VignetteEvaluationResult], None] | None = None,
    ) -> tuple[list[VignetteEvaluationResult], BenchmarkMetrics]:
        """Evaluate ``vignettes`` (skipping ``completed`` ones) and compute the metrics."""
        done = dict(completed or {})
        results: list[VignetteEvaluationResult] = []
        for index, vignette in enumerate(vignettes, 1):
            if vignette.id in done:
                results.append(done[vignette.id])
                continue
            result = await self.evaluate_vignette(vignette)
            if checkpoint is not None:
                append_checkpoint(checkpoint, result)
            results.append(result)
            if progress is not None:
                progress(index, len(vignettes), result)
        return results, self.calculate_metrics(results)

    async def close(self) -> None:
        if self._executor is not None and hasattr(self._executor, "close"):
            await self._executor.close()

    # -- metrics ------------------------------------------------------------

    def calculate_metrics(self, results: list[VignetteEvaluationResult]) -> BenchmarkMetrics:
        return calculate_metrics(results, mode=self.mode)


def _pct(numerator: float, denominator: float) -> float:
    return round(numerator / denominator * 100.0, 2) if denominator else 0.0


def _percentile(values: Iterable[float], q: float) -> float:
    data = [v for v in values]
    return round(float(np.percentile(data, q)), 2) if data else 0.0


def calculate_metrics(
    results: list[VignetteEvaluationResult], mode: str = "mock"
) -> BenchmarkMetrics:
    """Aggregate, per-subset, confusion-matrix, safety and latency metrics."""
    actions = [a.value for a in ActionEnum]
    total = len(results)
    if total == 0:
        return BenchmarkMetrics(
            mode=mode,
            total_vignettes=0,
            total_correct_routings=0,
            overall_routing_accuracy=0.0,
            emergency_count=0,
            emergency_detected_count=0,
            emergency_detection_recall=0.0,
            out_of_domain_count=0,
            out_of_domain_abstain_count=0,
            hallucination_rate=0.0,
            confusion_matrix={e: {a: 0 for a in actions} for e in actions},
        )

    correct = sum(1 for r in results if r.action_match)
    scored = [r for r in results if r.citation_precision is not None]
    precision = (
        round(sum(r.citation_precision or 0.0 for r in scored) / len(scored) * 100.0, 2)
        if scored
        else None
    )
    expected_answer = [r for r in results if r.expected_action == ActionEnum.ANSWER]
    answer_with_citations = sum(1 for r in expected_answer if r.actual_citations)

    emergencies = [r for r in results if r.is_emergency]
    detected = sum(1 for r in emergencies if r.emergency_detected)
    fastpath = sum(
        1
        for r in emergencies
        if r.emergency_detected and r.explanation and r.explanation.gate2_source == "fastpath"
    )
    ood = [r for r in results if r.is_out_of_domain]
    ood_answer = sum(1 for r in ood if r.hallucinated)
    ood_abstain = sum(1 for r in ood if r.actual_action == ActionEnum.ABSTAIN)

    loop = [r for r in results if r.soft_ask_count >= LOOP_BREAKER_THRESHOLD]
    loop_generalize = sum(1 for r in loop if r.actual_action == ActionEnum.GENERALIZE)
    loop_forced = sum(
        1 for r in loop if r.explanation and r.explanation.gate2_source == "loop_breaker"
    )

    subsets: dict[str, SubsetMetrics] = {}
    for name in sorted({r.subset for r in results}):
        items = [r for r in results if r.subset == name]
        s_scored = [r for r in items if r.citation_precision is not None]
        subsets[name] = SubsetMetrics(
            subset_name=name,
            total_vignettes=len(items),
            correct_routings=sum(1 for r in items if r.action_match),
            routing_accuracy=_pct(sum(1 for r in items if r.action_match), len(items)),
            avg_citation_precision=(
                round(sum(r.citation_precision or 0.0 for r in s_scored) / len(s_scored) * 100, 2)
                if s_scored
                else None
            ),
            citation_evaluated=len(s_scored),
            avg_latency_ms=round(float(np.mean([r.latency_ms for r in items])), 2),
            errors=sum(1 for r in items if r.error),
        )

    confusion = {e: {a: 0 for a in actions} for e in actions}
    for r in results:
        confusion[r.expected_action.value][r.actual_action.value] += 1

    sources = Counter(r.decision_source or "unknown" for r in results)
    latencies = [r.latency_ms for r in results if not r.error]
    server = [
        r.explanation.server_total_ms
        for r in results
        if r.explanation and r.explanation.server_total_ms is not None
    ]

    accuracy = _pct(correct, total)
    recall = _pct(detected, len(emergencies)) if emergencies else 100.0
    ood_rate = _pct(ood_answer, len(ood))
    # A target is only judged when the run contains vignettes it applies to.
    acceptance = {"routing_accuracy": accuracy >= ACCEPTANCE_ROUTING_ACCURACY}
    if any(r.expected_action == ActionEnum.ANSWER and r.expected_citations for r in results):
        acceptance["citation_precision"] = (
            precision is not None and precision >= ACCEPTANCE_CITATION_PRECISION
        )
    if emergencies:
        acceptance["emergency_recall"] = recall >= ACCEPTANCE_EMERGENCY_RECALL
    if ood:
        acceptance["ood_answer_rate"] = ood_rate <= ACCEPTANCE_OOD_ANSWER_RATE
    return BenchmarkMetrics(
        mode=mode,
        total_vignettes=total,
        total_correct_routings=correct,
        overall_routing_accuracy=accuracy,
        overall_citation_precision=precision,
        citation_evaluated=len(scored),
        expected_answer_count=len(expected_answer),
        expected_answer_with_citations=answer_with_citations,
        citation_presence_rate=(
            _pct(answer_with_citations, len(expected_answer)) if expected_answer else None
        ),
        emergency_count=len(emergencies),
        emergency_detected_count=detected,
        emergency_detection_recall=recall,
        emergency_fastpath_count=fastpath,
        out_of_domain_count=len(ood),
        out_of_domain_abstain_count=ood_abstain,
        out_of_domain_answer_count=ood_answer,
        hallucination_rate=ood_rate,
        loop_breaker_count=len(loop),
        loop_breaker_generalize_count=loop_generalize,
        loop_breaker_forced_count=loop_forced,
        loop_breaker_rate=_pct(loop_generalize, len(loop)) if loop else None,
        errors=sum(1 for r in results if r.error),
        subsets=subsets,
        confusion_matrix=confusion,
        decision_sources=dict(sorted(sources.items())),
        mean_latency_ms=round(float(np.mean(latencies)), 2) if latencies else 0.0,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        latency_p99_ms=_percentile(latencies, 99),
        server_latency_p50_ms=_percentile(server, 50) if server else None,
        server_latency_p95_ms=_percentile(server, 95) if server else None,
        passed_acceptance_criteria=all(acceptance.values()),
        acceptance=acceptance,
    )
