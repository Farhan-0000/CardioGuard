"""
Operational metrics route (Module M-04) — ``GET /api/metrics`` (admin only).

Aggregates the most recent ``audit_logs`` rows (plaintext columns only:
actions, latency and the PHI-free ``trace``) and the ``feedback`` table:

* final-action, Gate 1 and Gate 2 decision counts (+ decision sources);
* end-to-end latency p50/p95 overall and per action, per-stage p50/p95 from
  ``trace.timings`` (only stages that ran: ``skipped`` and ``discarded``
  timings are excluded);
* spec-fallback counts (``evidence_gate_partial``: V1 turns that lost some,
  but not all, of their votes), loop-breaker / fast-path / small-talk counts,
  citation downgrades, speculative-retrieval discard rate and the latency it
  saved;
* Gate 3 (grounding verifier) counters: verified turns, failures, turns where
  the NLI model was unavailable, action changes, claims checked, unsupported
  claims and removed sentences;
* feedback count and 👍/👎 totals.

The computation is a pure function (:func:`compute_metrics`) so it is unit
tested without a database.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.db import models as db_models
from medbridge.db.connection import get_session
from medbridge.privacy.deps import require_admin

logger = structlog.get_logger(__name__)

metrics_router = APIRouter(tags=["metrics"])

# Stage timings that measure real work. "skipped" (stage not run, 0 ms) and
# "discarded" (speculative retrieval thrown away after a SOFT-ASK) are excluded
# from the per-stage percentiles.
_TIMED_STATUSES = frozenset({"ok", "failed"})


class LatencyStats(BaseModel):
    """Latency distribution in milliseconds."""

    count: int = 0
    p50_ms: float | None = None
    p95_ms: float | None = None
    mean_ms: float | None = None
    max_ms: float | None = None


class FeedbackStats(BaseModel):
    """👍/👎 feedback totals."""

    count: int = 0
    thumbs_up: int = 0
    thumbs_down: int = 0
    positive_rate: float | None = None


class MetricsResponse(BaseModel):
    """Response body for ``GET /api/metrics``."""

    generated_at: datetime
    turns: int = 0
    window: int = 0
    final_actions: dict[str, int] = Field(default_factory=dict)
    gate_1: dict[str, int] = Field(default_factory=dict)
    gate_2: dict[str, int] = Field(default_factory=dict)
    gate_sources: dict[str, dict[str, int]] = Field(default_factory=dict)
    latency: LatencyStats = Field(default_factory=LatencyStats)
    latency_by_action: dict[str, LatencyStats] = Field(default_factory=dict)
    stage_latency: dict[str, LatencyStats] = Field(default_factory=dict)
    fallbacks: dict[str, int] = Field(default_factory=dict)
    routing: dict[str, int] = Field(default_factory=dict)
    citations: dict[str, int] = Field(default_factory=dict)
    speculative_retrieval: dict[str, float | int | None] = Field(default_factory=dict)
    grounding_verifier: dict[str, int] = Field(default_factory=dict)
    redaction: dict[str, int] = Field(default_factory=dict)
    feedback: FeedbackStats = Field(default_factory=FeedbackStats)


# ----------------------------------------------------------------------
# Pure aggregation
# ----------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (``q`` in 0..100); ``None`` for no data."""
    if not values:
        return None
    data = sorted(values)
    rank = (len(data) - 1) * q / 100.0
    low = math.floor(rank)
    high = min(low + 1, len(data) - 1)
    return data[low] + (data[high] - data[low]) * (rank - low)


def latency_stats(values: Sequence[float]) -> LatencyStats:
    """Summarise a list of millisecond durations."""
    if not values:
        return LatencyStats()
    p50, p95 = percentile(values, 50), percentile(values, 95)
    return LatencyStats(
        count=len(values),
        p50_ms=round(p50, 1) if p50 is not None else None,
        p95_ms=round(p95, 1) if p95 is not None else None,
        mean_ms=round(sum(values) / len(values), 1),
        max_ms=round(max(values), 1),
    )


def _value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _source(trace: dict[str, Any], gate: str) -> str | None:
    decision = trace.get(gate)
    return str(decision.get("source")) if isinstance(decision, dict) else None


def _failed_votes(trace: dict[str, Any]) -> int:
    """V1 evidence-gate votes that failed (``gate2.features.failed_votes``)."""
    gate2 = trace.get("gate2")
    features = gate2.get("features") if isinstance(gate2, dict) else None
    return _int(features.get("failed_votes")) if isinstance(features, dict) else 0


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)


GATE3_COUNTERS: tuple[str, ...] = (
    "verified_turns", "failures", "unavailable", "action_changes",
    "claims", "unsupported_claims", "removed_sentences",
)


def _count_gate3(counter: Counter[str], verification: dict[str, Any]) -> None:
    """Gate 3 (grounding verifier) counters from the public ``trace.gate3`` record."""
    counter["verified_turns"] += 1
    status = verification.get("status")
    if status == "failed":
        counter["failures"] += 1
    if status == "unavailable" or verification.get("available") is False:
        counter["unavailable"] += 1
    if verification.get("action_change"):
        counter["action_changes"] += 1
    counter["claims"] += _int(verification.get("n_claims"))
    counter["unsupported_claims"] += _int(verification.get("n_unsupported"))
    counter["removed_sentences"] += _int(verification.get("removed_sentences_count"))


def compute_metrics(
    rows: Iterable[tuple[Any, Any, Any, int | None, dict[str, Any] | None]],
    feedback_ratings: Iterable[int],
    window: int,
) -> MetricsResponse:
    """Aggregate audit rows ``(final_action, gate_1, gate_2, latency_ms, trace)`` and ratings."""
    finals: Counter[str] = Counter()
    gate1: Counter[str] = Counter()
    gate2: Counter[str] = Counter()
    sources: dict[str, Counter[str]] = {"gate1": Counter(), "gate2": Counter()}
    latencies: list[float] = []
    by_action: dict[str, list[float]] = defaultdict(list)
    by_stage: dict[str, list[float]] = defaultdict(list)
    fallbacks: Counter[str] = Counter()
    routing: Counter[str] = Counter()
    citations: Counter[str] = Counter()
    gate3: Counter[str] = Counter()
    redaction: Counter[str] = Counter()
    launched = used = discarded = 0
    saved: list[float] = []
    turns = 0

    for final_action, gate_1, gate_2, latency_ms, raw_trace in rows:
        turns += 1
        final = _value(final_action) or "UNKNOWN"
        finals[final] += 1
        if (g1 := _value(gate_1)) is not None:
            gate1[g1] += 1
        if (g2 := _value(gate_2)) is not None:
            gate2[g2] += 1
        if latency_ms is not None:
            latencies.append(float(latency_ms))
            by_action[final].append(float(latency_ms))

        trace = raw_trace if isinstance(raw_trace, dict) else {}
        for gate in ("gate1", "gate2"):
            if (src := _source(trace, gate)) is not None:
                sources[gate][src] += 1
        if _source(trace, "gate1") == "fallback":
            fallbacks["context_gate"] += 1
        if _source(trace, "gate2") == "fallback":
            fallbacks["evidence_gate"] += 1
        elif _source(trace, "gate2") == "calibrated" and _failed_votes(trace) > 0:
            fallbacks["evidence_gate_partial"] += 1
        routing["loop_breaker"] += _source(trace, "gate2") == "loop_breaker"
        routing["emergency_fastpath"] += _source(trace, "gate2") == "fastpath"
        routing["small_talk"] += _source(trace, "gate2") == "rule"

        for timing in trace.get("timings", []) or []:
            if not isinstance(timing, dict):
                continue
            stage, status = str(timing.get("stage")), timing.get("status")
            if status in _TIMED_STATUSES and isinstance(timing.get("ms"), (int, float)):
                by_stage[stage].append(float(timing["ms"]))
            if status == "failed" and stage in ("extraction", "generation", "retrieval"):
                fallbacks[stage] += 1

        validation = trace.get("citation_validation")
        if isinstance(validation, dict):
            citations["validated_turns"] += 1
            citations["downgrades"] += bool(validation.get("downgraded"))
            citations["stripped_markers"] += int(validation.get("stripped") or 0)

        speculative = trace.get("speculative_retrieval")
        if isinstance(speculative, dict) and speculative.get("launched"):
            launched += 1
            if speculative.get("discarded"):
                discarded += 1
            if speculative.get("used"):
                used += 1
                saved.append(float(speculative.get("saved_ms") or 0.0))

        verification = trace.get("gate3")
        if isinstance(verification, dict):
            _count_gate3(gate3, verification)

        redactions = int(trace.get("redactions") or 0)
        redaction["redacted_items"] += redactions
        redaction["turns_with_redaction"] += redactions > 0

    ratings = list(feedback_ratings)
    up = sum(1 for r in ratings if r > 0)
    down = sum(1 for r in ratings if r < 0)
    return MetricsResponse(
        generated_at=datetime.now(UTC),
        turns=turns,
        window=window,
        final_actions=dict(finals),
        gate_1=dict(gate1),
        gate_2=dict(gate2),
        gate_sources={gate: dict(counter) for gate, counter in sources.items()},
        latency=latency_stats(latencies),
        latency_by_action={a: latency_stats(v) for a, v in sorted(by_action.items())},
        stage_latency={s: latency_stats(v) for s, v in sorted(by_stage.items())},
        fallbacks=dict(fallbacks),
        routing=dict(routing),
        citations=dict(citations),
        speculative_retrieval={
            "launched": launched,
            "used": used,
            "discarded": discarded,
            "discard_rate": round(discarded / launched, 3) if launched else None,
            "mean_saved_ms": round(sum(saved) / len(saved), 1) if saved else None,
        },
        grounding_verifier={name: gate3[name] for name in GATE3_COUNTERS},
        redaction=dict(redaction),
        feedback=FeedbackStats(
            count=len(ratings),
            thumbs_up=up,
            thumbs_down=down,
            positive_rate=round(up / (up + down), 3) if up + down else None,
        ),
    )


# ----------------------------------------------------------------------
# Route
# ----------------------------------------------------------------------


async def _feedback_ratings(db: AsyncSession) -> list[int]:
    feedback_model = getattr(db_models, "Feedback", None)
    if feedback_model is None:
        return []
    result = await db.execute(select(feedback_model.rating))
    return [int(r) for r in result.scalars().all() if r is not None]


@metrics_router.get(
    "/metrics",
    response_model=MetricsResponse,
    dependencies=[Depends(require_admin)],
    summary="Latency percentiles, gate decision counts and feedback statistics (admin)",
)
async def get_metrics(
    limit: int = Query(default=5000, ge=1, le=100_000, description="Most recent turns to include"),
    db: AsyncSession = Depends(get_session),
) -> MetricsResponse:
    """Aggregate the last ``limit`` audited turns and all feedback."""
    audit = db_models.AuditLog
    trace_column = getattr(audit, "trace", None)
    columns: list[Any] = [audit.final_action, audit.gate_1_action, audit.gate_2_action]
    columns.append(audit.latency_ms)
    if trace_column is not None:
        columns.append(trace_column)
    stmt = select(*columns).order_by(audit.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    rows = [
        (row[0], row[1], row[2], row[3], row[4] if trace_column is not None else None)
        for row in result.all()
    ]
    return compute_metrics(rows, await _feedback_ratings(db), window=limit)
