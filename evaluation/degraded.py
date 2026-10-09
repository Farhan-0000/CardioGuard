"""
Degraded-turn accounting (finalization review: ablation-study-validity-1/-2).

The orchestrator never raises: when a component fails it substitutes a spec
fallback (Gate 1 → PROCEED, Gate 2 → GENERALIZE, generator → pre-vetted
template, unhandled error → ``PIPELINE_ERROR_TEMPLATE``), keeps going after a
failed ``load_state`` / persistence, and V2 keeps the draft when the NLI model
is unavailable. Such a turn is a valid patient response but NOT a valid
measurement of the condition under test: a failure window that hits some
conditions and not others would shift every paired comparison.

``degradation_reasons(record)`` lists every sign of such a failure in a run
record (one item-turn). It is used

* by the runner, which stores ``degraded`` / ``degraded_reasons`` on every
  record, treats degraded attempts as incomplete for ``--resume`` and re-runs
  them with ``--retry-degraded``;
* by the analysis, which reports degraded counts per condition, excludes
  degraded turns from the headline metrics and repeats them in a sensitivity
  analysis that includes them (records written before this module existed are
  classified from their stored explanation).

Reasons are short stable strings, ``kind`` or ``kind:detail``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from typing import Any

#: Bypass every cached read (used when the failing call cannot be identified).
ALL_CALLS = "*"

#: Pipeline stage → LLM call made in that stage (for targeted cache bypass on re-runs).
STAGE_CALLS: dict[str, str] = {
    "extraction": "context_extractor",
    "context_gate": "context_gate",
    "evidence_gate": "evidence_gate",
    "generation": "response_generator",
}
#: Reasons whose cause is an LLM call (→ that call's cached response is bypassed on re-run).
_REASON_CALLS: dict[str, str] = {
    "gate1_fallback": "context_gate",
    "gate2_votes_failed": "evidence_gate",
    "generator_fallback_template": "response_generator",
}
#: Reasons caused by something other than an LLM response (DB, Qdrant, NLI model, ...):
#: re-running the item is enough, no cached LLM response needs to be bypassed.
_INFRASTRUCTURE_PREFIXES: tuple[str, ...] = (
    "error",
    "stage_failed:load_state",
    "stage_failed:persistence",
    "stage_failed:retrieval",
    "stage_failed:rerank",
    "stage_failed:verification",
    "retrieval_failed",
    "retrieval_empty",
    "gate3_",
    "evidence_fetch_failed",
    "evidence_text_missing",
    "earlier_turn_degraded",
)


@cache
def _templates() -> tuple[str, str]:
    """(PIPELINE_ERROR_TEMPLATE, GENERALIZE_FALLBACK_TEMPLATE); empty when unavailable."""
    try:
        from medbridge.core.templates import (
            GENERALIZE_FALLBACK_TEMPLATE,
            PIPELINE_ERROR_TEMPLATE,
        )
    except Exception:  # analysis must work without the application package configured
        return "", ""
    return PIPELINE_ERROR_TEMPLATE.strip(), GENERALIZE_FALLBACK_TEMPLATE.strip()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _add(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def degradation_reasons(record: Mapping[str, Any]) -> list[str]:
    """Every sign that this turn ran on a fallback / failure path (empty = clean turn).

    Covered: an infrastructure ``error`` of the runner; evidence that could not be
    fetched for the judges; the pipeline-error and generator-fallback templates;
    Gate 1 / Gate 2 decisions taken by the spec fallback (or with failed votes);
    any stage timing with status ``failed`` (extraction, load_state, context_gate,
    retrieval, rerank, evidence_gate, generation, verification, persistence, ...);
    ``explanation.retrieval_status == "failed"`` (``"empty"`` — the search ran and
    matched nothing — is a valid outcome); for records without that field, a
    retrieval stage that reported success but returned no evidence at all (the
    retriever used to swallow outages); Gate 3 unavailable or failed; and any LLM
    call that failed permanently.
    """
    reasons: list[str] = []
    if record.get("error"):
        _add(reasons, "error")
        return reasons
    if record.get("action") is None:
        _add(reasons, "no_action")
    if record.get("evidence_error"):
        _add(reasons, "evidence_fetch_failed")
    if any(
        isinstance(e, dict) and e.get("text_source") == "excerpt"
        for e in record.get("evidence_texts") or []
    ):
        _add(reasons, "evidence_text_missing")

    pipeline_error, generator_fallback = _templates()
    text = str(record.get("response_text") or "").strip()
    if pipeline_error and text == pipeline_error:
        _add(reasons, "pipeline_error")
    if generator_fallback and generator_fallback in text:
        _add(reasons, "generator_fallback_template")

    explanation = _dict(record.get("explanation"))
    for name, reason in (("gate1", "gate1_fallback"), ("gate2", "gate2_fallback")):
        gate = _dict(explanation.get(name))
        if gate.get("source") == "fallback":
            _add(reasons, reason)
    gate2_features = _dict(_dict(explanation.get("gate2")).get("features"))
    if gate2_features.get("failed_votes"):
        _add(reasons, "gate2_votes_failed")

    timings = [t for t in explanation.get("timings") or [] if isinstance(t, dict)]
    for timing in timings:
        if timing.get("status") == "failed":
            _add(reasons, f"stage_failed:{timing.get('stage')}")
    retrieval_status = explanation.get("retrieval_status")
    if retrieval_status == "failed":
        _add(reasons, "retrieval_failed")
    elif retrieval_status is None:
        # Records written before the orchestrator reported retrieval_status: the
        # retriever swallowed outages and returned [], which dense k-NN never does
        # on a reachable, non-empty collection.
        ran = any(t.get("stage") == "retrieval" and t.get("status") == "ok" for t in timings)
        if ran and not explanation.get("retrieval"):
            _add(reasons, "retrieval_empty")

    gate3 = _dict(explanation.get("gate3"))
    if gate3.get("status") in ("unavailable", "failed"):
        _add(reasons, f"gate3_{gate3['status']}")
    elif gate3 and gate3.get("available") is False:
        _add(reasons, "gate3_unavailable")

    calls = record.get("llm_calls") or explanation.get("llm_calls") or []
    for call in calls:
        if isinstance(call, dict) and call.get("success") is False:
            _add(reasons, f"llm_call_failed:{call.get('call_name')}")
    return reasons


def annotate_attempt(records: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """Reasons for every turn of ONE attempt (records sorted by ``turn_index``).

    A later turn of a multi-turn item inherits ``earlier_turn_degraded`` when an
    earlier turn was degraded: its session state / history then differs from the
    protocol (e.g. a failed persistence leaves turn 1 with an empty snapshot and
    ``soft_ask_count = 0``, a Gate 1 fallback skips the intended SOFT-ASK).
    """
    out: list[list[str]] = []
    earlier = False
    for record in records:
        reasons = degradation_reasons(record)
        for stored in record.get("degraded_reasons") or []:
            _add(reasons, str(stored))
        if earlier:
            _add(reasons, "earlier_turn_degraded")
        out.append(reasons)
        earlier = earlier or bool(reasons)
    return out


def record_degraded(record: Mapping[str, Any]) -> bool:
    """True when a stored record is degraded (stored flag, stored reasons or recomputed)."""
    return bool(
        record.get("degraded") or record.get("degraded_reasons") or degradation_reasons(record)
    )


def bypass_calls(reasons: Iterable[str]) -> frozenset[str]:
    """LLM call names whose cached responses must not be replayed when the turn is re-run.

    Calls that failed permanently, calls of failed LLM stages and the calls behind
    a fallback decision are bypassed; an unexplained failure inside the pipeline
    (pipeline-error template, missing action) bypasses every call (``ALL_CALLS``).
    Infrastructure reasons (runner/DB error, Qdrant, NLI model) need no bypass.
    """
    names: set[str] = set()
    for reason in reasons:
        kind, _, detail = reason.partition(":")
        if kind == "llm_call_failed" and detail:
            names.add(detail)
        elif kind == "stage_failed" and detail in STAGE_CALLS:
            names.add(STAGE_CALLS[detail])
        elif reason in _REASON_CALLS:
            names.add(_REASON_CALLS[reason])
        elif reason == "gate2_fallback":
            names.add("evidence_gate")
        elif reason.startswith(_INFRASTRUCTURE_PREFIXES):
            continue
        elif not reason.startswith(("stage_failed:", "llm_call_failed:")):
            names.add(ALL_CALLS)  # pipeline_error, no_action or an unknown reason
    return frozenset(names)


def summarise(reason_lists: Iterable[Sequence[str]]) -> dict[str, int]:
    """Count of turns per reason kind (``stage_failed:x`` kept with its detail)."""
    counts: dict[str, int] = {}
    for reasons in reason_lists:
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))
