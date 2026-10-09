"""Degraded-turn classification (finding ablation-study-validity-1/-2): every fallback is flagged."""

import pytest

from evaluation.degraded import (
    ALL_CALLS,
    annotate_attempt,
    bypass_calls,
    degradation_reasons,
    record_degraded,
)
from medbridge.core.templates import GENERALIZE_FALLBACK_TEMPLATE, PIPELINE_ERROR_TEMPLATE


def _record(**explanation):
    base_expl = {
        "gate1": {"action": "PROCEED", "source": "llm"},
        "gate2": {"action": "ANSWER", "source": "llm"},
        "retrieval": [{"chunk_id": "c1", "score_rerank": 2.0}],
        "timings": [
            {"stage": "retrieval", "ms": 5.0, "status": "ok"},
            {"stage": "generation", "ms": 5.0, "status": "ok"},
        ],
        "llm_calls": [],
    }
    base_expl.update(explanation)
    return {
        "action": "ANSWER",
        "response_text": "Salt raises BP [1].",
        "error": None,
        "evidence_error": None,
        "evidence_texts": [{"chunk_id": "c1", "text": "t", "text_source": "qdrant"}],
        "llm_calls": [{"call_name": "response_generator", "success": True}],
        "explanation": base_expl,
    }


def test_clean_turn_is_not_degraded():
    assert degradation_reasons(_record()) == []
    assert not record_degraded(_record())


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"gate1": {"action": "PROCEED", "source": "fallback"}}, "gate1_fallback"),
        ({"gate2": {"action": "GENERALIZE", "source": "fallback"}}, "gate2_fallback"),
        (
            {
                "gate2": {
                    "action": "GENERALIZE",
                    "source": "calibrated",
                    "features": {"failed_votes": 2},
                }
            },
            "gate2_votes_failed",
        ),
        (
            {"timings": [{"stage": "extraction", "ms": 1, "status": "failed"}]},
            "stage_failed:extraction",
        ),
        (
            {"timings": [{"stage": "load_state", "ms": 1, "status": "failed"}]},
            "stage_failed:load_state",
        ),
        (
            {"timings": [{"stage": "persistence", "ms": 1, "status": "failed"}]},
            "stage_failed:persistence",
        ),
        (
            {"timings": [{"stage": "retrieval", "ms": 1, "status": "failed"}], "retrieval": []},
            "stage_failed:retrieval",
        ),
        ({"retrieval_status": "failed"}, "retrieval_failed"),
        ({"retrieval": []}, "retrieval_empty"),  # outage swallowed by the retriever
        ({"gate3": {"status": "unavailable", "available": False}}, "gate3_unavailable"),
        ({"gate3": {"status": "failed", "error": "RuntimeError"}}, "gate3_failed"),
    ],
)
def test_each_fallback_is_flagged(change, reason):
    assert reason in degradation_reasons(_record(**change))


def test_record_level_failures_are_flagged():
    rec = _record()
    rec["llm_calls"] = [{"call_name": "evidence_gate", "success": False}]
    assert "llm_call_failed:evidence_gate" in degradation_reasons(rec)
    rec = _record()
    rec["response_text"] = GENERALIZE_FALLBACK_TEMPLATE
    assert "generator_fallback_template" in degradation_reasons(rec)
    rec = _record()
    rec["response_text"] = "Prefix\n\n" + GENERALIZE_FALLBACK_TEMPLATE  # loop-breaker prefix
    assert "generator_fallback_template" in degradation_reasons(rec)
    rec = _record()
    rec["response_text"] = PIPELINE_ERROR_TEMPLATE
    assert "pipeline_error" in degradation_reasons(rec)
    rec = _record()
    rec["evidence_error"] = "ConnectError: qdrant"
    assert "evidence_fetch_failed" in degradation_reasons(rec)
    rec = _record()
    rec["evidence_texts"][0]["text_source"] = "excerpt"
    assert "evidence_text_missing" in degradation_reasons(rec)
    assert degradation_reasons({"error": "db down", "action": None}) == ["error"]


def test_explicit_retrieval_status_is_trusted():
    assert "retrieval_failed" in degradation_reasons(
        _record(retrieval_status="failed", retrieval=[])
    )
    # the search ran and matched nothing: a valid outcome, not an outage
    assert degradation_reasons(_record(retrieval_status="empty", retrieval=[])) == []


def test_discarded_or_skipped_retrieval_is_not_empty_retrieval():
    rec = _record(
        retrieval=[],
        timings=[
            {"stage": "retrieval", "ms": 3, "status": "discarded"},
            {"stage": "generation", "ms": 3, "status": "ok"},
        ],
    )
    assert degradation_reasons(rec) == []
    rec = _record(retrieval=[], timings=[{"stage": "retrieval", "ms": 0, "status": "skipped"}])
    assert degradation_reasons(rec) == []


def test_later_turns_inherit_an_earlier_degradation():
    turn0 = _record(timings=[{"stage": "persistence", "ms": 1, "status": "failed"}], retrieval=[])
    turn1 = _record()
    reasons = annotate_attempt([turn0, turn1])
    assert "stage_failed:persistence" in reasons[0]
    assert reasons[1] == ["earlier_turn_degraded"]
    assert annotate_attempt([_record(), _record()]) == [[], []]


def test_bypass_targets_only_the_failed_calls():
    assert bypass_calls(["stage_failed:generation"]) == {"response_generator"}
    assert bypass_calls(["gate1_fallback", "gate2_fallback"]) == {"context_gate", "evidence_gate"}
    assert bypass_calls(["llm_call_failed:context_extractor"]) == {"context_extractor"}
    assert bypass_calls(["generator_fallback_template"]) == {"response_generator"}
    # infrastructure: the item is re-run, no cached LLM response needs bypassing
    assert (
        bypass_calls(["error", "stage_failed:persistence", "retrieval_empty", "gate3_unavailable"])
        == frozenset()
    )
    assert bypass_calls(["pipeline_error"]) == {ALL_CALLS}
