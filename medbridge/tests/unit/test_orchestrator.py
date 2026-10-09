"""
Unit tests for the Pipeline Orchestrator (Module M-08, finalization §3.1/§3.2).

Every collaborator (turn store, redactor, emergency classifier, extractor,
both gates, retriever, reranker, generator, grounding verifier) is a
deterministic fake, so these tests are hermetic: no LLM, PostgreSQL or Qdrant.

Covered: the five routes, the loop-breaker (threshold 2 forces GENERALIZE and
resets the counter), template bypass, emergency-before-any-LLM, every spec
fallback, speculative-retrieval semantics, citation validation + downgrade,
small talk, ablation presets, the Gate 3 grounding verifier, regeneration,
the single persistence call per turn and the never-throw guarantee.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from medbridge.ai.schemas.context_gate import ContextGateOutput
from medbridge.ai.schemas.evidence_gate import EvidenceGateOutput
from medbridge.ai.schemas.extractor import DeltaEvent, ExtractorOutput
from medbridge.ai.schemas.response_generator import ResponseGeneratorOutput
from medbridge.api.schemas.enums import ActionEnum, EventTypeEnum
from medbridge.config import get_settings
from medbridge.core.orchestrator import PipelineOrchestrator, get_orchestrator
from medbridge.core.pipeline_types import AblationConfig, PipelineEvent, StageName
from medbridge.core.templates import (
    ABSTAIN_TEMPLATE,
    GENERALIZE_FALLBACK_TEMPLATE,
    LOOP_BREAKER_PREFIX,
    PIPELINE_ERROR_TEMPLATE,
    SMALL_TALK_TEMPLATES,
    get_template,
)
from medbridge.retrieval.hybrid_retriever import RetrievedChunk
from medbridge.state.projector import StateProjector
from medbridge.state.turn_store import ChatTurn, SessionState, TurnRecord

SESSION_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")
MESSAGE = "I'm John Smith, my BP is 150/95 on amlodipine. What target should I aim for?"
REDACTED = "I'm [NAME], my BP is 150/95 on amlodipine. What target should I aim for?"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def make_chunks(n: int = 6) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"chunk-{i}",
            chunk_text=f"Guideline evidence number {i} about blood pressure targets. " * 3,
            guideline_id="AHA_ACC_2025",
            section_title=f"Section {i}",
            page_number=10 + i,
            source_url="https://example.org/guideline",
            doc_title="2025 AHA/ACC High Blood Pressure Guideline",
            org="AHA/ACC",
            year=2025,
            score=1.0 / (60 + i),
            score_rrf=1.0 / (60 + i),
        )
        for i in range(1, n + 1)
    ]


class FakeStore:
    """In-memory stand-in for ``medbridge.state.turn_store``."""

    def __init__(
        self,
        state: SessionState | None = None,
        history: list[ChatTurn] | None = None,
        last_message: tuple[uuid.UUID, str] | None = None,
        persist_error: Exception | None = None,
        load_error: Exception | None = None,
    ) -> None:
        self.state = state or SessionState(snapshot={}, soft_ask_count=0)
        self.history = history or []
        self.last_message = last_message
        self.persist_error = persist_error
        self.load_error = load_error
        self.records: list[TurnRecord] = []
        self.history_limits: list[int] = []
        self.ids = (uuid.uuid4(), uuid.uuid4())

    async def load_session_state(self, db: Any, session_id: uuid.UUID) -> SessionState:
        if self.load_error is not None:
            raise self.load_error
        return self.state

    async def get_recent_turns(self, db: Any, session_id: uuid.UUID, n: int) -> list[ChatTurn]:
        self.history_limits.append(n)
        return self.history[-n:]

    async def get_last_user_message(
        self, db: Any, session_id: uuid.UUID
    ) -> tuple[uuid.UUID, str] | None:
        return self.last_message

    async def persist_turn(self, db: Any, record: TurnRecord) -> tuple[uuid.UUID, uuid.UUID]:
        if self.persist_error is not None:
            raise self.persist_error
        self.records.append(record)
        return self.ids


def fake_redact(text: str) -> SimpleNamespace:
    redacted = text.replace("John Smith", "[NAME]")
    count = 1 if redacted != text else 0
    return SimpleNamespace(text=redacted, count=count, categories={"NAME": count} if count else {})


async def rerank_top(
    query: str, chunks: list[RetrievedChunk], top_k: int = 5
) -> list[RetrievedChunk]:
    return [
        c.model_copy(update={"score_rerank": 5.0 - i, "score": 5.0 - i})
        for i, c in enumerate(chunks[:top_k])
    ]


class EventRecorder:
    def __init__(self) -> None:
        self.events: list[PipelineEvent] = []

    async def __call__(self, event: PipelineEvent) -> None:
        self.events.append(event)

    def stages(self, status: str) -> list[str]:
        return [e.stage.value for e in self.events if e.status == status and e.stage]


@pytest.fixture
def chunks() -> list[RetrievedChunk]:
    return make_chunks()


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def components(chunks: list[RetrievedChunk]) -> SimpleNamespace:
    """Fakes for every orchestrator collaborator (default: a clean ANSWER turn)."""
    extractor = MagicMock()
    extractor.extract = AsyncMock(
        return_value=ExtractorOutput(
            delta_events=[
                DeltaEvent(
                    event_type=EventTypeEnum.BP_READING, payload={"systolic": 150, "diastolic": 95}
                )
            ],
            search_query="blood pressure target amlodipine",
            raw_intent="ask BP target",
        )
    )
    context_gate = MagicMock()
    context_gate.evaluate = AsyncMock(
        return_value=ContextGateOutput(action="PROCEED", rationale="Context is sufficient.")
    )
    retriever = MagicMock()
    retriever.hybrid_search = AsyncMock(return_value=chunks)
    reranker = MagicMock()
    reranker.rerank_chunks = AsyncMock(side_effect=rerank_top)
    evidence_gate = MagicMock()
    evidence_gate.evaluate = AsyncMock(
        return_value=EvidenceGateOutput(
            action="ANSWER", evidence_sufficient=True, rationale="Evidence covers the target."
        )
    )
    generator = MagicMock()
    generator.generate = AsyncMock(
        return_value=ResponseGeneratorOutput(
            response_text="The usual target is below 130/80 mmHg [2]. Keep checking at home [1]."
        )
    )
    classifier = SimpleNamespace(classify=MagicMock(return_value=None))
    return SimpleNamespace(
        emergency_classifier=classifier,
        context_extractor=extractor,
        state_projector=StateProjector(),
        context_gate=context_gate,
        hybrid_retriever=retriever,
        reranker=reranker,
        evidence_gate=evidence_gate,
        response_generator=generator,
    )


def make_settings(**overrides: Any) -> Any:
    base = {
        "PII_REDACTION_ENABLED": True,
        "SPECULATIVE_RETRIEVAL": False,
        "SMALL_TALK_ENABLED": True,
        "CITATION_VALIDATION_ENABLED": True,
        "EVIDENCE_GATE_CALIBRATION_ENABLED": False,
        "GROUNDING_VERIFIER_ENABLED": False,
        "SOFT_ASK_MAX_COUNT": 2,
        "EXTRACTOR_HISTORY_TURNS": 4,
    }
    base.update(overrides)
    return get_settings().model_copy(update=base)


def build(
    components: SimpleNamespace,
    store: FakeStore,
    *,
    verifier: Any | None = None,
    redactor: Any = fake_redact,
    **settings_overrides: Any,
) -> PipelineOrchestrator:
    return PipelineOrchestrator(
        **vars(components),
        settings=make_settings(**settings_overrides),
        store=store,
        redactor=redactor,
        grounding_verifier=verifier,
    )


async def run(
    orchestrator: PipelineOrchestrator,
    message: str = MESSAGE,
    **kwargs: Any,
) -> Any:
    return await orchestrator.process_message(SESSION_ID, message, db=MagicMock(), **kwargs)


def generator_calls(components: SimpleNamespace) -> list[dict[str, Any]]:
    return [c.kwargs for c in components.response_generator.generate.await_args_list]


# ---------------------------------------------------------------------------
# 1. The five routes
# ---------------------------------------------------------------------------


async def test_answer_route_runs_every_stage_in_order(components, store, chunks):
    recorder = EventRecorder()
    response = await run(build(components, store), on_event=recorder)

    assert response.action == ActionEnum.ANSWER
    # markers renumbered by first appearance: [2] → [1], [1] → [2]
    assert response.response_text == (
        "The usual target is below 130/80 mmHg [1]. Keep checking at home [2]."
    )
    assert [c.chunk_id for c in response.citations] == ["chunk-2", "chunk-1"]
    assert response.citations[0].source == "AHA/ACC 2025"
    assert response.citations[0].page == 12
    assert response.message_id == store.ids[1]
    assert response.user_message_id == store.ids[0]

    assert recorder.stages("start") == [
        "privacy",
        "emergency",
        "load_state",
        "small_talk",
        "extraction",
        "context_gate",
        "retrieval",
        "rerank",
        "evidence_gate",
        "generation",
        "persistence",
    ]
    exp = response.explanation
    assert exp.gate1.action == "PROCEED" and exp.gate1.source == "llm"
    assert exp.gate2.action == "ANSWER" and exp.gate2.source == "llm"
    assert [r.chunk_id for r in exp.retrieval] == [c.chunk_id for c in chunks[:5]]
    assert [r.cited for r in exp.retrieval] == [True, True, False, False, False]
    assert exp.retrieval[0].score_rerank == 5.0 and exp.retrieval[0].score_rrf > 0
    assert exp.citation_validation == {"valid": 2, "stripped": 0, "downgraded": False}
    assert exp.models == {
        "gates": get_settings().gates_model,
        "generator": get_settings().generator_model,
    }

    kwargs = generator_calls(components)[0]
    assert kwargs["action"] == ActionEnum.ANSWER
    assert [c.chunk_id for c in kwargs["chunks"]] == [c.chunk_id for c in chunks[:5]]
    assert kwargs["mode"] == "grounded"


@pytest.mark.parametrize("gate2_action", ["GENERALIZE", "ABSTAIN", "ESCALATE"])
async def test_gate2_routes(components, store, gate2_action):
    components.evidence_gate.evaluate.return_value = EvidenceGateOutput(
        action=gate2_action, evidence_sufficient=False, rationale="r"
    )
    response = await run(build(components, store))
    assert response.action == ActionEnum(gate2_action)
    if gate2_action == "GENERALIZE":
        assert generator_calls(components)[0]["action"] == ActionEnum.GENERALIZE
    else:
        # Template bypass (Non-Negotiable #6): the generator is never invoked.
        components.response_generator.generate.assert_not_called()
        assert response.response_text == get_template(ActionEnum(gate2_action))
        assert response.citations == []
    assert len(store.records) == 1
    assert store.records[0].gate_2_action == gate2_action


async def test_soft_ask_route_skips_retrieval_and_gate2(components, store):
    components.context_gate.evaluate.return_value = ContextGateOutput(
        action="SOFT-ASK", rationale="Need meds.", missing_fields=["current_medications"]
    )
    response = await run(build(components, store))

    assert response.action == ActionEnum.SOFT_ASK
    assert response.soft_ask_count == 1
    components.hybrid_retriever.hybrid_search.assert_not_called()
    components.evidence_gate.evaluate.assert_not_called()
    kwargs = generator_calls(components)[0]
    assert kwargs["action"] == ActionEnum.SOFT_ASK
    assert kwargs["chunks"] == []
    assert kwargs["missing_fields"] == ["current_medications"]
    record = store.records[0]
    assert record.soft_ask_count == 1
    assert record.gate_1_action == "SOFT-ASK" and record.gate_2_action is None
    assert record.evidence_chunk_ids == []


async def test_soft_ask_strips_markers_from_question(components, store):
    components.context_gate.evaluate.return_value = ContextGateOutput(
        action="SOFT-ASK", rationale="r", missing_fields=["age"]
    )
    components.response_generator.generate.return_value = ResponseGeneratorOutput(
        response_text="Could you tell me your age [1]?"
    )
    response = await run(build(components, store))
    assert response.response_text == "Could you tell me your age?"


# ---------------------------------------------------------------------------
# 2. Loop-breaker (Non-Negotiable #4)
# ---------------------------------------------------------------------------


async def test_loop_breaker_at_threshold_forces_generalize_and_resets(components, store):
    store.state = SessionState(snapshot={}, soft_ask_count=2)
    components.context_gate.evaluate.return_value = ContextGateOutput(
        action="SOFT-ASK", rationale="still missing", missing_fields=["age"]
    )
    response = await run(build(components, store))

    assert response.action == ActionEnum.GENERALIZE
    assert response.soft_ask_count == 0
    assert response.response_text.startswith(LOOP_BREAKER_PREFIX)
    components.evidence_gate.evaluate.assert_not_called()
    components.hybrid_retriever.hybrid_search.assert_awaited_once()
    assert generator_calls(components)[0]["action"] == ActionEnum.GENERALIZE
    assert response.explanation.gate2.source == "loop_breaker"
    record = store.records[0]
    assert record.soft_ask_count == 0
    assert record.gate_2_action == "GENERALIZE"
    assert "loop-breaker" in record.gate_2_rationale.lower()


async def test_soft_ask_below_threshold_increments_to_two(components, store):
    store.state = SessionState(snapshot={}, soft_ask_count=1)
    components.context_gate.evaluate.return_value = ContextGateOutput(
        action="SOFT-ASK", rationale="r", missing_fields=["age"]
    )
    response = await run(build(components, store))
    assert response.action == ActionEnum.SOFT_ASK
    assert response.soft_ask_count == 2


async def test_loop_breaker_with_generator_failure_keeps_prefix(components, store):
    store.state = SessionState(snapshot={}, soft_ask_count=2)
    components.context_gate.evaluate.return_value = ContextGateOutput(
        action="SOFT-ASK", rationale="r", missing_fields=["age"]
    )
    components.response_generator.generate.return_value = None
    response = await run(build(components, store))
    assert response.action == ActionEnum.GENERALIZE
    assert response.response_text == f"{LOOP_BREAKER_PREFIX}\n\n{GENERALIZE_FALLBACK_TEMPLATE}"


async def test_proceed_resets_counter(components, store):
    store.state = SessionState(snapshot={}, soft_ask_count=1)
    response = await run(build(components, store))
    assert response.soft_ask_count == 0
    assert store.records[0].soft_ask_count == 0


# ---------------------------------------------------------------------------
# 3. Emergency fast-path (Non-Negotiable #1) and privacy
# ---------------------------------------------------------------------------


async def test_emergency_fastpath_runs_before_any_llm(components, store):
    seen: list[str] = []

    def classify(text: str) -> ActionEnum | None:
        seen.append(text)
        return ActionEnum.ESCALATE

    components.emergency_classifier = SimpleNamespace(classify=classify)
    store.state = SessionState(snapshot={}, soft_ask_count=1)
    response = await run(build(components, store))

    assert response.action == ActionEnum.ESCALATE
    assert "112" in response.response_text and "108" in response.response_text
    assert "911" in response.response_text
    assert seen == [MESSAGE]  # classifier gets the RAW text (higher recall)
    components.context_extractor.extract.assert_not_awaited()
    components.context_gate.evaluate.assert_not_called()
    components.hybrid_retriever.hybrid_search.assert_not_called()
    components.evidence_gate.evaluate.assert_not_called()
    components.response_generator.generate.assert_not_called()
    assert response.soft_ask_count == 1  # counter untouched
    assert response.explanation.gate2.source == "fastpath"
    record = store.records[0]
    assert record.final_action == ActionEnum.ESCALATE
    assert record.gate_2_action == "ESCALATE" and record.gate_1_action is None
    assert record.user_message == REDACTED  # persisted text is redacted
    assert record.soft_ask_count == 1


async def test_emergency_detailed_classifier_records_rules(components, store):
    detailed = SimpleNamespace(
        action=ActionEnum.ESCALATE, matched_rules=["chest_pain", "bp_crisis"]
    )
    components.emergency_classifier = SimpleNamespace(
        classify_detailed=MagicMock(return_value=detailed), classify=MagicMock()
    )
    response = await run(build(components, store))
    assert response.explanation.emergency == {
        "action": "ESCALATE",
        "matched_rules": ["chest_pain", "bp_crisis"],
    }
    components.emergency_classifier.classify.assert_not_called()
    assert "chest_pain" in store.records[0].gate_2_rationale


async def test_emergency_plain_callable_classifier(components, store):
    components.emergency_classifier = lambda text: ActionEnum.ESCALATE
    response = await run(build(components, store))
    assert response.action == ActionEnum.ESCALATE


async def test_emergency_classifier_crash_continues_pipeline(components, store):
    components.emergency_classifier = SimpleNamespace(classify=MagicMock(side_effect=RuntimeError))
    response = await run(build(components, store))
    assert response.action == ActionEnum.ANSWER
    timing = {t.stage: t.status for t in response.explanation.timings}
    assert timing["emergency"] == "failed"


async def test_redacted_text_used_everywhere_except_emergency(components, store):
    seen: list[str] = []
    components.emergency_classifier = SimpleNamespace(classify=lambda t: seen.append(t))
    components.context_extractor.extract.return_value = None  # search query falls back to text
    response = await run(build(components, store))

    assert seen == [MESSAGE]
    assert components.context_extractor.extract.await_args.args[0] == REDACTED
    assert components.context_gate.evaluate.await_args.args[1] == REDACTED
    assert components.hybrid_retriever.hybrid_search.await_args.args[0] == REDACTED
    assert components.evidence_gate.evaluate.await_args.kwargs["query"] == REDACTED
    assert generator_calls(components)[0]["query"] == REDACTED
    assert store.records[0].user_message == REDACTED
    assert response.redaction_count == 1
    assert response.explanation.redaction_categories == {"NAME": 1}
    trace_json = json.dumps(store.records[0].trace)
    assert "John Smith" not in trace_json and REDACTED not in trace_json


async def test_redaction_disabled_skips_redactor(components, store):
    redactor = MagicMock(side_effect=AssertionError("must not be called"))
    response = await run(build(components, store, redactor=redactor, PII_REDACTION_ENABLED=False))
    assert response.action == ActionEnum.ANSWER
    assert store.records[0].user_message == MESSAGE


async def test_redaction_failure_fails_closed(components, store):
    redactor = MagicMock(side_effect=RuntimeError("regex engine down"))
    response = await run(build(components, store, redactor=redactor))
    assert response.action == ActionEnum.GENERALIZE
    assert response.response_text == PIPELINE_ERROR_TEMPLATE
    components.context_extractor.extract.assert_not_awaited()
    assert store.records == []


# ---------------------------------------------------------------------------
# 4. Spec fallbacks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("failure", [None, RuntimeError("boom")])
async def test_extractor_failure_falls_back_to_raw_query(components, store, failure):
    if failure is None:
        components.context_extractor.extract.return_value = None
    else:
        components.context_extractor.extract.side_effect = failure
    response = await run(build(components, store))
    assert response.action == ActionEnum.ANSWER
    assert components.hybrid_retriever.hybrid_search.await_args.args[0] == REDACTED
    assert store.records[0].delta_events == []
    assert store.records[0].new_snapshot is None


@pytest.mark.parametrize("failure", [None, TimeoutError("slow")])
async def test_context_gate_failure_proceeds(components, store, failure):
    if failure is None:
        components.context_gate.evaluate.return_value = None
    else:
        components.context_gate.evaluate.side_effect = failure
    response = await run(build(components, store))
    assert response.explanation.gate1.action == "PROCEED"
    assert response.explanation.gate1.source == "fallback"
    components.evidence_gate.evaluate.assert_awaited_once()


@pytest.mark.parametrize("failure", [None, RuntimeError("boom")])
async def test_evidence_gate_failure_generalizes(components, store, failure):
    if failure is None:
        components.evidence_gate.evaluate.return_value = None
    else:
        components.evidence_gate.evaluate.side_effect = failure
    response = await run(build(components, store))
    assert response.action == ActionEnum.GENERALIZE
    assert response.explanation.gate2.source == "fallback"
    assert generator_calls(components)[0]["action"] == ActionEnum.GENERALIZE


@pytest.mark.parametrize("failure", [None, RuntimeError("boom")])
async def test_generator_failure_returns_prevetted_template(components, store, failure):
    if failure is None:
        components.response_generator.generate.return_value = None
    else:
        components.response_generator.generate.side_effect = failure
    response = await run(build(components, store))
    assert response.action == ActionEnum.GENERALIZE
    assert response.response_text == GENERALIZE_FALLBACK_TEMPLATE
    assert response.citations == []
    timing = {t.stage: t.status for t in response.explanation.timings}
    assert timing["generation"] == "failed"


async def test_retrieval_failure_yields_empty_evidence(components, store):
    components.hybrid_retriever.hybrid_search.side_effect = ConnectionError("qdrant down")
    components.evidence_gate.evaluate.return_value = EvidenceGateOutput(
        action="ABSTAIN", evidence_sufficient=False, rationale="no evidence"
    )
    response = await run(build(components, store))
    assert components.evidence_gate.evaluate.await_args.kwargs["chunks"] == []
    assert response.action == ActionEnum.ABSTAIN
    assert response.response_text == ABSTAIN_TEMPLATE
    components.reranker.rerank_chunks.assert_not_called()


async def test_reranker_failure_keeps_rrf_order(components, store, chunks):
    components.reranker.rerank_chunks.side_effect = RuntimeError("cuda oom")
    response = await run(build(components, store))
    passed = components.evidence_gate.evaluate.await_args.kwargs["chunks"]
    assert [c.chunk_id for c in passed] == [c.chunk_id for c in chunks[:5]]
    timing = {t.stage: t.status for t in response.explanation.timings}
    assert timing["rerank"] == "failed"


async def test_projection_failure_keeps_snapshot(components, store):
    components.state_projector = MagicMock()
    components.state_projector.apply_events.side_effect = ValueError("bad payload")
    response = await run(build(components, store))
    assert response.action == ActionEnum.ANSWER
    assert store.records[0].delta_events == []
    assert store.records[0].new_snapshot is None


# ---------------------------------------------------------------------------
# 5. Speculative retrieval (ADL-027)
# ---------------------------------------------------------------------------


def slow_retriever(components: SimpleNamespace, chunks: list[RetrievedChunk], state: dict) -> None:
    async def search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        state["started"] = True
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        state["finished"] = True
        return chunks

    components.hybrid_retriever.hybrid_search = AsyncMock(side_effect=search)


def slow_gate(components: SimpleNamespace, output: ContextGateOutput, state: dict) -> None:
    async def evaluate(snapshot: Any, message: str, raw_intent: str) -> ContextGateOutput:
        await asyncio.sleep(0.02)
        state["retrieval_running_during_gate"] = state.get("started", False)
        return output

    components.context_gate.evaluate = AsyncMock(side_effect=evaluate)


async def test_speculative_soft_ask_discards_retrieval(components, store, chunks):
    state: dict[str, bool] = {}
    slow_retriever(components, chunks, state)
    slow_gate(
        components,
        ContextGateOutput(action="SOFT-ASK", rationale="r", missing_fields=["age"]),
        state,
    )
    response = await run(build(components, store, SPECULATIVE_RETRIEVAL=True))

    assert response.action == ActionEnum.SOFT_ASK
    assert state["retrieval_running_during_gate"] is True  # launched before Gate 1 returned
    assert state.get("cancelled") is True and "finished" not in state
    components.evidence_gate.evaluate.assert_not_called()
    assert generator_calls(components)[0]["chunks"] == []  # results never reach the generator
    assert response.explanation.retrieval == []
    assert response.explanation.speculative_retrieval == {
        "launched": True,
        "used": False,
        "discarded": True,
        "saved_ms": 0.0,
    }
    assert store.records[0].evidence_chunk_ids == []
    timing = {t.stage: t.status for t in response.explanation.timings}
    assert timing["retrieval"] == "discarded"


async def test_speculative_soft_ask_discards_completed_retrieval(components, store, chunks):
    components.context_gate.evaluate.return_value = ContextGateOutput(
        action="SOFT-ASK", rationale="r", missing_fields=["age"]
    )

    async def slow_evaluate(*args: Any) -> ContextGateOutput:
        await asyncio.sleep(0.02)  # retrieval (instant) finishes first
        return ContextGateOutput(action="SOFT-ASK", rationale="r", missing_fields=["age"])

    components.context_gate.evaluate = AsyncMock(side_effect=slow_evaluate)
    response = await run(build(components, store, SPECULATIVE_RETRIEVAL=True))
    assert response.action == ActionEnum.SOFT_ASK
    components.hybrid_retriever.hybrid_search.assert_awaited_once()
    assert generator_calls(components)[0]["chunks"] == []
    statuses = [
        t.status for t in response.explanation.timings if t.stage in ("retrieval", "rerank")
    ]
    assert statuses == ["discarded", "discarded"]


async def test_speculative_proceed_uses_retrieval(components, store, chunks):
    state: dict[str, bool] = {}
    slow_retriever(components, chunks, state)
    slow_gate(components, ContextGateOutput(action="PROCEED", rationale="ok"), state)
    response = await run(build(components, store, SPECULATIVE_RETRIEVAL=True))

    assert response.action == ActionEnum.ANSWER
    assert state["retrieval_running_during_gate"] is True
    components.hybrid_retriever.hybrid_search.assert_awaited_once()
    passed = components.evidence_gate.evaluate.await_args.kwargs["chunks"]
    assert [c.chunk_id for c in passed] == [c.chunk_id for c in chunks[:5]]
    spec = response.explanation.speculative_retrieval
    assert spec["launched"] and spec["used"] and not spec["discarded"]
    assert spec["saved_ms"] > 0


async def test_speculative_results_identical_on_and_off(chunks):
    outputs = []
    for speculative in (True, False):
        components = fresh_components(chunks)
        store = FakeStore()
        response = await run(build(components, store, SPECULATIVE_RETRIEVAL=speculative))
        outputs.append(
            (
                response.action,
                response.response_text,
                [c.model_dump() for c in response.citations],
                [c.chunk_id for c in components.evidence_gate.evaluate.await_args.kwargs["chunks"]],
                store.records[0].evidence_chunk_ids,
            )
        )
    assert outputs[0] == outputs[1]


def fresh_components(chunks: list[RetrievedChunk]) -> SimpleNamespace:
    """A fresh default component set (fixtures are per-test, this is per-iteration)."""
    extractor = MagicMock()
    extractor.extract = AsyncMock(
        return_value=ExtractorOutput(search_query="bp target", raw_intent="target")
    )
    gate = MagicMock()
    gate.evaluate = AsyncMock(return_value=ContextGateOutput(action="PROCEED", rationale="ok"))
    retriever = MagicMock()
    retriever.hybrid_search = AsyncMock(return_value=chunks)
    reranker = MagicMock()
    reranker.rerank_chunks = AsyncMock(side_effect=rerank_top)
    evidence = MagicMock()
    evidence.evaluate = AsyncMock(
        return_value=EvidenceGateOutput(action="ANSWER", evidence_sufficient=True, rationale="ok")
    )
    generator = MagicMock()
    generator.generate = AsyncMock(
        return_value=ResponseGeneratorOutput(response_text="Aim below 130/80 [3].")
    )
    return SimpleNamespace(
        emergency_classifier=SimpleNamespace(classify=lambda t: None),
        context_extractor=extractor,
        state_projector=StateProjector(),
        context_gate=gate,
        hybrid_retriever=retriever,
        reranker=reranker,
        evidence_gate=evidence,
        response_generator=generator,
    )


async def test_speculative_not_launched_when_gate1_bypassed(components, store):
    response = await run(
        build(components, store, SPECULATIVE_RETRIEVAL=True), ablation=AblationConfig.preset("C4")
    )
    assert response.explanation.speculative_retrieval["launched"] is False
    components.context_gate.evaluate.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Citation validation & ANSWER → GENERALIZE downgrade (ADL-028)
# ---------------------------------------------------------------------------


async def test_invalid_markers_stripped_and_citations_from_metadata(components, store, chunks):
    components.response_generator.generate.return_value = ResponseGeneratorOutput(
        response_text="Aim below 130/80 [1][9]. Salt matters [7].",
    )
    response = await run(build(components, store))
    assert response.response_text == "Aim below 130/80 [1]. Salt matters."
    assert len(response.citations) == 1
    citation = response.citations[0]
    assert citation.chunk_id == "chunk-1"
    assert citation.excerpt.startswith("Guideline evidence number 1")
    assert citation.title == "2025 AHA/ACC High Blood Pressure Guideline"
    assert response.explanation.citation_validation == {
        "valid": 1,
        "stripped": 2,
        "downgraded": False,
    }


async def test_answer_without_valid_citation_downgrades(components, store):
    components.response_generator.generate.return_value = ResponseGeneratorOutput(
        response_text="Aim below 130/80 [8].",
    )
    response = await run(build(components, store))
    assert response.action == ActionEnum.GENERALIZE
    assert response.response_text == "Aim below 130/80."
    assert response.explanation.citation_validation["downgraded"] is True
    record = store.records[0]
    assert record.gate_2_action == "ANSWER" and record.final_action == ActionEnum.GENERALIZE


async def test_downgrade_disabled_by_setting(components, store):
    components.response_generator.generate.return_value = ResponseGeneratorOutput(
        response_text="Aim below 130/80.",
    )
    response = await run(build(components, store, CITATION_VALIDATION_ENABLED=False))
    assert response.action == ActionEnum.ANSWER


# ---------------------------------------------------------------------------
# 7. Small talk (ADL-029)
# ---------------------------------------------------------------------------


async def test_small_talk_uses_template_without_llm(components, store):
    store.state = SessionState(snapshot={}, soft_ask_count=1)
    response = await run(build(components, store), message="Hello there!")
    assert response.action == ActionEnum.GENERALIZE
    assert response.response_text == SMALL_TALK_TEMPLATES["greeting"]
    components.context_extractor.extract.assert_not_awaited()
    components.context_gate.evaluate.assert_not_called()
    components.evidence_gate.evaluate.assert_not_called()
    components.response_generator.generate.assert_not_called()
    components.hybrid_retriever.hybrid_search.assert_not_called()
    assert response.explanation.gate1.source == "rule"
    assert response.explanation.gate2.source == "rule"
    assert response.soft_ask_count == 1  # loop-breaker counter untouched
    record = store.records[0]
    assert record.gate_1_action is None and record.gate_2_action is None
    assert store.history_limits == [4]


async def test_small_talk_disabled_by_ablation(components, store):
    ablation = AblationConfig(name="no-smalltalk", use_small_talk=False)
    response = await run(build(components, store), message="Hello there!", ablation=ablation)
    components.context_extractor.extract.assert_awaited_once()
    assert response.action == ActionEnum.ANSWER


async def test_clinical_message_is_not_small_talk(components, store):
    response = await run(build(components, store), message="hi, my bp is 160/100")
    components.context_extractor.extract.assert_awaited_once()
    assert response.action == ActionEnum.ANSWER


# ---------------------------------------------------------------------------
# 8. Ablation presets (evaluation harness)
# ---------------------------------------------------------------------------


async def test_c0_direct_llm(components, store):
    response = await run(build(components, store), ablation=AblationConfig.preset("C0"))
    assert response.action == ActionEnum.ANSWER
    components.hybrid_retriever.hybrid_search.assert_not_called()
    components.context_gate.evaluate.assert_not_called()
    components.evidence_gate.evaluate.assert_not_called()
    kwargs = generator_calls(components)[0]
    assert kwargs["mode"] == "direct" and kwargs["chunks"] == []
    exp = response.explanation
    assert exp.gate1.source == "ablation" and exp.gate2.source == "ablation"
    assert exp.ablation == "C0"
    assert response.citations == []  # fabricated markers stripped, no downgrade in C0
    assert store.records[0].gate_1_action is None and store.records[0].gate_2_action is None


async def test_c0_has_no_emergency_fastpath(components, store):
    components.emergency_classifier = SimpleNamespace(classify=lambda t: ActionEnum.ESCALATE)
    response = await run(build(components, store), ablation=AblationConfig.preset("C0"))
    assert response.action == ActionEnum.ANSWER


@pytest.mark.parametrize("preset,mode", [("C1", "grounded"), ("C2", "prompt_safety")])
async def test_c1_c2_no_gates_always_answer(components, store, preset, mode):
    components.context_gate.evaluate.return_value = ContextGateOutput(
        action="SOFT-ASK", rationale="r", missing_fields=["age"]
    )
    components.response_generator.generate.return_value = ResponseGeneratorOutput(
        response_text="No citations at all."
    )
    response = await run(build(components, store), ablation=AblationConfig.preset(preset))
    assert response.action == ActionEnum.ANSWER  # no downgrade without the Evidence Gate
    components.context_gate.evaluate.assert_not_called()
    components.evidence_gate.evaluate.assert_not_called()
    components.hybrid_retriever.hybrid_search.assert_awaited_once()
    assert generator_calls(components)[0]["mode"] == mode


async def test_c3_context_gate_only(components, store):
    components.evidence_gate.evaluate.return_value = EvidenceGateOutput(
        action="ABSTAIN", evidence_sufficient=False, rationale="r"
    )
    response = await run(build(components, store), ablation=AblationConfig.preset("C3"))
    assert response.action == ActionEnum.ANSWER
    components.context_gate.evaluate.assert_awaited_once()
    components.evidence_gate.evaluate.assert_not_called()
    assert response.explanation.gate2.source == "ablation"


async def test_c3_still_soft_asks(components, store):
    components.context_gate.evaluate.return_value = ContextGateOutput(
        action="SOFT-ASK", rationale="r", missing_fields=["age"]
    )
    response = await run(build(components, store), ablation=AblationConfig.preset("C3"))
    assert response.action == ActionEnum.SOFT_ASK


async def test_c4_evidence_gate_only(components, store):
    components.context_gate.evaluate.return_value = ContextGateOutput(
        action="SOFT-ASK", rationale="r", missing_fields=["age"]
    )
    components.evidence_gate.evaluate.return_value = EvidenceGateOutput(
        action="ABSTAIN", evidence_sufficient=False, rationale="r"
    )
    response = await run(build(components, store), ablation=AblationConfig.preset("C4"))
    assert response.action == ActionEnum.ABSTAIN
    components.context_gate.evaluate.assert_not_called()
    assert response.explanation.gate1.source == "ablation"


async def test_c5_runs_both_gates(components, store):
    response = await run(
        build(
            components,
            store,
            EVIDENCE_GATE_CALIBRATION_ENABLED=True,
            GROUNDING_VERIFIER_ENABLED=True,
        ),
        ablation=AblationConfig.preset("C5"),
    )
    assert response.action == ActionEnum.ANSWER
    components.context_gate.evaluate.assert_awaited_once()
    components.evidence_gate.evaluate.assert_awaited_once()
    assert response.explanation.gate2.source == "llm"  # C5 forces V1/V2 off
    assert response.explanation.gate3 is None


@pytest.fixture
def calibration_module(monkeypatch):
    module = types.ModuleType("medbridge.ai.evidence_calibration")
    module.calibrated_evidence_decision = AsyncMock(
        return_value=SimpleNamespace(
            action="GENERALIZE",
            rationale="max rerank below threshold",
            votes=["ANSWER", "GENERALIZE", "GENERALIZE"],
            features={"max_rerank": -2.5},
            overridden=True,
        )
    )
    monkeypatch.setitem(sys.modules, "medbridge.ai.evidence_calibration", module)
    return module


async def test_v1_calibrated_evidence_gate(components, store, calibration_module):
    response = await run(build(components, store), ablation=AblationConfig.preset("V1"))
    assert response.action == ActionEnum.GENERALIZE
    gate2 = response.explanation.gate2
    assert gate2.source == "calibrated"
    assert gate2.votes == ["ANSWER", "GENERALIZE", "GENERALIZE"]
    assert gate2.features == {"max_rerank": -2.5, "overridden": True}
    call = calibration_module.calibrated_evidence_decision.await_args
    assert call.kwargs["votes"] == get_settings().EVIDENCE_GATE_VOTES
    assert call.args[3] == REDACTED


async def test_v1_follows_settings_when_ablation_is_none(components, store, calibration_module):
    response = await run(build(components, store, EVIDENCE_GATE_CALIBRATION_ENABLED=True))
    assert response.explanation.gate2.source == "calibrated"


async def test_v1_failure_falls_back_to_generalize(components, store, calibration_module):
    calibration_module.calibrated_evidence_decision.side_effect = RuntimeError("boom")
    response = await run(build(components, store), ablation=AblationConfig.preset("V1"))
    assert response.action == ActionEnum.GENERALIZE
    assert response.explanation.gate2.source == "fallback"


# ---------------------------------------------------------------------------
# 9. V2 — Gate 3 grounding verifier
# ---------------------------------------------------------------------------


def verifier_with(*results: SimpleNamespace) -> MagicMock:
    verifier = MagicMock()
    verifier.verify = AsyncMock(side_effect=list(results))
    return verifier


def grounding(
    fraction: float, cleaned: str = "", n_claims: int = 4, n_unsupported: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        unsupported_fraction=fraction,
        cleaned_text=cleaned,
        n_claims=n_claims,
        n_unsupported=n_unsupported,
        sentences=[],
    )


async def test_v2_supported_draft_uses_cleaned_text(components, store):
    verifier = verifier_with(grounding(0.25, cleaned="Keep checking at home [1].", n_unsupported=1))
    response = await run(
        build(components, store, verifier=verifier), ablation=AblationConfig.preset("V2")
    )
    assert response.action == ActionEnum.ANSWER
    assert response.response_text == "Keep checking at home [1]."
    assert [c.chunk_id for c in response.citations] == ["chunk-1"]
    gate3 = response.explanation.gate3
    assert gate3["n_claims"] == 4 and gate3["n_unsupported"] == 1
    assert gate3["action_change"] is None and gate3["removed_sentences_count"] == 1
    draft, used_chunks = verifier.verify.await_args.args
    assert draft.startswith("The usual target") and len(used_chunks) == 5


async def test_v2_unsupported_answer_regenerates_as_generalize(components, store):
    components.response_generator.generate.side_effect = [
        ResponseGeneratorOutput(response_text="Unsupported claim [1]."),
        ResponseGeneratorOutput(response_text="General guidance [2]."),
    ]
    verifier = verifier_with(
        grounding(0.9, n_unsupported=3), grounding(0.0, cleaned="General guidance [2].")
    )
    response = await run(
        build(components, store, verifier=verifier), ablation=AblationConfig.preset("V2")
    )
    assert response.action == ActionEnum.GENERALIZE
    assert response.response_text == "General guidance [1]."
    assert generator_calls(components)[1]["action"] == ActionEnum.GENERALIZE
    gate3 = response.explanation.gate3
    assert gate3["action_change"] == "ANSWER->GENERALIZE" and gate3["attempts"] == 2


async def test_v2_still_unsupported_abstains(components, store):
    components.response_generator.generate.side_effect = [
        ResponseGeneratorOutput(response_text="Claim [1]."),
        ResponseGeneratorOutput(response_text="Other claim [1]."),
    ]
    verifier = verifier_with(grounding(0.9), grounding(0.8))
    response = await run(
        build(components, store, verifier=verifier), ablation=AblationConfig.preset("V2")
    )
    assert response.action == ActionEnum.ABSTAIN
    assert response.response_text == ABSTAIN_TEMPLATE
    assert response.citations == []
    assert response.explanation.gate3["action_change"] == "ANSWER->ABSTAIN"


async def test_v2_unsupported_generalize_abstains(components, store):
    components.evidence_gate.evaluate.return_value = EvidenceGateOutput(
        action="GENERALIZE", evidence_sufficient=False, rationale="r"
    )
    verifier = verifier_with(grounding(0.6))
    response = await run(
        build(components, store, verifier=verifier), ablation=AblationConfig.preset("V2")
    )
    assert response.action == ActionEnum.ABSTAIN
    assert components.response_generator.generate.await_count == 1


async def test_v2_verifier_failure_keeps_draft(components, store):
    verifier = MagicMock()
    verifier.verify = AsyncMock(side_effect=RuntimeError("nli model missing"))
    response = await run(
        build(components, store, verifier=verifier), ablation=AblationConfig.preset("V2")
    )
    assert response.action == ActionEnum.ANSWER
    assert response.explanation.gate3 == {"status": "failed", "error": "RuntimeError"}


async def test_v2_not_run_for_templates(components, store):
    components.evidence_gate.evaluate.return_value = EvidenceGateOutput(
        action="ABSTAIN", evidence_sufficient=False, rationale="r"
    )
    verifier = verifier_with()
    await run(build(components, store, verifier=verifier), ablation=AblationConfig.preset("V2"))
    verifier.verify.assert_not_called()


# ---------------------------------------------------------------------------
# 10. Regenerate
# ---------------------------------------------------------------------------


async def test_regenerate_reuses_last_message_without_new_events(components, store):
    stored_id = uuid.uuid4()
    store.last_message = (stored_id, "What target should I aim for?")
    store.history = [
        ChatTurn(role="user", content="earlier question"),
        ChatTurn(role="assistant", content="earlier answer"),
        ChatTurn(role="user", content="What target should I aim for?"),
        ChatTurn(role="assistant", content="old answer"),
    ]
    response = await run(build(components, store), message="", regenerate=True)

    assert (
        components.context_extractor.extract.await_args.args[0] == "What target should I aim for?"
    )
    history = components.context_extractor.extract.await_args.kwargs["history"]
    assert [t.content for t in history] == ["earlier question", "earlier answer"]
    record = store.records[0]
    assert record.delta_events == [] and record.new_snapshot is None
    assert record.trace["regenerated"] is True
    assert response.explanation.regenerated is True
    assert response.user_message_id == stored_id
    assert response.message_id == store.ids[1]


async def test_regenerate_without_previous_message_is_safe(components, store):
    response = await run(build(components, store), message="", regenerate=True)
    assert response.response_text == PIPELINE_ERROR_TEMPLATE
    assert store.records == []


# ---------------------------------------------------------------------------
# 11. Persistence & never-throw (Non-Negotiable #7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario",
    ["answer", "soft_ask", "abstain", "escalate_fastpath", "small_talk", "loop_breaker"],
)
async def test_exactly_one_persist_call_per_turn(components, store, scenario):
    message = MESSAGE
    if scenario == "soft_ask":
        components.context_gate.evaluate.return_value = ContextGateOutput(
            action="SOFT-ASK", rationale="r", missing_fields=["age"]
        )
    elif scenario == "abstain":
        components.evidence_gate.evaluate.return_value = EvidenceGateOutput(
            action="ABSTAIN", evidence_sufficient=False, rationale="r"
        )
    elif scenario == "escalate_fastpath":
        components.emergency_classifier = SimpleNamespace(classify=lambda t: ActionEnum.ESCALATE)
    elif scenario == "small_talk":
        message = "thanks!"
    elif scenario == "loop_breaker":
        store.state = SessionState(snapshot={}, soft_ask_count=2)
        components.context_gate.evaluate.return_value = ContextGateOutput(
            action="SOFT-ASK", rationale="r", missing_fields=["age"]
        )
    store.persist_turn = AsyncMock(side_effect=store.persist_turn)  # type: ignore[method-assign]
    await run(build(components, store), message=message)
    store.persist_turn.assert_awaited_once()


async def test_persisted_record_contents(components, store):
    await run(build(components, store))
    record = store.records[0]
    assert record.session_id == SESSION_ID
    assert record.assistant_message.startswith("The usual target")
    assert record.gate_1_action == "PROCEED" and record.gate_1_rationale == "Context is sufficient."
    assert record.gate_2_rationale == "Evidence covers the target."
    assert record.evidence_chunk_ids == [f"chunk-{i}" for i in range(1, 6)]
    assert record.citations[0]["chunk_id"] == "chunk-2"
    assert len(record.delta_events) == 1
    assert record.new_snapshot["recent_bp_readings"] == [{"systolic": 150, "diastolic": 95}]
    assert record.latency_ms >= 0
    # the full explanation; the turn store splits PHI-bearing parts into an encrypted envelope
    assert record.trace["gate1"] == {
        "action": "PROCEED",
        "rationale": "Context is sufficient.",
        "missing_fields": [],
        "source": "llm",
        "votes": [],
        "features": {},
    }
    assert record.trace["gate2"]["action"] == "ANSWER"
    assert {t["stage"] for t in record.trace["timings"]} >= {"extraction", "generation"}


async def test_persist_failure_still_returns_answer(components, store):
    store.persist_error = RuntimeError("db down")
    db = MagicMock()
    db.rollback = AsyncMock()
    response = await build(components, store).process_message(SESSION_ID, MESSAGE, db=db)
    assert response.action == ActionEnum.ANSWER
    assert response.message_id is None
    db.rollback.assert_awaited_once()
    timing = {t.stage: t.status for t in response.explanation.timings}
    assert timing["persistence"] == "failed"


async def test_load_state_failure_degrades(components, store):
    store.load_error = ConnectionError("db down")
    response = await run(build(components, store))
    assert response.action == ActionEnum.ANSWER
    timing = {t.stage: t.status for t in response.explanation.timings}
    assert timing["load_state"] == "failed"


async def test_unexpected_error_returns_safe_response(components, store):
    with patch.object(PipelineOrchestrator, "_finalise_response_text", side_effect=KeyError("bug")):
        response = await run(build(components, store))
    assert response.action == ActionEnum.GENERALIZE
    assert response.response_text == PIPELINE_ERROR_TEMPLATE
    assert response.explanation.gate2.source == "fallback"
    assert "KeyError" in response.explanation.gate2.rationale
    assert store.records == []


async def test_event_callback_errors_are_swallowed(components, store):
    async def broken(event: PipelineEvent) -> None:
        raise RuntimeError("client went away")

    response = await run(build(components, store), on_event=broken)
    assert response.action == ActionEnum.ANSWER


async def test_stage_events_and_timings_cover_every_stage(components, store):
    recorder = EventRecorder()
    response = await run(build(components, store), on_event=recorder)
    assert all(e.type == "stage" and e.label for e in recorder.events)
    assert set(recorder.stages("skipped")) == {"verification"}
    stages = [t.stage for t in response.explanation.timings]
    assert set(stages) == {s.value for s in StageName}
    assert response.explanation.total_ms > 0


async def test_db_none_uses_own_session_and_commits(components, store):
    session = MagicMock()
    session.commit = AsyncMock()

    class Maker:
        def __call__(self) -> Maker:
            return self

        async def __aenter__(self) -> MagicMock:
            return session

        async def __aexit__(self, *exc: Any) -> None:
            return None

    with patch("medbridge.core.orchestrator.get_sessionmaker", return_value=Maker()):
        response = await build(components, store).process_message(SESSION_ID, MESSAGE)
    assert response.action == ActionEnum.ANSWER
    assert session.commit.await_count >= 1
    assert len(store.records) == 1


async def test_string_session_id_is_accepted(components, store):
    response = await build(components, store).process_message(
        str(SESSION_ID), MESSAGE, db=MagicMock()
    )
    assert response.session_id == SESSION_ID


async def test_llm_calls_recorded_in_trace(components, store):
    from medbridge.ai import llm_wrapper

    if not hasattr(llm_wrapper, "llm_call_log"):
        pytest.skip("llm_call_log not available yet")

    async def generate(**kwargs: Any) -> ResponseGeneratorOutput:
        llm_wrapper._call_log.get().append(  # type: ignore[attr-defined]
            llm_wrapper.LLMCallStat(call_name="response_generator", model="m", provider="ollama")
        )
        return ResponseGeneratorOutput(response_text="Aim below 130/80 [1].")

    components.response_generator.generate = AsyncMock(side_effect=generate)
    response = await run(build(components, store))
    assert [c["call_name"] for c in response.explanation.llm_calls] == ["response_generator"]
    assert store.records[0].trace["llm_calls"][0]["model"] == "m"


# ---------------------------------------------------------------------------
# 12. Warm-up and factory
# ---------------------------------------------------------------------------


async def test_warmup_reports_readiness(components, store):
    embedder = MagicMock()
    embedder.warmup = MagicMock(return_value={"dense": True, "sparse": True})
    components.hybrid_retriever.embedder = embedder
    components.hybrid_retriever.warmup = AsyncMock()
    components.reranker.warmup = AsyncMock(side_effect=RuntimeError("no model"))
    llm = MagicMock()
    llm.warmup = AsyncMock(return_value={"gemma3:4b": True})
    with patch("medbridge.core.orchestrator.get_llm_wrapper", return_value=llm):
        readiness = await build(components, store).warmup()
    assert readiness["embedder_dense"] and readiness["embedder_sparse"]
    assert readiness["retriever"] is True
    assert readiness["reranker"] is False
    assert readiness["llm"] is True
    assert "nli_verifier" not in readiness
    assert readiness["warmup_ms"] >= 0


async def test_warmup_never_raises_and_warms_verifier(components, store):
    components.hybrid_retriever.embedder = None
    components.hybrid_retriever.warmup = AsyncMock()
    components.reranker.warmup = AsyncMock()
    verifier = MagicMock(spec=["verify", "warmup"])
    verifier.warmup = MagicMock()
    llm = MagicMock(spec=[])  # no warmup attribute at all
    with patch("medbridge.core.orchestrator.get_llm_wrapper", return_value=llm):
        readiness = await build(
            components, store, verifier=verifier, GROUNDING_VERIFIER_ENABLED=True
        ).warmup()
    assert readiness["llm"] is False
    assert readiness["nli_verifier"] is True
    verifier.warmup.assert_called_once()


def test_get_orchestrator_returns_shared_instance():
    sentinel = MagicMock(spec=PipelineOrchestrator)
    with patch("medbridge.core.orchestrator._orchestrator_instance", sentinel):
        assert get_orchestrator() is sentinel


async def test_warmup_remote_provider_has_nothing_to_load(components, store):
    components.hybrid_retriever.embedder = None
    components.hybrid_retriever.warmup = AsyncMock()
    components.reranker.warmup = AsyncMock()
    llm = MagicMock()
    llm.warmup = AsyncMock(return_value={})  # remote provider: nothing to warm
    with patch("medbridge.core.orchestrator.get_llm_wrapper", return_value=llm):
        readiness = await build(components, store).warmup()
    assert readiness["llm"] is True and readiness["llm_models"] == {}
