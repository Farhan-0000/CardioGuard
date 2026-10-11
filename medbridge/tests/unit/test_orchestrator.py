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
from medbridge.nlp.query_expansion import expand_query
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
        self.counter_before: int | None = None
        self.counter_lookups: list[uuid.UUID] = []

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

    async def get_soft_ask_count_before(
        self, db: Any, session_id: uuid.UUID, user_message_id: uuid.UUID
    ) -> int | None:
        self.counter_lookups.append(user_message_id)
        return self.counter_before

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
    return make_components(chunks)


def make_components(chunks: list[RetrievedChunk]) -> SimpleNamespace:
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
        # Pin the Gate 3 thresholds the V2 tests exercise (the production defaults are the
        # dev calibration, tau = 1.0, which never regenerates or withdraws an answer).
        "GROUNDING_ENTAILMENT_THRESHOLD": 0.5,
        "GROUNDING_MAX_UNSUPPORTED_FRACTION": 0.34,
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
        action=ActionEnum.ESCALATE,
        matched_rules=["chest_pain", "bp_crisis"],
        negated=["syncope"],
        suppressed=["stroke_signs"],
        bp=(200, 125),
        evidence_spans=["my bp is 200/125", "my chest feels tight"],
    )
    components.emergency_classifier = SimpleNamespace(
        classify_detailed=MagicMock(return_value=detailed), classify=MagicMock()
    )
    response = await run(build(components, store))
    assert response.explanation.emergency == {
        "action": "ESCALATE",
        "matched_rules": ["chest_pain", "bp_crisis"],
        "negated": ["syncope"],
        "suppressed": ["stroke_signs"],
        "bp": [200, 125],
    }
    components.emergency_classifier.classify.assert_not_called()
    assert "chest_pain" in store.records[0].gate_2_rationale
    # evidence spans are fragments of the patient's text: never in the trace
    trace_json = json.dumps(store.records[0].trace)
    assert "chest feels tight" not in trace_json and "evidence_spans" not in trace_json


async def test_emergency_non_escalation_records_negated_and_suppressed(components, store):
    detailed = SimpleNamespace(
        action=None, matched_rules=[], negated=["chest_pain"], suppressed=[], bp=(138, 88),
        evidence_spans=[],
    )
    components.emergency_classifier = SimpleNamespace(
        classify_detailed=MagicMock(return_value=detailed)
    )
    response = await run(build(components, store))
    assert response.action == ActionEnum.ANSWER
    assert response.explanation.emergency == {
        "action": None, "matched_rules": [], "negated": ["chest_pain"], "suppressed": [],
        "bp": [138, 88],
    }


async def test_emergency_record_with_real_classifier(components, store):
    from medbridge.core import emergency_classifier

    components.emergency_classifier = emergency_classifier
    response = await run(
        build(components, store), message="my bp is 200/125 and my chest feels tight"
    )
    assert response.action == ActionEnum.ESCALATE
    emergency = response.explanation.emergency
    assert emergency["action"] == "ESCALATE" and "bp_crisis" in emergency["matched_rules"]
    assert emergency["bp"] == [200, 125]
    assert set(emergency) == {"action", "matched_rules", "negated", "suppressed", "bp"}


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
    expanded = expand_query(REDACTED)  # extractor failed: expanded redacted message
    assert components.hybrid_retriever.hybrid_search.await_args.args[0] == expanded
    assert components.evidence_gate.evaluate.await_args.kwargs["message"] == REDACTED
    assert components.evidence_gate.evaluate.await_args.kwargs["query"] == expanded
    assert generator_calls(components)[0]["message"] == REDACTED
    assert generator_calls(components)[0]["query"] == expanded
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
    query = components.hybrid_retriever.hybrid_search.await_args.args[0]
    assert query == expand_query(REDACTED)  # redacted message + clinical synonyms
    assert query.startswith(REDACTED)
    assert store.records[0].delta_events == []
    assert store.records[0].new_snapshot is None
    assert components.evidence_gate.evaluate.await_args.kwargs["message"] == REDACTED


async def test_extractor_failure_query_expansion_error_uses_message(components, store):
    components.context_extractor.extract.return_value = None
    with patch("medbridge.core.orchestrator.expand_query", side_effect=ValueError("bad")):
        response = await run(build(components, store))
    assert response.action == ActionEnum.ANSWER
    assert components.hybrid_retriever.hybrid_search.await_args.args[0] == REDACTED


async def test_gates_and_generator_get_message_and_search_query(components, store):
    """False premises and unknown drugs are judged on the patient's own words."""
    await run(build(components, store))
    gate_kwargs = components.evidence_gate.evaluate.await_args.kwargs
    assert gate_kwargs["message"] == REDACTED
    assert gate_kwargs["query"] == "blood pressure target amlodipine"
    generator_kwargs = generator_calls(components)[0]
    assert generator_kwargs["message"] == REDACTED
    assert generator_kwargs["query"] == "blood pressure target amlodipine"
    assert components.hybrid_retriever.hybrid_search.await_args.args[0] == (
        "blood pressure target amlodipine"
    )


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


@pytest.mark.parametrize("preset,mode", [("C1", "vanilla"), ("C2", "prompt_safety")])
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
    assert gate2.features == {"max_rerank": -2.5, "overridden": True, "failed_votes": 0.0}
    call = calibration_module.calibrated_evidence_decision.await_args
    assert call.kwargs["votes"] == get_settings().EVIDENCE_GATE_VOTES
    assert call.args[3] == "blood pressure target amlodipine"  # extractor search query
    assert call.kwargs["message"] == REDACTED  # the patient's own words


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


async def test_v2_production_tau_withdraws_fully_unsupported_answer(components, store):
    """tau = 1.0 (dev calibration) must still not deliver a draft whose every claim failed."""
    components.response_generator.generate.side_effect = [
        ResponseGeneratorOutput(response_text="Unsupported claim [1]."),
        ResponseGeneratorOutput(response_text="General guidance [2]. Another claim [1]."),
    ]
    verifier = verifier_with(
        grounding(1.0, cleaned="", n_claims=3, n_unsupported=3),
        grounding(0.5, cleaned="General guidance [2].", n_claims=2, n_unsupported=1),
    )
    response = await run(
        build(components, store, verifier=verifier, GROUNDING_MAX_UNSUPPORTED_FRACTION=1.0),
        ablation=AblationConfig.preset("V2"),
    )
    assert response.action == ActionEnum.GENERALIZE
    assert response.response_text == "General guidance [1]."
    assert response.explanation.gate3["action_change"] == "ANSWER->GENERALIZE"


async def test_v2_production_tau_abstains_when_regeneration_also_fully_unsupported(
    components, store
):
    components.response_generator.generate.side_effect = [
        ResponseGeneratorOutput(response_text="Claim [1]."),
        ResponseGeneratorOutput(response_text="Other claim [1]."),
    ]
    verifier = verifier_with(
        grounding(1.0, n_claims=2, n_unsupported=2), grounding(1.0, n_claims=1, n_unsupported=1)
    )
    response = await run(
        build(components, store, verifier=verifier, GROUNDING_MAX_UNSUPPORTED_FRACTION=1.0),
        ablation=AblationConfig.preset("V2"),
    )
    assert response.action == ActionEnum.ABSTAIN
    assert response.response_text == ABSTAIN_TEMPLATE


async def test_v2_production_tau_keeps_partially_supported_draft(components, store):
    verifier = verifier_with(
        grounding(0.75, cleaned="Keep checking at home [1].", n_claims=4, n_unsupported=3)
    )
    response = await run(
        build(components, store, verifier=verifier, GROUNDING_MAX_UNSUPPORTED_FRACTION=1.0),
        ablation=AblationConfig.preset("V2"),
    )
    assert response.action == ActionEnum.ANSWER
    assert response.response_text == "Keep checking at home [1]."
    assert components.response_generator.generate.await_count == 1


async def test_v2_unavailable_verifier_never_withdraws(components, store):
    unavailable = grounding(0.0, cleaned="", n_claims=2, n_unsupported=2)
    unavailable.available = False
    response = await run(
        build(components, store, verifier=verifier_with(unavailable),
              GROUNDING_MAX_UNSUPPORTED_FRACTION=1.0),
        ablation=AblationConfig.preset("V2"),
    )
    assert response.action == ActionEnum.ANSWER
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


# ---------------------------------------------------------------------------
# 13. Integration contract (round 2): Gate-2 features, Gate 3 availability,
#     NLI warm-up, HTML evidence and a PHI-free plaintext trace
# ---------------------------------------------------------------------------


async def test_plain_gate2_records_production_evidence_features(components, store, chunks):
    from medbridge.ai.evidence_calibration import evidence_features

    response = await run(build(components, store))
    gate2 = response.explanation.gate2
    assert gate2.source == "llm"
    reranked = await rerank_top("q", chunks)
    expected = evidence_features(REDACTED, reranked, {"recent_bp_readings": [
        {"systolic": 150, "diastolic": 95}
    ]})
    assert gate2.features == pytest.approx(expected)
    assert set(gate2.features) == {"max_rerank", "mean_top3_rerank", "entity_coverage", "n_chunks"}
    assert gate2.features["max_rerank"] == 5.0 and gate2.features["n_chunks"] == 5.0


async def test_gate2_fallback_still_records_features(components, store):
    components.evidence_gate.evaluate.return_value = None
    response = await run(build(components, store))
    assert response.explanation.gate2.source == "fallback"
    assert response.explanation.gate2.features["n_chunks"] == 5.0


async def test_gate2_features_with_no_evidence(components, store):
    components.hybrid_retriever.hybrid_search.return_value = []
    response = await run(build(components, store))
    features = response.explanation.gate2.features
    assert features["n_chunks"] == 0.0 and features["max_rerank"] < -1


async def test_gate2_features_not_recorded_for_bypassed_gates(components, store):
    components.context_gate.evaluate.return_value = ContextGateOutput(
        action="SOFT-ASK", rationale="Need meds.", missing_fields=["current_medications"]
    )
    response = await run(build(components, store))
    assert response.explanation.gate2 is None
    c1 = await run(build(components, FakeStore()), ablation=AblationConfig.preset("C1"))
    assert c1.explanation.gate2.source == "ablation" and c1.explanation.gate2.features == {}


async def test_gate2_features_survive_the_plaintext_trace_split(components, store):
    from medbridge.state.turn_store import split_trace

    await run(build(components, store))
    public, private = split_trace(store.records[0].trace)
    assert public["gate2"]["features"]["n_chunks"] == 5.0
    assert "rationale" not in public["gate2"]
    assert private["gate2"] == {"rationale": "Evidence covers the target."}


def unavailable_grounding(draft: str) -> SimpleNamespace:
    return SimpleNamespace(
        unsupported_fraction=0.0, cleaned_text=draft, n_claims=2, n_unsupported=0,
        available=False, sentences=[{"text": draft, "supported": True}],
    )


async def test_v2_unavailable_verifier_keeps_draft_and_is_recorded(components, store):
    draft = "The usual target is below 130/80 mmHg [2]. Keep checking at home [1]."
    verifier = verifier_with(unavailable_grounding(draft))
    response = await run(
        build(components, store, verifier=verifier), ablation=AblationConfig.preset("V2")
    )
    assert response.action == ActionEnum.ANSWER
    gate3 = response.explanation.gate3
    assert gate3["available"] is False and gate3["status"] == "unavailable"
    assert gate3["n_claims"] == 2 and gate3["action_change"] is None


async def test_v2_gate3_record_is_numeric_and_text_free(components, store):
    from medbridge.state.turn_store import split_trace

    verifier = verifier_with(SimpleNamespace(
        unsupported_fraction=0.25, cleaned_text="Keep checking at home [1].", n_claims=4,
        n_unsupported=1, available=True,
        sentences=[{"text": "The usual target is below 130/80 mmHg [2].", "supported": False}],
        unsupported_sentences=["The usual target is below 130/80 mmHg [2]."],
    ))
    await run(build(components, store, verifier=verifier), ablation=AblationConfig.preset("V2"))
    gate3 = store.records[0].trace["gate3"]
    assert gate3["available"] is True and gate3["status"] == "ok"
    assert "sentences" not in gate3 and "cleaned_text" not in gate3
    public, private = split_trace(store.records[0].trace)
    assert public["gate3"] == gate3  # every Gate 3 member is numeric/status → public
    assert "gate3" not in private
    assert "130/80" not in json.dumps(public["gate3"])


async def test_warmup_reflects_nli_verifier_result(components, store):
    components.hybrid_retriever.embedder = None
    components.hybrid_retriever.warmup = AsyncMock()
    components.reranker.warmup = AsyncMock()
    llm = MagicMock()
    llm.warmup = AsyncMock(return_value={})
    for result, expected in ((True, True), (False, False)):
        verifier = MagicMock(spec=["verify", "warmup"])
        verifier.warmup = AsyncMock(return_value=result)
        with patch("medbridge.core.orchestrator.get_llm_wrapper", return_value=llm):
            readiness = await build(
                components, store, verifier=verifier, GROUNDING_VERIFIER_ENABLED=True
            ).warmup()
        verifier.warmup.assert_awaited_once()
        assert readiness["nli_verifier"] is expected


async def test_warmup_uses_the_shared_grounding_verifier(components, store):
    components.hybrid_retriever.embedder = None
    components.hybrid_retriever.warmup = AsyncMock()
    components.reranker.warmup = AsyncMock()
    shared = MagicMock(spec=["verify", "warmup"])
    shared.warmup = AsyncMock(return_value=True)
    llm = MagicMock()
    llm.warmup = AsyncMock(return_value={})
    with (
        patch("medbridge.core.orchestrator.get_llm_wrapper", return_value=llm),
        patch("medbridge.ai.grounding_verifier.get_grounding_verifier", return_value=shared),
    ):
        readiness = await build(components, store, GROUNDING_VERIFIER_ENABLED=True).warmup()
    assert readiness["nli_verifier"] is True
    shared.warmup.assert_awaited_once()


async def test_html_evidence_has_no_physical_page(components, store, chunks):
    html_chunks = [
        c.model_copy(update={"source_format": "html", "section_title": "Treatment"})
        for c in chunks
    ]
    components.hybrid_retriever.hybrid_search.return_value = html_chunks
    response = await run(build(components, store))
    evidence = response.explanation.retrieval[0]
    assert evidence.page is None and evidence.section == "Treatment (section 11)"
    assert response.citations[0].page is None
    assert response.citations[0].section == "Treatment (section 12)"


async def test_metrics_grounding_counters_from_orchestrator_traces():
    """Gate 3 records → plaintext audit trace (split_trace) → /api/metrics counters."""
    from medbridge.api.routes.metrics import compute_metrics
    from medbridge.state.turn_store import split_trace

    def result(fraction: float, n_claims: int, n_unsupported: int, available: bool = True):
        return SimpleNamespace(
            unsupported_fraction=fraction, cleaned_text="Keep checking at home [1].",
            n_claims=n_claims, n_unsupported=n_unsupported, available=available,
            sentences=[{"text": "secret sentence 150/95"}],
        )

    failing = MagicMock()
    failing.verify = AsyncMock(side_effect=RuntimeError("nli"))
    verifiers = [
        verifier_with(result(0.25, 4, 1)),                     # kept, 1 sentence removed
        verifier_with(result(0.0, 2, 0, available=False)),     # NLI model unavailable
        verifier_with(result(0.9, 3, 3), result(0.0, 2, 0)),  # ANSWER -> GENERALIZE
        failing,                                               # verifier crashed
    ]
    traces = []
    for verifier in verifiers:
        store = FakeStore()
        orchestrator = build(make_components(make_chunks()), store, verifier=verifier)
        await run(orchestrator, ablation=AblationConfig.preset("V2"))
        public, _ = split_trace(store.records[0].trace)
        traces.append(public)

    assert all("secret sentence" not in json.dumps(t) for t in traces)
    rows = [("ANSWER", "PROCEED", "ANSWER", 100, t) for t in traces]
    assert compute_metrics(rows, [], window=10).grounding_verifier == {
        "verified_turns": 4,
        "failures": 1,
        "unavailable": 1,
        "action_changes": 1,
        "claims": 4 + 2 + 2,
        "unsupported_claims": 1,
        "removed_sentences": 1,
    }


# ---------------------------------------------------------------------------
# 12. Review fixes (stream F2): regenerate counter, per-session serialisation,
#     V1 all-votes-failed, retrieval outage, effective patient message
# ---------------------------------------------------------------------------

from medbridge.core.orchestrator import (  # noqa: E402
    FOLLOW_UP_LABEL,
    ORIGINAL_QUESTION_LABEL,
    SessionTurnLocks,
    compose_effective_message,
    soft_ask_chain,
)
from medbridge.retrieval.hybrid_retriever import RetrievalError  # noqa: E402
from medbridge.state.turn_store import RegenerateConflictError, split_trace  # noqa: E402


class StatefulStore(FakeStore):
    """Turn-store fake that keeps state across turns the way the real store does.

    The counter is written only when ``record.update_soft_ask_count``; the
    snapshot is folded from the persisted events; a regenerated answer is
    attached to the re-run user message; the first ``soft_ask_count_before``
    recorded for a user message is what ``get_soft_ask_count_before`` returns.
    """

    def __init__(self, soft_ask_count: int = 0) -> None:
        super().__init__(state=SessionState(snapshot={}, soft_ask_count=soft_ask_count))
        self.messages: list[tuple[uuid.UUID, ChatTurn]] = []
        self.counter_by_user: dict[uuid.UUID, int | None] = {}
        self.loads = 0

    async def load_session_state(self, db: Any, session_id: uuid.UUID) -> SessionState:
        self.loads += 1
        return SessionState(
            snapshot=dict(self.state.snapshot), soft_ask_count=self.state.soft_ask_count
        )

    async def get_recent_turns(self, db: Any, session_id: uuid.UUID, n: int) -> list[ChatTurn]:
        self.history_limits.append(n)
        return [turn for _, turn in self.messages][-n:]

    async def get_last_user_message(
        self, db: Any, session_id: uuid.UUID
    ) -> tuple[uuid.UUID, str] | None:
        for message_id, turn in reversed(self.messages):
            if turn.role == "user":
                return message_id, turn.content
        return None

    async def get_soft_ask_count_before(
        self, db: Any, session_id: uuid.UUID, user_message_id: uuid.UUID
    ) -> int | None:
        self.counter_lookups.append(user_message_id)
        return self.counter_by_user.get(user_message_id)

    async def persist_turn(self, db: Any, record: TurnRecord) -> tuple[uuid.UUID, uuid.UUID]:
        self.records.append(record)
        if record.delta_events:
            self.state.snapshot = StateProjector().apply_events(
                self.state.snapshot, list(record.delta_events)
            )
        if record.update_soft_ask_count:
            self.state.soft_ask_count = record.soft_ask_count
        if record.regenerated_user_message_id is not None:
            user_id = record.regenerated_user_message_id
        else:
            user_id = uuid.uuid4()
            self.messages.append((user_id, ChatTurn(role="user", content=record.user_message)))
        assistant_id = uuid.uuid4()
        self.messages.append(
            (
                assistant_id,
                ChatTurn(
                    role="assistant",
                    content=record.assistant_message,
                    action=str(record.final_action),
                ),
            )
        )
        self.counter_by_user.setdefault(user_id, record.trace.get("soft_ask_count_before"))
        return user_id, assistant_id


def always_soft_ask(components: SimpleNamespace) -> None:
    components.context_gate.evaluate.return_value = ContextGateOutput(
        action="SOFT-ASK", rationale="need the medicine name", missing_fields=["medication"]
    )


# --- invariants-1 / persistence-db-2 / web-ui-3: regenerate is idempotent -------------


async def test_regenerating_a_soft_ask_never_counts_twice(components):
    always_soft_ask(components)
    store = StatefulStore()
    orchestrator = build(components, store)

    first = await run(orchestrator, "Should I double my BP tablet?")
    assert first.action == ActionEnum.SOFT_ASK and first.soft_ask_count == 1
    assert first.explanation.soft_ask_count_before == 0

    for _ in range(3):  # any number of regenerates: same start value, same result
        regenerated = await run(orchestrator, "", regenerate=True)
        assert regenerated.action == ActionEnum.SOFT_ASK
        assert regenerated.soft_ask_count == 1
        assert regenerated.explanation.soft_ask_count_before == 0
        assert regenerated.explanation.gate2 is None  # no loop-breaker
        assert store.state.soft_ask_count == 1
        assert regenerated.user_message_id == first.user_message_id

    # The patient's next real reply is the 2nd consecutive SOFT-ASK, not a loop-breaker.
    follow_up = await run(orchestrator, "It's the white one, I forget the name")
    assert follow_up.action == ActionEnum.SOFT_ASK and follow_up.soft_ask_count == 2


async def test_regenerating_a_loop_breaker_answer_breaks_the_loop_again(components):
    always_soft_ask(components)
    store = StatefulStore(soft_ask_count=2)
    orchestrator = build(components, store)

    broken = await run(orchestrator, "I really don't know, sorry")
    assert broken.action == ActionEnum.GENERALIZE
    assert broken.explanation.gate2.source == "loop_breaker"
    assert broken.explanation.soft_ask_count_before == 2
    assert store.state.soft_ask_count == 0

    for _ in range(2):
        again = await run(orchestrator, "", regenerate=True)
        assert again.action == ActionEnum.GENERALIZE  # never a 3rd clarifying question
        assert again.explanation.gate2.source == "loop_breaker"
        assert again.response_text.startswith(LOOP_BREAKER_PREFIX)
        assert again.explanation.soft_ask_count_before == 2
        assert store.state.soft_ask_count == 0
    assert set(store.counter_lookups) == {broken.user_message_id}


async def test_regenerate_without_recorded_counter_keeps_stored_value(components):
    """A store without the lookup (or no recorded value) falls back to the stored counter."""
    always_soft_ask(components)
    store = FakeStore(
        state=SessionState(snapshot={}, soft_ask_count=1),
        last_message=(uuid.uuid4(), "Can I take ibuprofen?"),
    )
    response = await run(build(components, store), "", regenerate=True)
    assert store.counter_lookups == [store.last_message[0]]
    assert response.explanation.soft_ask_count_before == 1
    assert response.soft_ask_count == 2


async def test_regenerate_passes_the_rerun_message_id_and_survives_a_conflict(components, store):
    stored_id = uuid.uuid4()
    store.last_message = (stored_id, "What target should I aim for?")
    store.persist_error = RegenerateConflictError(SESSION_ID)
    db = MagicMock()
    db.rollback = AsyncMock()
    response = await build(components, store).process_message(
        SESSION_ID, "", db=db, regenerate=True
    )
    assert response.action == ActionEnum.ANSWER  # the patient still gets the answer
    assert response.message_id is None
    timing = {t.stage: t.status for t in response.explanation.timings}
    assert timing["persistence"] == "failed"
    db.rollback.assert_awaited_once()

    store.persist_error = None
    await build(components, store).process_message(SESSION_ID, "", db=db, regenerate=True)
    assert store.records[-1].regenerated_user_message_id == stored_id


async def test_counter_left_untouched_when_gate1_did_not_run(components, store):
    await run(build(components, store), message="thanks!")  # small talk
    components.emergency_classifier = SimpleNamespace(classify=lambda t: ActionEnum.ESCALATE)
    await run(build(components, store))  # emergency fast-path
    assert [r.update_soft_ask_count for r in store.records] == [False, False]

    components.emergency_classifier = SimpleNamespace(classify=lambda t: None)
    await run(build(components, store))
    await run(build(components, store), ablation=AblationConfig.preset("C1"))
    assert [r.update_soft_ask_count for r in store.records[2:]] == [True, True]


# --- invariants-2 / persistence-db-1 / web-ui-2: per-session serialisation ------------


def gated_extractor(
    components: SimpleNamespace, release: asyncio.Event
) -> tuple[list[str], asyncio.Event]:
    """Extractor whose FIRST call signals ``started`` and waits for ``release`` (a slow LLM)."""
    seen: list[str] = []
    started = asyncio.Event()

    async def extract(message: str, snapshot: dict[str, Any], history: Any = None) -> Any:
        seen.append(message)
        if len(seen) == 1:
            started.set()
            await release.wait()
        return ExtractorOutput(search_query="bp target", raw_intent="target")

    components.context_extractor.extract = AsyncMock(side_effect=extract)
    return seen, started


async def test_concurrent_turns_on_one_session_are_serialised(components):
    always_soft_ask(components)
    store = StatefulStore()
    orchestrator = build(components, store)
    release = asyncio.Event()
    seen, started = gated_extractor(components, release)

    first = asyncio.create_task(run(orchestrator, "Should I double my amlodipine?"))
    await started.wait()
    second = asyncio.create_task(run(orchestrator, "I'm 62 by the way"))
    for _ in range(20):
        await asyncio.sleep(0)
    assert store.loads == 1 and len(seen) == 1  # the 2nd turn waits; no stale state load

    release.set()
    a, b = await asyncio.gather(first, second)
    assert (a.soft_ask_count, b.soft_ask_count) == (1, 2)  # gated on the 1st turn's counter
    assert b.explanation.soft_ask_count_before == 1
    assert store.state.soft_ask_count == 2
    assert orchestrator._turn_locks._locks == {}  # locks are released and dropped


async def test_emergency_is_never_queued_behind_a_running_turn(components):
    store = StatefulStore(soft_ask_count=1)
    release = asyncio.Event()
    _, started = gated_extractor(components, release)
    components.emergency_classifier = SimpleNamespace(
        classify=lambda t: ActionEnum.ESCALATE if "chest pain" in t else None
    )
    orchestrator = build(components, store)

    slow = asyncio.create_task(run(orchestrator, "What target should I aim for?"))
    await started.wait()
    urgent = await asyncio.wait_for(
        run(orchestrator, "crushing chest pain and I can't breathe"), timeout=2
    )
    assert urgent.action == ActionEnum.ESCALATE
    assert not slow.done()  # answered while the other turn was still running
    release.set()
    await slow
    assert store.records[0].update_soft_ask_count is False  # the emergency turn
    assert store.state.soft_ask_count == 0  # the slow turn (PROCEED) set it


async def test_session_turn_locks_are_per_session():
    locks = SessionTurnLocks()
    a, b = uuid.uuid4(), uuid.uuid4()
    async with locks.hold(a):
        async with locks.hold(b) as waited:  # another session never waits
            assert waited < 50
    assert locks._locks == {} and locks._users == {}


# --- invariants-3: V1 with every vote failed is the spec fallback --------------------


async def test_v1_all_votes_failed_is_a_fallback_not_a_calibrated_decision(
    components, store, calibration_module
):
    calibration_module.calibrated_evidence_decision.return_value = SimpleNamespace(
        action="GENERALIZE",
        rationale="Evidence Gate failed on every vote; spec fallback to GENERALIZE.",
        votes=[],
        features={"max_rerank": 1.0},
        overridden=False,
        failed_votes=3,
    )
    calibration_module.evidence_features = lambda query, chunks, snapshot: {"max_rerank": 2.0}
    response = await run(build(components, store), ablation=AblationConfig.preset("V1"))
    gate2 = response.explanation.gate2
    assert response.action == ActionEnum.GENERALIZE
    assert gate2.source == "fallback" and gate2.votes == []
    assert gate2.features["failed_votes"] == 3.0
    assert "max_rerank" in gate2.features  # production evidence features still recorded
    timing = {t.stage: t.status for t in response.explanation.timings}
    assert timing["evidence_gate"] == "failed"
    assert store.records[0].gate_2_action == "GENERALIZE"


async def test_v1_partial_vote_loss_is_recorded_and_counted(
    components, store, calibration_module
):
    from medbridge.api.routes.metrics import compute_metrics

    calibration_module.calibrated_evidence_decision.return_value = SimpleNamespace(
        action="ANSWER",
        rationale="r",
        votes=["ANSWER", "ANSWER"],
        features={"max_rerank": 4.0},
        overridden=False,
        failed_votes=1,
    )
    partial = await run(build(components, store), ablation=AblationConfig.preset("V1"))
    assert partial.explanation.gate2.source == "calibrated"
    assert partial.explanation.gate2.features["failed_votes"] == 1.0

    calibration_module.calibrated_evidence_decision.return_value = SimpleNamespace(
        action="GENERALIZE", rationale="r", votes=[], features={}, overridden=False,
        failed_votes=3,
    )
    await run(build(components, store), ablation=AblationConfig.preset("V1"))
    rows = [("ANSWER", "PROCEED", "ANSWER", 10, split_trace(r.trace)[0]) for r in store.records]
    fallbacks = compute_metrics(rows, [], window=10).fallbacks
    assert fallbacks["evidence_gate"] == 1 and fallbacks["evidence_gate_partial"] == 1


# --- ablation-study-validity-2: a retrieval outage is visible ------------------------


async def test_retrieval_outage_is_marked_failed_and_gate2_still_decides(components, store):
    components.hybrid_retriever.hybrid_search.side_effect = RetrievalError("qdrant down")
    response = await run(build(components, store))
    exp = response.explanation
    assert exp.retrieval_status == "failed"
    assert {t.stage: t.status for t in exp.timings}["retrieval"] == "failed"
    components.evidence_gate.evaluate.assert_awaited_once()  # Gate 2 decides (spec)
    assert components.evidence_gate.evaluate.await_args.kwargs["chunks"] == []
    public, _ = split_trace(store.records[0].trace)
    assert public["retrieval_status"] == "failed"  # readable by metrics/evaluation


async def test_retrieval_status_distinguishes_empty_skipped_and_discarded(
    components, store, chunks
):
    components.hybrid_retriever.hybrid_search.return_value = []
    empty = await run(build(components, store))
    assert empty.explanation.retrieval_status == "empty"

    always_soft_ask(components)
    skipped = await run(build(components, store))
    assert skipped.explanation.retrieval_status == "skipped"
    discarded = await run(build(components, store, SPECULATIVE_RETRIEVAL=True))
    assert discarded.explanation.retrieval_status == "discarded"

    components.context_gate.evaluate.return_value = ContextGateOutput(
        action="PROCEED", rationale="ok"
    )
    components.hybrid_retriever.hybrid_search.return_value = chunks
    ok = await run(build(components, store))
    assert ok.explanation.retrieval_status == "ok"


# --- dev error analysis: multi-turn follow-ups keep the original question ------------

STATIN_QUESTION = "Should I take a cholesterol tablet even though my cholesterol is normal?"
STATIN_SOFT_ASK = "Have you ever had a heart attack or a stroke?"
STROKE_FOLLOW_UP = "I had a stroke two years ago"


async def test_follow_up_after_soft_ask_reaches_gates_with_the_original_question(
    components, store
):
    store.history = [
        ChatTurn(role="user", content=STATIN_QUESTION),
        ChatTurn(role="assistant", content=STATIN_SOFT_ASK, action="SOFT-ASK"),
    ]
    raw_seen: list[str] = []
    components.emergency_classifier = SimpleNamespace(classify=lambda t: raw_seen.append(t))
    response = await run(build(components, store), STROKE_FOLLOW_UP)

    effective = (
        f"{ORIGINAL_QUESTION_LABEL} {STATIN_QUESTION}\n{FOLLOW_UP_LABEL} {STROKE_FOLLOW_UP}"
    )
    assert components.context_gate.evaluate.await_args.args[1] == effective  # Gate 1
    assert components.evidence_gate.evaluate.await_args.kwargs["message"] == effective  # Gate 2
    assert generator_calls(components)[0]["message"] == effective
    # the emergency classifier and the extractor see only the current message
    assert raw_seen == [STROKE_FOLLOW_UP]
    assert components.context_extractor.extract.await_args.args[0] == STROKE_FOLLOW_UP
    record = store.records[0]
    assert record.user_message == STROKE_FOLLOW_UP  # stored as said
    assert response.explanation.effective_question == {"used": True, "earlier_messages": 1}
    public, private = split_trace(record.trace)
    assert public["effective_question"] == {"used": True, "earlier_messages": 1}
    assert "cholesterol" not in json.dumps(public) and "stroke" not in json.dumps(public)


async def test_three_turn_loop_breaker_sequence_reaches_generator_with_original_question(
    components,
):
    always_soft_ask(components)
    store = StatefulStore()
    orchestrator = build(components, store)
    turns = [
        "what bp is normal for me?",
        "I don't have a recent reading",
        "nope not gonna check it. just answer in general",
    ]
    responses = [await run(orchestrator, text) for text in turns]

    assert [r.action for r in responses] == [
        ActionEnum.SOFT_ASK, ActionEnum.SOFT_ASK, ActionEnum.GENERALIZE
    ]
    assert responses[2].explanation.gate2.source == "loop_breaker"
    final_call = generator_calls(components)[-1]
    assert final_call["action"] == ActionEnum.GENERALIZE
    assert final_call["message"] == (
        f"{ORIGINAL_QUESTION_LABEL} {turns[0]}\n"
        f"{FOLLOW_UP_LABEL} {turns[1]}\n"
        f"{FOLLOW_UP_LABEL} {turns[2]}"
    )
    assert components.context_gate.evaluate.await_args.args[1] == final_call["message"]
    assert generator_calls(components)[0]["message"] == turns[0]  # 1st turn: nothing to carry
    assert responses[2].explanation.effective_question["earlier_messages"] == 2


async def test_no_effective_message_after_an_answer(components, store):
    store.history = [
        ChatTurn(role="user", content=STATIN_QUESTION),
        ChatTurn(role="assistant", content="Statins are ...", action="ANSWER"),
    ]
    response = await run(build(components, store))
    assert components.context_gate.evaluate.await_args.args[1] == REDACTED
    assert response.explanation.effective_question is None


async def test_effective_message_redacts_earlier_text_and_falls_back_on_extractor_failure(
    components, store
):
    store.history = [
        ChatTurn(role="user", content="I'm John Smith, can I take aspirin daily?"),
        ChatTurn(role="assistant", content="Do you take blood thinners?", action="SOFT-ASK"),
    ]
    components.context_extractor.extract.return_value = None
    await run(build(components, store), "no, only amlodipine")
    message = components.evidence_gate.evaluate.await_args.kwargs["message"]
    assert "John Smith" not in message and "[NAME]" in message
    query = components.hybrid_retriever.hybrid_search.await_args.args[0]
    assert query == expand_query("I'm [NAME], can I take aspirin daily? no, only amlodipine")


def test_soft_ask_chain_pairs_latest_reply_and_respects_the_limit():
    def turn(role: str, content: str, action: str | None = None) -> ChatTurn:
        return ChatTurn(role=role, content=content, action=action)

    regenerated_soft_ask = [
        turn("user", "q1"), turn("assistant", "a", "ANSWER"), turn("assistant", "b", "SOFT-ASK")
    ]
    assert soft_ask_chain(regenerated_soft_ask, 2) == ["q1"]  # the latest reply counts
    regenerated_answer = [
        turn("user", "q1"), turn("assistant", "a", "SOFT-ASK"), turn("assistant", "b", "ANSWER")
    ]
    assert soft_ask_chain(regenerated_answer, 2) == []
    long_chain = [
        turn("user", "q0"), turn("assistant", "x", "ANSWER"),
        turn("user", "q1"), turn("assistant", "x", "SOFT-ASK"),
        turn("user", "q2"), turn("assistant", "x", "SOFT_ASK"),
        turn("user", "q3"), turn("assistant", "x", ActionEnum.SOFT_ASK),
    ]
    assert soft_ask_chain(long_chain, 2) == ["q2", "q3"]
    assert soft_ask_chain(long_chain, 5) == ["q1", "q2", "q3"]
    assert soft_ask_chain([turn("assistant", "orphan", "SOFT-ASK")], 2) == []
    assert compose_effective_message(["q"], "f") == (
        f"{ORIGINAL_QUESTION_LABEL} q\n{FOLLOW_UP_LABEL} f"
    )
