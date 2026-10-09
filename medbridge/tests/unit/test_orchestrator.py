"""
Unit tests for TASK-23: Pipeline Orchestrator (Module M-08).

Tests all 5 routing paths, emergency fast-path bypass, loop-breaker threshold (soft_ask_count >= 2),
LLM fallback guarantees, and audit persistence using mocked pipeline stages and database session.
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from medbridge.ai.schemas.context_gate import ContextGateOutput
from medbridge.ai.schemas.evidence_gate import EvidenceGateOutput
from medbridge.ai.schemas.extractor import DeltaEvent, ExtractorOutput
from medbridge.ai.schemas.response_generator import (
    GeneratedCitation,
    ResponseGeneratorOutput,
)
from medbridge.api.schemas.enums import ActionEnum, EventTypeEnum
from medbridge.core.orchestrator import PipelineOrchestrator, get_orchestrator
from medbridge.core.templates import (
    ABSTAIN_TEMPLATE,
    ESCALATE_TEMPLATE,
    LOOP_BREAKER_PREFIX,
)
from medbridge.db.models import (
    AuditLog,
    FinalActionEnum,
    Gate1ActionEnum,
    Gate2ActionEnum,
    MessageHistory,
)
from medbridge.retrieval.hybrid_retriever import RetrievedChunk


@pytest.fixture
def mock_db_session():
    """Mock async SQLAlchemy session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def sample_session_id():
    return uuid.uuid4()


@pytest.fixture
def mock_components():
    """Fixture providing mocks for all 8 orchestrator sub-components."""
    emergency_classifier = MagicMock()
    emergency_classifier.classify.return_value = None  # Default: non-emergency

    context_extractor = MagicMock()
    context_extractor.extract = AsyncMock(
        return_value=ExtractorOutput(
            delta_events=[
                DeltaEvent(
                    event_type=EventTypeEnum.BP_READING,
                    payload={"systolic": 140, "diastolic": 90},
                )
            ],
            search_query="hypertension treatment guidelines",
            raw_intent="blood pressure management",
        )
    )

    state_projector = MagicMock()
    state_projector.persist_and_project = AsyncMock(
        return_value={"recent_bp_readings": [{"systolic": 140, "diastolic": 90}]}
    )

    context_gate = MagicMock()
    context_gate.evaluate = AsyncMock(
        return_value=ContextGateOutput(
            action="PROCEED",
            missing_fields=[],
            rationale="Sufficient clinical context for guidance.",
        )
    )

    hybrid_retriever = MagicMock()
    hybrid_retriever.hybrid_search = AsyncMock(
        return_value=[
            RetrievedChunk(
                chunk_id="aha_chunk_1",
                chunk_text="Thiazide diuretics are recommended first-line therapy for Stage 1 hypertension.",
                guideline_id="AHA_ACC_2025",
                section_title="First-Line Therapy",
                page_number=12,
                source_url="https://guidelines.acc.org/htn2025",
                score=0.95,
            )
        ]
    )

    reranker = MagicMock()
    reranker.rerank = AsyncMock(
        return_value=[
            ("Thiazide diuretics are recommended first-line therapy for Stage 1 hypertension.", 0.95)
        ]
    )

    evidence_gate = MagicMock()
    evidence_gate.evaluate = AsyncMock(
        return_value=EvidenceGateOutput(
            action="ANSWER",
            evidence_sufficient=True,
            rationale="Guidelines directly provide first-line pharmacotherapy recommendations.",
        )
    )

    response_generator = MagicMock()
    response_generator.generate = AsyncMock(
        return_value=ResponseGeneratorOutput(
            response_text="First-line therapy includes thiazide diuretics [1].",
            citations=[
                GeneratedCitation(
                    marker="[1]",
                    chunk_id="aha_chunk_1",
                    source="AHA_ACC_2025",
                    section="First-Line Therapy",
                    excerpt="Thiazide diuretics are recommended first-line therapy.",
                )
            ],
        )
    )

    return {
        "emergency_classifier": emergency_classifier,
        "context_extractor": context_extractor,
        "state_projector": state_projector,
        "context_gate": context_gate,
        "hybrid_retriever": hybrid_retriever,
        "reranker": reranker,
        "evidence_gate": evidence_gate,
        "response_generator": response_generator,
    }


@pytest.fixture
def orchestrator(mock_components):
    return PipelineOrchestrator(
        emergency_classifier=mock_components["emergency_classifier"],
        context_extractor=mock_components["context_extractor"],
        state_projector=mock_components["state_projector"],
        context_gate=mock_components["context_gate"],
        hybrid_retriever=mock_components["hybrid_retriever"],
        reranker=mock_components["reranker"],
        evidence_gate=mock_components["evidence_gate"],
        response_generator=mock_components["response_generator"],
    )


# ---------------------------------------------------------------------------
# 1. Emergency Fast-Path Bypass Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emergency_fast_path_bypass(orchestrator, mock_components, mock_db_session, sample_session_id):
    """Emergency match triggers immediate ESCALATE bypass without calling LLMs or retrieval."""
    mock_components["emergency_classifier"].classify.return_value = ActionEnum.ESCALATE

    with patch("medbridge.core.orchestrator.load_session_state", new=AsyncMock(return_value={"snapshot": {}, "soft_ask_count": 0})), \
         patch("medbridge.core.orchestrator.save_message", new=AsyncMock(return_value=uuid.uuid4())) as mock_save:

        response = await orchestrator.process_message(
            session_id=sample_session_id,
            message="My BP is 210/125 and I have severe crushing chest pain",
            db=mock_db_session,
        )

        assert response.action == ActionEnum.ESCALATE
        assert response.response_text == ESCALATE_TEMPLATE
        assert response.citations == []

        # Downstream stages must NOT have been called
        mock_components["context_extractor"].extract.assert_not_called()
        mock_components["context_gate"].evaluate.assert_not_called()
        mock_components["hybrid_retriever"].hybrid_search.assert_not_called()
        mock_components["evidence_gate"].evaluate.assert_not_called()
        mock_components["response_generator"].generate.assert_not_called()

        # Messages saved to history (user + assistant)
        assert mock_save.call_count == 2

        # Audit log written to DB
        assert mock_db_session.add.called
        added_audit = next(call[0][0] for call in mock_db_session.add.call_args_list if isinstance(call[0][0], AuditLog))
        assert added_audit.final_action == FinalActionEnum.ESCALATE
        assert added_audit.gate_2_action == Gate2ActionEnum.ESCALATE


# ---------------------------------------------------------------------------
# 2. SOFT-ASK & Loop-Breaker Threshold Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_soft_ask_short_circuit_when_count_under_2(orchestrator, mock_components, mock_db_session, sample_session_id):
    """Context Gate SOFT-ASK with soft_ask_count < 2 increments count and returns clarifying question."""
    mock_components["context_gate"].evaluate.return_value = ContextGateOutput(
        action="SOFT-ASK",
        missing_fields=["current_medications"],
        rationale="Need to know current medications.",
    )
    mock_components["response_generator"].generate.return_value = ResponseGeneratorOutput(
        response_text="What medications are you currently taking?",
        citations=[],
    )

    with patch("medbridge.core.orchestrator.load_session_state", new=AsyncMock(return_value={"snapshot": {}, "soft_ask_count": 0})), \
         patch("medbridge.core.orchestrator.increment_soft_ask_count", new=AsyncMock(return_value=1)) as mock_inc, \
         patch("medbridge.core.orchestrator.save_message", new=AsyncMock(return_value=uuid.uuid4())):

        response = await orchestrator.process_message(
            session_id=sample_session_id,
            message="What should my BP goal be?",
            db=mock_db_session,
        )

        assert response.action == ActionEnum.SOFT_ASK
        assert response.soft_ask_count == 1
        assert "medications" in response.response_text
        mock_inc.assert_called_once_with(mock_db_session, sample_session_id)

        # Retrieval and Evidence Gate must be bypassed on SOFT-ASK short-circuit
        mock_components["hybrid_retriever"].hybrid_search.assert_not_called()
        mock_components["evidence_gate"].evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_soft_ask_loop_breaker_forces_generalize(orchestrator, mock_components, mock_db_session, sample_session_id):
    """When soft_ask_count >= 2, loop-breaker triggers, forcing GENERALIZE with prefix."""
    mock_components["context_gate"].evaluate.return_value = ContextGateOutput(
        action="SOFT-ASK",
        missing_fields=["age", "medications"],
        rationale="Context still insufficient.",
    )
    mock_components["response_generator"].generate.return_value = ResponseGeneratorOutput(
        response_text="General hypertension guidelines recommend lifestyle modifications and regular monitoring [1].",
        citations=[
            GeneratedCitation(
                marker="[1]",
                chunk_id="c1",
                source="AHA",
                section="General",
                excerpt="Lifestyle modification guidance",
            )
        ],
    )

    with patch("medbridge.core.orchestrator.load_session_state", new=AsyncMock(return_value={"snapshot": {}, "soft_ask_count": 2})), \
         patch("medbridge.core.orchestrator.increment_soft_ask_count", new=AsyncMock()) as mock_inc, \
         patch("medbridge.core.orchestrator.reset_soft_ask_count", new=AsyncMock()) as mock_reset, \
         patch("medbridge.core.orchestrator.save_message", new=AsyncMock(return_value=uuid.uuid4())):

        response = await orchestrator.process_message(
            session_id=sample_session_id,
            message="I still don't know my medications, just tell me what to do",
            db=mock_db_session,
        )

        # Must not increment soft_ask_count again
        mock_inc.assert_not_called()

        # Must reset soft_ask_count after loop-breaker fires
        mock_reset.assert_called_once_with(mock_db_session, sample_session_id)
        assert response.soft_ask_count == 0

        # Must force GENERALIZE
        assert response.action == ActionEnum.GENERALIZE
        assert LOOP_BREAKER_PREFIX in response.response_text

        # Must have proceeded to Retrieval and Generator
        mock_components["hybrid_retriever"].hybrid_search.assert_called_once()
        mock_components["response_generator"].generate.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Standard Routing Paths (ANSWER, GENERALIZE, ABSTAIN, ESCALATE)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_proceed_to_answer_path(orchestrator, mock_components, mock_db_session, sample_session_id):
    """PROCEED -> Retrieval -> Evidence Gate ANSWER -> Response Generator with citations."""
    mock_components["context_gate"].evaluate.return_value = ContextGateOutput(
        action="PROCEED",
        missing_fields=[],
        rationale="Context sufficient.",
    )
    mock_components["evidence_gate"].evaluate.return_value = EvidenceGateOutput(
        action="ANSWER",
        evidence_sufficient=True,
        rationale="Direct evidence match.",
    )

    with patch("medbridge.core.orchestrator.load_session_state", new=AsyncMock(return_value={"snapshot": {}, "soft_ask_count": 1})), \
         patch("medbridge.core.orchestrator.reset_soft_ask_count", new=AsyncMock()) as mock_reset, \
         patch("medbridge.core.orchestrator.save_message", new=AsyncMock(return_value=uuid.uuid4())):

        response = await orchestrator.process_message(
            session_id=sample_session_id,
            message="What is the first-line medication for Stage 1 HTN?",
            db=mock_db_session,
        )

        assert response.action == ActionEnum.ANSWER
        assert len(response.citations) == 1
        assert response.citations[0].marker == "[1]"
        assert response.soft_ask_count == 0
        mock_reset.assert_called_once_with(mock_db_session, sample_session_id)


@pytest.mark.asyncio
async def test_proceed_to_generalize_path(orchestrator, mock_components, mock_db_session, sample_session_id):
    """PROCEED -> Evidence Gate GENERALIZE -> Response Generator with general guidance."""
    mock_components["evidence_gate"].evaluate.return_value = EvidenceGateOutput(
        action="GENERALIZE",
        evidence_sufficient=False,
        rationale="Evidence covers topic generally.",
    )
    mock_components["response_generator"].generate.return_value = ResponseGeneratorOutput(
        response_text="In general, calcium channel blockers lower blood pressure.",
        citations=[],
    )

    with patch("medbridge.core.orchestrator.load_session_state", new=AsyncMock(return_value={"snapshot": {}, "soft_ask_count": 0})), \
         patch("medbridge.core.orchestrator.reset_soft_ask_count", new=AsyncMock()), \
         patch("medbridge.core.orchestrator.save_message", new=AsyncMock(return_value=uuid.uuid4())):

        response = await orchestrator.process_message(
            session_id=sample_session_id,
            message="How do blood pressure pills work?",
            db=mock_db_session,
        )

        assert response.action == ActionEnum.GENERALIZE
        assert "calcium channel blockers" in response.response_text


@pytest.mark.asyncio
async def test_proceed_to_abstain_path(orchestrator, mock_components, mock_db_session, sample_session_id):
    """PROCEED -> Evidence Gate ABSTAIN -> Deterministic ABSTAIN template (generator bypassed)."""
    mock_components["evidence_gate"].evaluate.return_value = EvidenceGateOutput(
        action="ABSTAIN",
        evidence_sufficient=False,
        rationale="Query is outside cardiovascular guideline scope.",
    )

    with patch("medbridge.core.orchestrator.load_session_state", new=AsyncMock(return_value={"snapshot": {}, "soft_ask_count": 0})), \
         patch("medbridge.core.orchestrator.reset_soft_ask_count", new=AsyncMock()), \
         patch("medbridge.core.orchestrator.save_message", new=AsyncMock(return_value=uuid.uuid4())):

        response = await orchestrator.process_message(
            session_id=sample_session_id,
            message="What is the chemotherapy protocol for glioblastoma?",
            db=mock_db_session,
        )

        assert response.action == ActionEnum.ABSTAIN
        assert response.response_text == ABSTAIN_TEMPLATE
        assert response.citations == []
        mock_components["response_generator"].generate.assert_not_called()


@pytest.mark.asyncio
async def test_proceed_to_escalate_gate_path(orchestrator, mock_components, mock_db_session, sample_session_id):
    """PROCEED -> Evidence Gate ESCALATE -> Deterministic ESCALATE template (generator bypassed)."""
    mock_components["evidence_gate"].evaluate.return_value = EvidenceGateOutput(
        action="ESCALATE",
        evidence_sufficient=False,
        rationale="Acute clinical deterioration detected.",
    )

    with patch("medbridge.core.orchestrator.load_session_state", new=AsyncMock(return_value={"snapshot": {}, "soft_ask_count": 0})), \
         patch("medbridge.core.orchestrator.reset_soft_ask_count", new=AsyncMock()), \
         patch("medbridge.core.orchestrator.save_message", new=AsyncMock(return_value=uuid.uuid4())):

        response = await orchestrator.process_message(
            session_id=sample_session_id,
            message="I have acute shortness of breath after taking my pills",
            db=mock_db_session,
        )

        assert response.action == ActionEnum.ESCALATE
        assert response.response_text == ESCALATE_TEMPLATE
        mock_components["response_generator"].generate.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Fallback Guarantees on Individual Stage Failures (§3.9)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extractor_failure_fallback(orchestrator, mock_components, mock_db_session, sample_session_id):
    """Context Extractor failure falls back to raw message without raising."""
    mock_components["context_extractor"].extract.return_value = None

    with patch("medbridge.core.orchestrator.load_session_state", new=AsyncMock(return_value={"snapshot": {}, "soft_ask_count": 0})), \
         patch("medbridge.core.orchestrator.reset_soft_ask_count", new=AsyncMock()), \
         patch("medbridge.core.orchestrator.save_message", new=AsyncMock(return_value=uuid.uuid4())):

        response = await orchestrator.process_message(
            session_id=sample_session_id,
            message="Hypertension query",
            db=mock_db_session,
        )

        assert response.action == ActionEnum.ANSWER
        # State projector was called with empty delta events
        mock_components["state_projector"].persist_and_project.assert_called_once_with(
            mock_db_session, sample_session_id, [], {}
        )


@pytest.mark.asyncio
async def test_context_gate_failure_fallback(orchestrator, mock_components, mock_db_session, sample_session_id):
    """Context Gate failure defaults to PROCEED per §3.9."""
    mock_components["context_gate"].evaluate.return_value = None

    with patch("medbridge.core.orchestrator.load_session_state", new=AsyncMock(return_value={"snapshot": {}, "soft_ask_count": 0})), \
         patch("medbridge.core.orchestrator.reset_soft_ask_count", new=AsyncMock()), \
         patch("medbridge.core.orchestrator.save_message", new=AsyncMock(return_value=uuid.uuid4())):

        response = await orchestrator.process_message(
            session_id=sample_session_id,
            message="Hypertension query",
            db=mock_db_session,
        )

        # Successfully proceeded through retrieval to answer
        assert response.action == ActionEnum.ANSWER
        mock_components["hybrid_retriever"].hybrid_search.assert_called_once()


@pytest.mark.asyncio
async def test_evidence_gate_failure_fallback(orchestrator, mock_components, mock_db_session, sample_session_id):
    """Evidence Gate failure defaults to GENERALIZE per §3.9."""
    mock_components["evidence_gate"].evaluate.return_value = None
    mock_components["response_generator"].generate.return_value = ResponseGeneratorOutput(
        response_text="General hypertension information.",
        citations=[],
    )

    with patch("medbridge.core.orchestrator.load_session_state", new=AsyncMock(return_value={"snapshot": {}, "soft_ask_count": 0})), \
         patch("medbridge.core.orchestrator.reset_soft_ask_count", new=AsyncMock()), \
         patch("medbridge.core.orchestrator.save_message", new=AsyncMock(return_value=uuid.uuid4())):

        response = await orchestrator.process_message(
            session_id=sample_session_id,
            message="General query",
            db=mock_db_session,
        )

        assert response.action == ActionEnum.GENERALIZE


@pytest.mark.asyncio
async def test_response_generator_failure_fallback(orchestrator, mock_components, mock_db_session, sample_session_id):
    """Response Generator failure returns safe pre-vetted GENERALIZE template."""
    mock_components["response_generator"].generate.return_value = None

    with patch("medbridge.core.orchestrator.load_session_state", new=AsyncMock(return_value={"snapshot": {}, "soft_ask_count": 0})), \
         patch("medbridge.core.orchestrator.reset_soft_ask_count", new=AsyncMock()), \
         patch("medbridge.core.orchestrator.save_message", new=AsyncMock(return_value=uuid.uuid4())):

        response = await orchestrator.process_message(
            session_id=sample_session_id,
            message="Any query",
            db=mock_db_session,
        )

        assert response.action == ActionEnum.GENERALIZE
        assert "consult your primary care physician" in response.response_text.lower()


@pytest.mark.asyncio
async def test_guaranteed_non_throwing_on_critical_exception(orchestrator, mock_components, mock_db_session, sample_session_id):
    """Critical exception anywhere in pipeline resolves to a safe MessageResponse (Non-Negotiable #7)."""
    # Cause an unexpected fatal error in state loading
    with patch("medbridge.core.orchestrator.load_session_state", side_effect=RuntimeError("Fatal database crash")):
        response = await orchestrator.process_message(
            session_id=sample_session_id,
            message="Any query",
            db=mock_db_session,
        )

        assert response.action == ActionEnum.GENERALIZE
        assert "unexpected error" in response.response_text.lower()
        assert response.session_id == sample_session_id


@pytest.mark.asyncio
async def test_duplicate_chunk_text_deduplication_and_matching(
    orchestrator, mock_components, mock_db_session, sample_session_id
):
    """Verifies candidate chunks with duplicate text are deduplicated before reranking and matched via dict lookup."""
    chunk1 = RetrievedChunk(
        chunk_id="chunk-1",
        chunk_text="Duplicate guideline text.",
        guideline_id="AHA_ACC_2025",
        section_title="Section A",
        score=0.8,
    )
    chunk2 = RetrievedChunk(
        chunk_id="chunk-2",
        chunk_text="Duplicate guideline text.",
        guideline_id="AHA_ACC_2025",
        section_title="Section B",
        score=0.7,
    )
    mock_components["hybrid_retriever"].hybrid_search = AsyncMock(return_value=[chunk1, chunk2])
    mock_components["reranker"].rerank = AsyncMock(
        return_value=[("Duplicate guideline text.", 0.95)]
    )

    with patch("medbridge.core.orchestrator.load_session_state", new=AsyncMock(return_value={"snapshot": {}, "soft_ask_count": 0})), \
         patch("medbridge.core.orchestrator.reset_soft_ask_count", new=AsyncMock()), \
         patch("medbridge.core.orchestrator.save_message", new=AsyncMock(return_value=uuid.uuid4())):

        response = await orchestrator.process_message(
            session_id=sample_session_id,
            message="What is the guideline?",
            db=mock_db_session,
        )

        assert response.action == ActionEnum.ANSWER
        mock_components["reranker"].rerank.assert_called_once()
        _, call_chunk_texts = mock_components["reranker"].rerank.call_args[0][:2]
        assert len(call_chunk_texts) == 1
        assert call_chunk_texts == ["Duplicate guideline text."]


def test_factory_function():
    orch = get_orchestrator()
    assert isinstance(orch, PipelineOrchestrator)

