"""
Unit tests for TASK-12: API Request & Response Pydantic Models.

Tests validation boundaries, JSON serialization, and schema compliance
for all models in api/schemas/requests.py and api/schemas/responses.py.
"""
import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from medbridge.api.schemas.enums import ActionEnum
from medbridge.api.schemas.requests import MessageRequest
from medbridge.api.schemas.responses import (
    CitationResponse,
    ErrorResponse,
    HealthResponse,
    HistoryMessage,
    HistoryResponse,
    MessageResponse,
    SessionResponse,
)


# ---------------------------------------------------------------------------
# MessageRequest Tests
# ---------------------------------------------------------------------------

class TestMessageRequest:
    """Validate MessageRequest boundaries per Constitution §8.5."""

    def test_valid_message(self) -> None:
        req = MessageRequest(message="What is my blood pressure target?")
        assert req.message == "What is my blood pressure target?"

    def test_min_length_single_char(self) -> None:
        req = MessageRequest(message="A")
        assert req.message == "A"

    def test_max_length_2000_chars(self) -> None:
        msg = "A" * 2000
        req = MessageRequest(message=msg)
        assert len(req.message) == 2000

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            MessageRequest(message="")
        errors = exc_info.value.errors()
        assert any(e["type"] == "string_too_short" for e in errors)

    def test_exceeds_2000_chars_rejected(self) -> None:
        msg = "A" * 2001
        with pytest.raises(ValidationError) as exc_info:
            MessageRequest(message=msg)
        errors = exc_info.value.errors()
        assert any(e["type"] == "string_too_long" for e in errors)

    def test_missing_message_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MessageRequest()  # type: ignore[call-arg]

    def test_json_serialization(self) -> None:
        req = MessageRequest(message="Test question")
        data = json.loads(req.model_dump_json())
        assert data == {"message": "Test question"}

    def test_json_deserialization(self) -> None:
        raw = '{"message": "What should I do about my BP?"}'
        req = MessageRequest.model_validate_json(raw)
        assert req.message == "What should I do about my BP?"


# ---------------------------------------------------------------------------
# CitationResponse Tests
# ---------------------------------------------------------------------------

class TestCitationResponse:
    """Verify CitationResponse contains all required fields."""

    @pytest.fixture
    def sample_citation(self) -> CitationResponse:
        return CitationResponse(
            marker="[1]",
            chunk_id="aha_2025_s8_c3",
            source="AHA/ACC 2025",
            section="Section 8.2",
            excerpt="For patients with hypertension and diabetes...",
        )

    def test_all_fields_present(self, sample_citation: CitationResponse) -> None:
        assert sample_citation.marker == "[1]"
        assert sample_citation.chunk_id == "aha_2025_s8_c3"
        assert sample_citation.source == "AHA/ACC 2025"
        assert sample_citation.section == "Section 8.2"
        assert sample_citation.excerpt == "For patients with hypertension and diabetes..."

    def test_json_round_trip(self, sample_citation: CitationResponse) -> None:
        json_str = sample_citation.model_dump_json()
        restored = CitationResponse.model_validate_json(json_str)
        assert restored == sample_citation

    def test_missing_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CitationResponse(
                marker="[1]",
                chunk_id="aha_2025_s8_c3",
                # missing: source, section, excerpt
            )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# MessageResponse Tests
# ---------------------------------------------------------------------------

class TestMessageResponse:
    """Verify MessageResponse schema and serialization."""

    @pytest.fixture
    def sample_response(self) -> MessageResponse:
        return MessageResponse(
            session_id=uuid4(),
            response_text="Based on current guidelines...",
            action=ActionEnum.ANSWER,
            citations=[
                CitationResponse(
                    marker="[1]",
                    chunk_id="aha_s8_c3",
                    source="AHA/ACC 2025",
                    section="Section 8.2",
                    excerpt="Target BP for patients...",
                )
            ],
            soft_ask_count=0,
            timestamp=datetime.now(timezone.utc),
        )

    def test_action_is_enum(self, sample_response: MessageResponse) -> None:
        assert isinstance(sample_response.action, ActionEnum)
        assert sample_response.action == ActionEnum.ANSWER

    def test_citations_list(self, sample_response: MessageResponse) -> None:
        assert len(sample_response.citations) == 1
        assert isinstance(sample_response.citations[0], CitationResponse)

    def test_empty_citations_default(self) -> None:
        resp = MessageResponse(
            session_id=uuid4(),
            response_text="General guidance...",
            action=ActionEnum.GENERALIZE,
            soft_ask_count=0,
            timestamp=datetime.now(timezone.utc),
        )
        assert resp.citations == []

    def test_json_round_trip(self, sample_response: MessageResponse) -> None:
        json_str = sample_response.model_dump_json()
        data = json.loads(json_str)
        assert "session_id" in data
        assert data["action"] == "ANSWER"
        assert len(data["citations"]) == 1

    def test_all_actions_accepted(self) -> None:
        for action in ActionEnum:
            resp = MessageResponse(
                session_id=uuid4(),
                response_text="text",
                action=action,
                soft_ask_count=0,
                timestamp=datetime.now(timezone.utc),
            )
            assert resp.action == action


# ---------------------------------------------------------------------------
# SessionResponse Tests
# ---------------------------------------------------------------------------

class TestSessionResponse:
    """Verify SessionResponse schema for POST /api/sessions."""

    def test_valid_session_response(self) -> None:
        sid = uuid4()
        now = datetime.now(timezone.utc)
        resp = SessionResponse(session_id=sid, created_at=now)
        assert resp.session_id == sid
        assert resp.created_at == now

    def test_json_serialization(self) -> None:
        resp = SessionResponse(
            session_id=uuid4(),
            created_at=datetime.now(timezone.utc),
        )
        data = json.loads(resp.model_dump_json())
        assert "session_id" in data
        assert "created_at" in data


# ---------------------------------------------------------------------------
# HistoryResponse Tests
# ---------------------------------------------------------------------------

class TestHistoryResponse:
    """Verify HistoryResponse schema for GET /api/sessions/{id}/history."""

    def test_empty_history(self) -> None:
        resp = HistoryResponse(session_id=uuid4())
        assert resp.messages == []

    def test_history_with_messages(self) -> None:
        now = datetime.now(timezone.utc)
        resp = HistoryResponse(
            session_id=uuid4(),
            messages=[
                HistoryMessage(
                    role="user",
                    content="What is hypertension?",
                    timestamp=now,
                ),
                HistoryMessage(
                    role="assistant",
                    content="Hypertension is high blood pressure...",
                    action=ActionEnum.ANSWER,
                    citations=[
                        CitationResponse(
                            marker="[1]",
                            chunk_id="medline_c1",
                            source="MedlinePlus",
                            section="Overview",
                            excerpt="High blood pressure is...",
                        )
                    ],
                    timestamp=now,
                ),
            ],
        )
        assert len(resp.messages) == 2
        assert resp.messages[0].role == "user"
        assert resp.messages[0].action is None
        assert resp.messages[1].role == "assistant"
        assert resp.messages[1].action == ActionEnum.ANSWER
        assert len(resp.messages[1].citations) == 1

    def test_json_round_trip(self) -> None:
        now = datetime.now(timezone.utc)
        resp = HistoryResponse(
            session_id=uuid4(),
            messages=[
                HistoryMessage(role="user", content="Hello", timestamp=now),
            ],
        )
        json_str = resp.model_dump_json()
        restored = HistoryResponse.model_validate_json(json_str)
        assert len(restored.messages) == 1
        assert restored.messages[0].content == "Hello"


# ---------------------------------------------------------------------------
# HealthResponse Tests
# ---------------------------------------------------------------------------

class TestHealthResponse:
    """Verify HealthResponse schema for GET /health."""

    def test_healthy_response(self) -> None:
        resp = HealthResponse(
            status="healthy",
            postgres_connected=True,
            qdrant_connected=True,
            timestamp=datetime.now(timezone.utc),
        )
        assert resp.status == "healthy"
        assert resp.postgres_connected is True
        assert resp.qdrant_connected is True

    def test_unhealthy_response(self) -> None:
        resp = HealthResponse(
            status="unhealthy",
            postgres_connected=True,
            qdrant_connected=False,
            timestamp=datetime.now(timezone.utc),
        )
        assert resp.status == "unhealthy"
        assert resp.qdrant_connected is False

    def test_json_serialization(self) -> None:
        resp = HealthResponse(
            status="healthy",
            postgres_connected=True,
            qdrant_connected=True,
            timestamp=datetime.now(timezone.utc),
        )
        data = json.loads(resp.model_dump_json())
        assert data["status"] == "healthy"
        assert data["postgres_connected"] is True


# ---------------------------------------------------------------------------
# ErrorResponse Tests
# ---------------------------------------------------------------------------

class TestErrorResponse:
    """Verify ErrorResponse schema (NON-NEGOTIABLE #7 support)."""

    def test_valid_error_response(self) -> None:
        resp = ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An unexpected error occurred.",
            safe_fallback="I'm unable to process your request right now. "
                          "Please consult your healthcare provider.",
        )
        assert resp.error_code == "INTERNAL_ERROR"
        assert resp.safe_fallback.startswith("I'm unable")

    def test_json_serialization(self) -> None:
        resp = ErrorResponse(
            error_code="SESSION_NOT_FOUND",
            message="Session does not exist.",
            safe_fallback="Please start a new conversation.",
        )
        data = json.loads(resp.model_dump_json())
        assert data["error_code"] == "SESSION_NOT_FOUND"
        assert "safe_fallback" in data

    def test_missing_safe_fallback_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ErrorResponse(
                error_code="ERROR",
                message="Something went wrong.",
                # missing: safe_fallback
            )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# AI (LLM output) schemas
# ---------------------------------------------------------------------------

from medbridge.ai.schemas import (
    MISSING_FIELD_VOCABULARY,
    ContextGateOutput,
    DeltaEvent,
    EvidenceGateOutput,
    ExtractorOutput,
    ResponseGeneratorOutput,
)
from medbridge.api.schemas.enums import EventTypeEnum


class TestAISchemas:
    """LLM output schemas: lenient validation, strict LLM-facing JSON schema."""

    def test_extractor_defaults(self) -> None:
        output = ExtractorOutput(search_query="q", raw_intent="i")
        assert output.delta_events == []

    def test_extractor_requires_query_and_intent(self) -> None:
        with pytest.raises(ValidationError):
            ExtractorOutput.model_validate({"delta_events": []})

    def test_delta_event_json_schema_restricts_types(self) -> None:
        schema = DeltaEvent.model_json_schema()
        assert schema["properties"]["event_type"]["enum"] == [e.value for e in EventTypeEnum]
        # Python-side validation stays lenient (filtered later by the extractor).
        assert DeltaEvent(event_type="UNKNOWN").event_type == "UNKNOWN"

    def test_context_gate_vocabulary(self) -> None:
        assert len(MISSING_FIELD_VOCABULARY) == 12
        schema = ContextGateOutput.model_json_schema()
        assert schema["properties"]["missing_fields"]["items"]["enum"] == list(MISSING_FIELD_VOCABULARY)

    def test_evidence_gate_optional_fields(self) -> None:
        output = EvidenceGateOutput.model_validate({"action": "ABSTAIN", "evidence_sufficient": False})
        assert output.rationale == "" and output.evidence_used == []

    def test_generator_citations_not_requested_from_llm(self) -> None:
        schema = ResponseGeneratorOutput.model_json_schema()
        assert schema["required"] == ["response_text"]
        assert schema["properties"]["citations"]["llm_omit"] is True
        with pytest.raises(ValidationError):
            ResponseGeneratorOutput(response_text="")
