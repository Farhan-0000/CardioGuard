"""
Integration Tests for REST Route Handlers (TASK-29, Module M-04).

- POST   /api/sessions                      → 201 + signed session token
- GET    /api/sessions/{session_id}/history → 200 (token) / 401 / 403 / 404 / 410 / 422
- DELETE /api/sessions/{session_id}         → 204 crypto-shred (token)
- POST   /api/sessions/{session_id}/messages → 200 / 404 / 422 (pipeline mocked)
- GET    /health                            → 200 / 503

Hermetic: the persistence layer and the orchestrator are mocked.
"""
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from medbridge.api.middleware.error_handler import register_error_handlers
from medbridge.api.routes.health import health_router
from medbridge.api.routes.messages import messages_router
from medbridge.api.routes.sessions import sessions_router
from medbridge.api.schemas.enums import ActionEnum
from medbridge.api.schemas.responses import (
    CitationResponse,
    HealthResponse,
    HistoryResponse,
    MessageResponse,
    SessionResponse,
)
from medbridge.core.orchestrator import PipelineOrchestrator, get_orchestrator
from medbridge.db.connection import get_session
from medbridge.db.models import Session as SessionModel
from medbridge.privacy.tokens import issue_session_token, verify_session_token
from medbridge.state import turn_store
from medbridge.state.session_manager import SessionNotFoundError
from medbridge.state.turn_store import HistoryRow, SessionState


def _auth(session_id: uuid.UUID) -> dict[str, str]:
    token, _ = issue_session_token(session_id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_db_session() -> MagicMock:
    """Mock AsyncSession that knows about Session rows (used by the messages route)."""
    sessions_store: dict[uuid.UUID, SessionModel] = {}
    db = MagicMock()
    db.info = {}
    db.get = AsyncMock(side_effect=lambda model, key: (
        sessions_store.get(key) if model is SessionModel else None
    ))
    db.add = MagicMock()
    db.execute = AsyncMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db._sessions_store = sessions_store
    return db


@pytest.fixture
def mock_orchestrator() -> AsyncMock:
    """Mock PipelineOrchestrator returning a valid MessageResponse."""
    orch = AsyncMock(spec=PipelineOrchestrator)

    async def mock_process(session_id: uuid.UUID, message: str, db: Any = None,
                           **_: Any) -> MessageResponse:
        return MessageResponse(
            session_id=session_id,
            response_text="Based on ACC/AHA guidelines [1], your BP target is <130/80 mmHg.",
            action=ActionEnum.ANSWER,
            citations=[CitationResponse(marker="[1]", chunk_id="aha_2025_s8_c1",
                                        source="AHA/ACC 2025", section="Section 8.1",
                                        excerpt="Target BP is <130/80 mmHg.")],
            soft_ask_count=0,
            timestamp=datetime.now(UTC),
        )

    orch.process_message.side_effect = mock_process
    return orch


@pytest.fixture
def test_client(mock_db_session: MagicMock, mock_orchestrator: AsyncMock) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    for router in (sessions_router, messages_router):
        app.include_router(router, prefix="/api")
    app.include_router(health_router)

    async def override_get_session() -> Any:
        yield mock_db_session

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    return TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# 1. POST /api/sessions
# ===========================================================================

def test_create_session_returns_signed_token(test_client: TestClient) -> None:
    sid = uuid.uuid4()
    with patch("medbridge.api.routes.sessions.create_session",
               AsyncMock(return_value=sid)) as create:
        response = test_client.post("/api/sessions")
    assert response.status_code == 201
    create.assert_awaited_once()
    parsed = SessionResponse.model_validate(response.json())
    assert parsed.session_id == sid
    assert parsed.session_token and verify_session_token(parsed.session_token, sid)
    assert parsed.expires_at is not None and parsed.expires_at > datetime.now(UTC)


# ===========================================================================
# 2. GET /api/sessions/{session_id}/history
# ===========================================================================

def test_get_history_empty_session(test_client: TestClient,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    sid = uuid.uuid4()
    monkeypatch.setattr(turn_store, "load_session_state", AsyncMock(return_value=SessionState()))
    monkeypatch.setattr(turn_store, "get_history", AsyncMock(return_value=[]))
    response = test_client.get(f"/api/sessions/{sid}/history", headers=_auth(sid))
    assert response.status_code == 200
    parsed = HistoryResponse.model_validate(response.json())
    assert parsed.session_id == sid and parsed.messages == []


def test_get_history_with_messages_ids_and_explanations(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid, user_id, assistant_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    now = datetime.now(UTC)
    rows = [
        HistoryRow(message_id=user_id, role="user", content="What is my BP target?",
                   action=None, citations=[], created_at=now),
        HistoryRow(message_id=assistant_id, role="assistant", content="Target is <130/80.",
                   action="ANSWER",
                   citations=[{"marker": "[1]", "chunk_id": "aha_c1", "source": "AHA 2025",
                               "section": "S1", "excerpt": "Guideline excerpt", "page": 12},
                              {"marker": "[2]", "source": "legacy row without fields"}],
                   created_at=now,
                   trace={"gate1": {"action": "PROCEED", "rationale": "enough context"},
                          "total_ms": 812.0}),
    ]
    monkeypatch.setattr(turn_store, "load_session_state", AsyncMock(return_value=SessionState()))
    monkeypatch.setattr(turn_store, "get_history", AsyncMock(return_value=rows))
    response = test_client.get(f"/api/sessions/{sid}/history", headers=_auth(sid))
    assert response.status_code == 200
    parsed = HistoryResponse.model_validate(response.json())
    assert [m.role for m in parsed.messages] == ["user", "assistant"]
    assert [m.message_id for m in parsed.messages] == [user_id, assistant_id]
    assistant = parsed.messages[1]
    assert assistant.action == ActionEnum.ANSWER
    assert [c.marker for c in assistant.citations] == ["[1]", "[2]"]
    assert assistant.citations[0].page == 12
    assert assistant.explanation is not None
    assert assistant.explanation.gate1.rationale == "enough context"
    assert assistant.explanation.total_ms == 812.0
    assert parsed.messages[0].explanation is None


def test_get_history_requires_a_valid_token(test_client: TestClient) -> None:
    sid = uuid.uuid4()
    assert test_client.get(f"/api/sessions/{sid}/history").status_code == 401
    other = test_client.get(f"/api/sessions/{sid}/history", headers=_auth(uuid.uuid4()))
    assert other.status_code == 403
    assert other.json()["error_code"] == "FORBIDDEN"


def test_get_history_missing_session_returns_404(test_client: TestClient,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    unknown_id = uuid.uuid4()
    monkeypatch.setattr(turn_store, "load_session_state",
                        AsyncMock(return_value=SessionState(exists=False)))
    response = test_client.get(f"/api/sessions/{unknown_id}/history", headers=_auth(unknown_id))
    assert response.status_code == 404
    data = response.json()
    assert data["error_code"] == "SESSION_NOT_FOUND"
    assert str(unknown_id) in data["message"]


def test_get_history_erased_session_returns_410(test_client: TestClient,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    sid = uuid.uuid4()
    monkeypatch.setattr(turn_store, "load_session_state",
                        AsyncMock(return_value=SessionState(erased=True)))
    response = test_client.get(f"/api/sessions/{sid}/history", headers=_auth(sid))
    assert response.status_code == 410
    assert response.json()["error_code"] == "SESSION_ERASED"


def test_get_history_invalid_uuid_returns_422(test_client: TestClient) -> None:
    response = test_client.get("/api/sessions/not-a-valid-uuid/history")
    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_ERROR"


# ===========================================================================
# 3. DELETE /api/sessions/{session_id}
# ===========================================================================

def test_delete_session_crypto_shreds(test_client: TestClient,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    sid = uuid.uuid4()
    erase = AsyncMock()
    monkeypatch.setattr(turn_store, "erase_session", erase)
    assert test_client.delete(f"/api/sessions/{sid}").status_code == 401
    response = test_client.delete(f"/api/sessions/{sid}", headers=_auth(sid))
    assert response.status_code == 204 and response.content == b""
    assert erase.await_args.args[1] == sid


def test_delete_unknown_session_returns_404(test_client: TestClient,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    sid = uuid.uuid4()
    monkeypatch.setattr(turn_store, "erase_session",
                        AsyncMock(side_effect=SessionNotFoundError(sid)))
    response = test_client.delete(f"/api/sessions/{sid}", headers=_auth(sid))
    assert response.status_code == 404


# ===========================================================================
# 4. POST /api/sessions/{session_id}/messages
# ===========================================================================

def test_send_message_success(test_client: TestClient, mock_db_session: MagicMock) -> None:
    sid = uuid.uuid4()
    mock_db_session._sessions_store[sid] = SessionModel(session_id=sid)
    response = test_client.post(f"/api/sessions/{sid}/messages",
                                json={"message": "My BP is 145/92 mmHg this morning."},
                                headers=_auth(sid))
    assert response.status_code == 200
    parsed = MessageResponse.model_validate(response.json())
    assert parsed.session_id == sid
    assert parsed.action == ActionEnum.ANSWER
    assert len(parsed.citations) == 1
    assert "ACC/AHA" in parsed.response_text


def test_send_message_missing_session_returns_404(test_client: TestClient) -> None:
    unknown_id = uuid.uuid4()
    response = test_client.post(f"/api/sessions/{unknown_id}/messages",
                                json={"message": "Hello, is this working?"},
                                headers=_auth(unknown_id))
    assert response.status_code == 404
    assert response.json()["error_code"] == "SESSION_NOT_FOUND"


def test_send_message_empty_message_returns_422(test_client: TestClient,
                                                mock_db_session: MagicMock) -> None:
    sid = uuid.uuid4()
    mock_db_session._sessions_store[sid] = SessionModel(session_id=sid)
    response = test_client.post(f"/api/sessions/{sid}/messages", json={"message": ""},
                                headers=_auth(sid))
    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_ERROR"


def test_send_message_exceeds_max_length_returns_422(test_client: TestClient,
                                                     mock_db_session: MagicMock) -> None:
    sid = uuid.uuid4()
    mock_db_session._sessions_store[sid] = SessionModel(session_id=sid)
    response = test_client.post(f"/api/sessions/{sid}/messages", json={"message": "a" * 2001},
                                headers=_auth(sid))
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"
    assert "a" * 100 not in response.text  # the offending input is not echoed back


# ===========================================================================
# 5. GET /health
# ===========================================================================

@patch("medbridge.api.routes.health.check_postgres_health", new_callable=AsyncMock)
@patch("medbridge.api.routes.health.check_qdrant_health", new_callable=AsyncMock)
def test_health_check_healthy(mock_qdrant: AsyncMock, mock_pg: AsyncMock,
                              test_client: TestClient) -> None:
    mock_pg.return_value = True
    mock_qdrant.return_value = True
    response = test_client.get("/health")
    assert response.status_code == 200
    parsed = HealthResponse.model_validate(response.json())
    assert parsed.status == "healthy"
    assert parsed.postgres_connected is True and parsed.qdrant_connected is True


@patch("medbridge.api.routes.health.check_postgres_health", new_callable=AsyncMock)
@patch("medbridge.api.routes.health.check_qdrant_health", new_callable=AsyncMock)
def test_health_check_postgres_failure_returns_503(mock_qdrant: AsyncMock, mock_pg: AsyncMock,
                                                   test_client: TestClient) -> None:
    mock_pg.return_value = False
    mock_qdrant.return_value = True
    response = test_client.get("/health")
    assert response.status_code == 503
    parsed = HealthResponse.model_validate(response.json())
    assert parsed.status == "unhealthy"
    assert parsed.postgres_connected is False and parsed.qdrant_connected is True


@patch("medbridge.api.routes.health.check_postgres_health", new_callable=AsyncMock)
@patch("medbridge.api.routes.health.check_qdrant_health", new_callable=AsyncMock)
def test_health_check_qdrant_failure_returns_503(mock_qdrant: AsyncMock, mock_pg: AsyncMock,
                                                 test_client: TestClient) -> None:
    mock_pg.return_value = True
    mock_qdrant.return_value = False
    response = test_client.get("/health")
    assert response.status_code == 503
    parsed = HealthResponse.model_validate(response.json())
    assert parsed.status == "unhealthy"
    assert parsed.postgres_connected is True and parsed.qdrant_connected is False
