"""
Route tests for message, SSE streaming and regenerate endpoints
(medbridge/api/routes/messages.py, finalization §3.7).

The orchestrator, the database session factory and the turn store are
replaced with fakes, so these tests need no database, Qdrant or LLM.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from medbridge.api.middleware.error_handler import register_error_handlers
from medbridge.api.routes import messages as messages_module
from medbridge.api.routes.messages import chunk_tokens, messages_router, sse_event, stream_turn
from medbridge.api.schemas.enums import ActionEnum
from medbridge.api.schemas.responses import CitationResponse, MessageResponse
from medbridge.core.orchestrator import get_orchestrator
from medbridge.core.pipeline_types import STAGE_LABELS, PipelineEvent, StageName
from medbridge.core.templates import PIPELINE_ERROR_TEMPLATE
from medbridge.db.connection import get_session
from medbridge.db.models import Session as SessionModel
from medbridge.privacy.deps import require_session_access

KNOWN = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
MESSAGE_ID = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000002")
ANSWER_TEXT = "Most adults should aim for a blood pressure below 130/80 mmHg [1]. Check at home."


class FakeOrchestrator:
    """Records calls, emits two stage events and returns a fixed answer."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    async def process_message(
        self,
        session_id: uuid.UUID,
        message: str,
        db: Any = None,
        *,
        on_event: Any = None,
        ablation: Any = None,
        regenerate: bool = False,
    ) -> MessageResponse:
        self.calls.append(
            {"session_id": session_id, "message": message, "db": db, "regenerate": regenerate}
        )
        if on_event is not None:
            for status in ("start", "end"):
                await on_event(
                    PipelineEvent(
                        type="stage",
                        stage=StageName.EMERGENCY,
                        status=status,  # type: ignore[arg-type]
                        label=STAGE_LABELS[StageName.EMERGENCY],
                    )
                )
        if self.error is not None:
            raise self.error
        return MessageResponse(
            session_id=session_id,
            response_text=ANSWER_TEXT,
            action=ActionEnum.ANSWER,
            citations=[
                CitationResponse(
                    marker="[1]", chunk_id="c1", source="AHA/ACC 2025", section="S", excerpt="E"
                )
            ],
            soft_ask_count=0,
            timestamp=datetime.now(UTC),
            message_id=MESSAGE_ID,
            redaction_count=2,
        )


ERASED = uuid.UUID("cccccccc-0000-0000-0000-000000000003")


class FakeDB:
    def __init__(self, known: set[uuid.UUID], erased: set[uuid.UUID] | None = None) -> None:
        self.known = known
        self.erased = erased or set()
        self.commit = AsyncMock()
        self.rollback = AsyncMock()

    async def get(self, model: Any, key: uuid.UUID) -> Any:
        if model is not SessionModel:
            return None
        if key in self.erased:
            return SessionModel(session_id=key, erased_at=datetime.now(UTC))
        return SessionModel(session_id=key) if key in self.known else None


class FakeMaker:
    """Stands in for ``get_sessionmaker()`` — every session is the same FakeDB."""

    def __init__(self, db: FakeDB) -> None:
        self.db = db
        self.opened = 0

    def __call__(self) -> FakeMaker:
        return self

    async def __aenter__(self) -> FakeDB:
        self.opened += 1
        return self.db

    async def __aexit__(self, *exc: Any) -> None:
        return None


@pytest.fixture
def db() -> FakeDB:
    return FakeDB({KNOWN}, erased={ERASED})


@pytest.fixture
def orchestrator() -> FakeOrchestrator:
    return FakeOrchestrator()


@pytest.fixture
def store(monkeypatch) -> SimpleNamespace:
    fake = SimpleNamespace(
        get_last_user_message=AsyncMock(return_value=(uuid.uuid4(), "previous question"))
    )
    monkeypatch.setattr(messages_module, "turn_store", fake)
    return fake


@pytest.fixture
def maker(monkeypatch, db: FakeDB) -> FakeMaker:
    fake = FakeMaker(db)
    monkeypatch.setattr(messages_module, "get_sessionmaker", lambda: fake)
    monkeypatch.setattr(messages_module, "TOKEN_DELAY_SECONDS", 0)
    return fake


@pytest.fixture
def client(db, orchestrator, store, maker) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(messages_router, prefix="/api")

    async def override_session():
        yield db

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app.dependency_overrides[require_session_access] = lambda: None
    return TestClient(app, raise_server_exceptions=False)


def parse_sse(body: str) -> list[tuple[str, Any]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


# ---------------------------------------------------------------------------
# POST /messages (JSON)
# ---------------------------------------------------------------------------


def test_send_message_returns_message_response(client, orchestrator, db):
    response = client.post(f"/api/sessions/{KNOWN}/messages", json={"message": "BP target?"})
    assert response.status_code == 200
    parsed = MessageResponse.model_validate(response.json())
    assert parsed.action == ActionEnum.ANSWER and parsed.message_id == MESSAGE_ID
    assert orchestrator.calls == [
        {"session_id": KNOWN, "message": "BP target?", "db": db, "regenerate": False}
    ]


def test_send_message_unknown_session_404(client):
    response = client.post(f"/api/sessions/{uuid.uuid4()}/messages", json={"message": "hi"})
    assert response.status_code == 404
    assert response.json()["error_code"] == "SESSION_NOT_FOUND"


@pytest.mark.parametrize("message", ["", "   \n\t", "a" * 2001], ids=["empty", "blank", "too-long"])
def test_send_message_validation_422(client, message):
    response = client.post(f"/api/sessions/{KNOWN}/messages", json={"message": message})
    assert response.status_code == 422


def test_send_message_requires_session_token(db, orchestrator, store, maker):
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(messages_router, prefix="/api")

    async def override_session():
        yield db

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    test_client = TestClient(app, raise_server_exceptions=False)
    for path in ("messages", "messages/stream", "regenerate"):
        response = test_client.post(f"/api/sessions/{KNOWN}/{path}", json={"message": "hi"})
        assert response.status_code == 401, path
    assert orchestrator.calls == []


# ---------------------------------------------------------------------------
# POST /messages/stream (SSE)
# ---------------------------------------------------------------------------


def test_stream_emits_stage_meta_tokens_done(client, orchestrator, db, maker):
    response = client.post(f"/api/sessions/{KNOWN}/messages/stream", json={"message": "BP?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"

    events = parse_sse(response.text)
    kinds = [kind for kind, _ in events]
    assert kinds[:2] == ["stage", "stage"]
    assert events[0][1] == {
        "stage": "emergency",
        "status": "start",
        "label": STAGE_LABELS[StageName.EMERGENCY],
    }
    assert kinds[2] == "meta"
    assert events[2][1] == {"action": "ANSWER", "message_id": str(MESSAGE_ID), "redaction_count": 2}
    tokens = [data["text"] for kind, data in events if kind == "token"]
    assert len(tokens) > 3
    assert "".join(tokens) == ANSWER_TEXT
    assert kinds[-1] == "done"
    done = MessageResponse.model_validate(events[-1][1])
    assert done.response_text == ANSWER_TEXT and done.citations[0].chunk_id == "c1"

    # the stream used its own session (not the request dependency) and committed it
    assert orchestrator.calls[0]["db"] is db
    assert db.commit.await_count == 1
    assert maker.opened == 2  # precheck + pipeline


def test_stream_unknown_session_404_before_streaming(client, orchestrator):
    response = client.post(f"/api/sessions/{uuid.uuid4()}/messages/stream", json={"message": "x"})
    assert response.status_code == 404
    assert response.json()["error_code"] == "SESSION_NOT_FOUND"
    assert orchestrator.calls == []


def test_stream_error_emits_error_then_safe_done(db, store, maker):
    failing = FakeOrchestrator(error=RuntimeError("boom"))
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(messages_router, prefix="/api")
    app.dependency_overrides[get_orchestrator] = lambda: failing
    app.dependency_overrides[require_session_access] = lambda: None
    response = TestClient(app).post(f"/api/sessions/{KNOWN}/messages/stream", json={"message": "x"})
    events = parse_sse(response.text)
    assert [k for k, _ in events][-2:] == ["error", "done"]
    assert "boom" not in response.text  # internal errors are never leaked
    done = MessageResponse.model_validate(events[-1][1])
    assert done.action == ActionEnum.GENERALIZE
    assert done.response_text == PIPELINE_ERROR_TEMPLATE


def test_stream_commit_failure_still_delivers_answer(client, db):
    db.commit.side_effect = RuntimeError("db down")
    response = client.post(f"/api/sessions/{KNOWN}/messages/stream", json={"message": "BP?"})
    events = parse_sse(response.text)
    meta = next(data for kind, data in events if kind == "meta")
    assert meta["message_id"] is None
    done = MessageResponse.model_validate(events[-1][1])
    assert done.response_text == ANSWER_TEXT and done.message_id is None
    db.rollback.assert_awaited_once()


async def test_stream_turn_completes_after_client_disconnect(db, maker):
    release = asyncio.Event()
    finished = asyncio.Event()

    class SlowOrchestrator(FakeOrchestrator):
        async def process_message(self, *args: Any, **kwargs: Any) -> MessageResponse:
            await kwargs["on_event"](
                PipelineEvent(type="stage", stage=StageName.PRIVACY, status="start", label="x")
            )
            await release.wait()
            kwargs["on_event"] = None
            result = await super().process_message(*args, **kwargs)
            finished.set()
            return result

    orchestrator = SlowOrchestrator()
    stream = stream_turn(orchestrator, KNOWN, "BP?")  # type: ignore[arg-type]
    first = await stream.__anext__()
    assert first.startswith("event: stage")
    await stream.aclose()  # the client went away
    release.set()
    await asyncio.wait_for(finished.wait(), timeout=2)
    await asyncio.sleep(0)
    assert db.commit.await_count == 1  # the turn was still committed


# ---------------------------------------------------------------------------
# POST /regenerate
# ---------------------------------------------------------------------------


def test_regenerate_json(client, orchestrator, db):
    response = client.post(f"/api/sessions/{KNOWN}/regenerate")
    assert response.status_code == 200
    assert MessageResponse.model_validate(response.json()).action == ActionEnum.ANSWER
    assert orchestrator.calls == [
        {"session_id": KNOWN, "message": "", "db": db, "regenerate": True}
    ]


def test_regenerate_stream(client, orchestrator):
    response = client.post(f"/api/sessions/{KNOWN}/regenerate?stream=true")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert parse_sse(response.text)[-1][0] == "done"
    assert orchestrator.calls[0]["regenerate"] is True


@pytest.mark.parametrize("stream", ["false", "true"])
def test_regenerate_nothing_to_regenerate_409(client, store, orchestrator, stream):
    store.get_last_user_message.return_value = None
    response = client.post(f"/api/sessions/{KNOWN}/regenerate?stream={stream}")
    assert response.status_code == 409
    assert orchestrator.calls == []


@pytest.mark.parametrize("stream", ["false", "true"])
def test_regenerate_unknown_session_404(client, stream):
    response = client.post(f"/api/sessions/{uuid.uuid4()}/regenerate?stream={stream}")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Erased (crypto-shredded) sessions: 410 before any pipeline/LLM work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["messages", "messages/stream", "regenerate", "regenerate?stream=true"],
)
def test_erased_session_rejected_with_410_before_any_llm_work(client, orchestrator, store, path):
    response = client.post(f"/api/sessions/{ERASED}/{path}", json={"message": "BP target?"})
    assert response.status_code == 410
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error_code"] == "SESSION_ERASED"
    assert orchestrator.calls == []
    store.get_last_user_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["", "one", "  leading space and trailing  ", "a b c d e f g", "line one\n\nline two [1]."],
)
def test_chunk_tokens_round_trip(text):
    tokens = chunk_tokens(text)
    assert "".join(tokens) == text
    assert all(len(t.split()) <= 3 for t in tokens)


def test_chunk_tokens_word_groups():
    assert chunk_tokens("a b c d e", words=2) == ["a b ", "c d ", "e"]


def test_sse_event_wire_format():
    assert sse_event("meta", {"action": "ANSWER", "x": "é"}) == (
        'event: meta\ndata: {"action": "ANSWER", "x": "é"}\n\n'
    )
