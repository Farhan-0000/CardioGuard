"""
Turn store (Module M-05): single-flush encrypted persistence and decrypted readers.

Hermetic tests use a mocked AsyncSession; ``postgres`` tests run against the
isolated ``medbridge_test`` database and roll back.
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.ai.schemas.extractor import DeltaEvent
from medbridge.api.schemas.enums import ActionEnum, EventTypeEnum
from medbridge.db.models import (
    AuditLog,
    ClinicalEvent,
    ContextSnapshot,
    MessageHistory,
    Session,
)
from medbridge.privacy.crypto import SessionCipher
from medbridge.state import turn_store
from medbridge.state.session_manager import (
    MessageNotFoundError,
    SessionErasedError,
    SessionNotFoundError,
    create_session,
)
from medbridge.state.turn_store import TurnRecord, merge_trace, split_trace

TRACE = {
    "gate1": {"action": "PROCEED", "rationale": "has BP 150/95", "source": "llm",
              "missing_fields": []},
    "gate2": {"action": "ANSWER", "rationale": "evidence fits", "source": "llm", "votes": []},
    "gate3": {"sentences": [{"text": "Your BP is high"}], "status": "ok",
              "action_change": "ANSWER->GENERALIZE", "unsupported_fraction": 0.5,
              "cleaned_text": "Your BP is high"},
    "emergency": {"action": None, "matched_rules": ["bp_crisis"]},
    "retrieval": [{"rank": 1, "chunk_id": "c1", "excerpt": "guideline text"}],
    "timings": [{"stage": "retrieval", "ms": 12.5, "status": "ok"}],
    "total_ms": 900.0,
    "citation_validation": {"valid": 1, "stripped": 0, "cited_chunk_ids": ["c1"]},
    "speculative_retrieval": {"used": True, "saved_ms": 120.0, "query": "bp 150/95 telma"},
    "redactions": 1,
    "regenerated": False,
}


def _record(session_id: uuid.UUID, **overrides: Any) -> TurnRecord:
    values: dict[str, Any] = dict(
        session_id=session_id,
        user_message="My BP is 150/95, what should I do?",
        assistant_message="That is stage 2 [1].",
        final_action=ActionEnum.ANSWER,
        citations=[{"marker": "[1]", "chunk_id": "c1", "source": "AHA", "section": "8",
                    "excerpt": "..."}],
        gate_1_action="PROCEED",
        gate_1_rationale="has BP 150/95",
        gate_2_action="ANSWER",
        gate_2_rationale="evidence fits",
        evidence_chunk_ids=["c1"],
        latency_ms=900,
        trace=dict(TRACE),
        delta_events=[DeltaEvent(event_type=EventTypeEnum.BP_READING,
                                 payload={"systolic": 150, "diastolic": 95})],
        new_snapshot={"recent_bp_readings": [{"systolic": 150, "diastolic": 95}]},
        soft_ask_count=0,
    )
    values.update(overrides)
    return TurnRecord(**values)


# ---------------------------------------------------------------------------
# Trace splitting (pure)
# ---------------------------------------------------------------------------

def test_split_trace_keeps_only_phi_free_fields_public() -> None:
    public, private = split_trace(TRACE)
    assert public["gate1"] == {"action": "PROCEED", "source": "llm", "missing_fields": []}
    assert public["gate2"]["action"] == "ANSWER" and "rationale" not in public["gate2"]
    assert public["timings"] == TRACE["timings"] and public["total_ms"] == 900.0
    assert public["speculative_retrieval"] == {"used": True, "saved_ms": 120.0}
    assert public["citation_validation"] == {"valid": 1, "stripped": 0}
    assert public["gate3"] == {"status": "ok", "action_change": "ANSWER->GENERALIZE",
                               "unsupported_fraction": 0.5}
    assert public["emergency"] == {"action": None}
    assert set(private) == {"gate1", "gate2", "gate3", "emergency", "citation_validation",
                            "speculative_retrieval"}
    assert set(private["gate3"]) == {"sentences", "cleaned_text"}
    assert "150/95" not in str(public) and "Your BP is high" not in str(public)
    assert merge_trace(public, private) == TRACE


# ---------------------------------------------------------------------------
# persist_turn with a mocked session: exactly one flush, everything encrypted
# ---------------------------------------------------------------------------

@pytest.fixture
def cipher() -> SessionCipher:
    return SessionCipher(Fernet(Fernet.generate_key()))


def _mock_db(rows: dict[tuple[type, uuid.UUID], Any]) -> MagicMock:
    db = MagicMock()
    db.info = {}
    db.get = AsyncMock(side_effect=lambda model, key: rows.get((model, key)))
    db.execute = AsyncMock()
    db.flush = AsyncMock()
    return db


async def test_persist_turn_single_flush_and_encryption(
    monkeypatch: pytest.MonkeyPatch, cipher: SessionCipher
) -> None:
    sid = uuid.uuid4()
    session = Session(session_id=sid)
    snapshot = ContextSnapshot(session_id=sid, snapshot={}, soft_ask_count=1)
    db = _mock_db({(Session, sid): session, (ContextSnapshot, sid): snapshot})
    monkeypatch.setattr(turn_store, "require_cipher", AsyncMock(return_value=cipher))

    user_id, assistant_id = await turn_store.persist_turn(db, _record(sid))

    db.flush.assert_awaited_once()
    added = [c.args[0] for c in db.add.call_args_list]
    events = [a for a in added if isinstance(a, ClinicalEvent)]
    messages = [a for a in added if isinstance(a, MessageHistory)]
    audits = [a for a in added if isinstance(a, AuditLog)]
    assert len(events) == 1 and len(messages) == 2 and len(audits) == 1
    assert cipher.decrypt_json(events[0].payload) == {"systolic": 150, "diastolic": 95}
    user, assistant = messages
    assert (user.message_id, assistant.message_id) == (user_id, assistant_id)
    assert user.created_at < assistant.created_at
    assert cipher.decrypt(user.content) == "My BP is 150/95, what should I do?"
    audit = audits[0]
    for field in ("request_message", "response_text", "gate_1_rationale", "gate_2_rationale"):
        assert getattr(audit, field).startswith("enc:v1:")
    assert audit.trace["assistant_message_id"] == str(assistant_id)
    assert cipher.decrypt_json(audit.trace["private"])["gate1"]["rationale"] == "has BP 150/95"
    assert audit.gate_1_action.value == "PROCEED" and audit.final_action.value == "ANSWER"
    assert cipher.decrypt_json(snapshot.snapshot)["recent_bp_readings"][0]["systolic"] == 150
    assert snapshot.soft_ask_count == 0
    assert session.last_active_at is not None


async def test_persist_turn_rejects_missing_and_erased_sessions(cipher: SessionCipher) -> None:
    sid = uuid.uuid4()
    with pytest.raises(SessionNotFoundError):
        await turn_store.persist_turn(_mock_db({}), _record(sid))
    from datetime import UTC, datetime
    erased = Session(session_id=sid, erased_at=datetime.now(UTC))
    db = _mock_db({(Session, sid): erased})
    with pytest.raises(SessionErasedError):
        await turn_store.persist_turn(db, _record(sid))
    db.flush.assert_not_awaited()


async def test_unknown_gate_and_event_types_are_dropped(
    monkeypatch: pytest.MonkeyPatch, cipher: SessionCipher
) -> None:
    sid = uuid.uuid4()
    db = _mock_db({(Session, sid): Session(session_id=sid)})
    monkeypatch.setattr(turn_store, "require_cipher", AsyncMock(return_value=cipher))
    record = _record(sid, gate_1_action="soft_ask", gate_2_action="MAYBE",
                     delta_events=[DeltaEvent(event_type="NOT_A_TYPE", payload={})])
    await turn_store.persist_turn(db, record)
    added = [c.args[0] for c in db.add.call_args_list]
    audit = next(a for a in added if isinstance(a, AuditLog))
    assert audit.gate_1_action.value == "SOFT-ASK" and audit.gate_2_action is None
    assert not any(isinstance(a, ClinicalEvent) for a in added)


# ---------------------------------------------------------------------------
# PostgreSQL behaviour
# ---------------------------------------------------------------------------

@pytest.fixture
async def db() -> AsyncIterator[AsyncSession]:
    from medbridge.db.connection import get_sessionmaker

    async with get_sessionmaker()() as session:
        yield session
        await session.rollback()


@pytest.mark.postgres
async def test_round_trip_readers(db: AsyncSession) -> None:
    sid = await create_session(db)
    state = await turn_store.load_session_state(db, sid)
    assert state.exists and not state.erased and state.snapshot == {}

    user_id, assistant_id = await turn_store.persist_turn(db, _record(sid))
    state = await turn_store.load_session_state(db, sid)
    assert state.snapshot["recent_bp_readings"] == [{"systolic": 150, "diastolic": 95}]

    history = await turn_store.get_history(db, sid)
    assert [(h.role, h.message_id) for h in history] == [("user", user_id),
                                                         ("assistant", assistant_id)]
    assert history[0].trace is None
    assert history[1].action == "ANSWER" and history[1].citations[0]["chunk_id"] == "c1"
    assert history[1].trace == {k: v for k, v in TRACE.items()}

    turns = await turn_store.get_recent_turns(db, sid, 1)
    assert [(t.role, t.action) for t in turns] == [("assistant", "ANSWER")]
    assert await turn_store.get_last_user_message(db, sid) == (
        user_id, "My BP is 150/95, what should I do?"
    )
    events = await turn_store.list_events(db, sid, ["BP_READING"])
    assert [e.payload for e in events] == [{"systolic": 150, "diastolic": 95}]
    assert await turn_store.list_events(db, sid, ["LAB_RESULT"]) == []
    assert await turn_store.list_events(db, sid, ["bogus"]) == []


@pytest.mark.postgres
async def test_regenerate_reuses_user_message_and_skips_events(db: AsyncSession) -> None:
    sid = await create_session(db)
    user_id, first_answer = await turn_store.persist_turn(db, _record(sid))
    regen = _record(sid, assistant_message="Regenerated answer [1].",
                    trace={**TRACE, "regenerated": True},
                    new_snapshot={"recent_bp_readings": []})
    regen_user_id, second_answer = await turn_store.persist_turn(db, regen)
    assert regen_user_id == user_id and second_answer != first_answer
    history = await turn_store.get_history(db, sid)
    assert [h.role for h in history] == ["user", "assistant", "assistant"]
    assert history[-1].content == "Regenerated answer [1]."
    assert len(await turn_store.list_events(db, sid)) == 1
    state = await turn_store.load_session_state(db, sid)
    assert state.snapshot["recent_bp_readings"]  # unchanged by the regenerate


@pytest.mark.postgres
async def test_manual_events_feedback_and_audit_listing(db: AsyncSession) -> None:
    sid = await create_session(db)
    _, assistant_id = await turn_store.persist_turn(db, _record(sid))
    snapshot = await turn_store.append_manual_events(db, sid, [
        DeltaEvent(event_type=EventTypeEnum.BP_READING,
                   payload={"systolic": 132, "diastolic": 84, "source": "manual"}),
    ])
    assert [r["systolic"] for r in snapshot["recent_bp_readings"]] == [150, 132]
    assert len(await turn_store.list_events(db, sid, ["BP_READING"])) == 2

    first = await turn_store.save_feedback(db, sid, assistant_id, 1)
    second = await turn_store.save_feedback(db, sid, assistant_id, -1, "Call me on 9876543210")
    assert first != second
    from sqlalchemy import select

    from medbridge.db.models import Feedback
    rows = (await db.execute(select(Feedback).where(Feedback.session_id == sid))).scalars().all()
    assert len(rows) == 1 and rows[0].rating == -1  # latest rating replaces the previous one
    assert rows[0].comment.startswith("enc:v1:")
    cipher = await turn_store.get_session_cipher(db, sid)
    assert cipher is not None and cipher.decrypt(rows[0].comment) == "Call me on [PHONE]"

    with pytest.raises(ValueError):
        await turn_store.save_feedback(db, sid, assistant_id, 0)
    with pytest.raises(MessageNotFoundError):
        await turn_store.save_feedback(db, sid, uuid.uuid4(), 1)
    user_msg = await turn_store.get_last_user_message(db, sid)
    assert user_msg is not None
    with pytest.raises(MessageNotFoundError):
        await turn_store.save_feedback(db, sid, user_msg[0], 1)

    entries = await turn_store.list_audit_entries(db, limit=50)
    mine = [e for e in entries if e.session_id == sid]
    assert mine and mine[0].request_text == "My BP is 150/95, what should I do?"
    assert mine[0].evidence_chunk_count == 1 and not mine[0].erased


@pytest.mark.postgres
async def test_missing_session_readers(db: AsyncSession) -> None:
    sid = uuid.uuid4()
    state = await turn_store.load_session_state(db, sid)
    assert not state.exists
    assert await turn_store.get_history(db, sid) == []
    with pytest.raises(SessionNotFoundError):
        await turn_store.erase_session(db, sid)
    with pytest.raises(SessionNotFoundError):
        await turn_store.append_manual_events(db, sid, [])
