"""Encryption at rest and crypto-shredding against the real (test) PostgreSQL database."""
from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import text

from medbridge.ai.schemas.extractor import DeltaEvent
from medbridge.api.schemas.enums import ActionEnum, EventTypeEnum
from medbridge.privacy.crypto import get_session_cipher
from medbridge.state import turn_store
from medbridge.state.session_manager import create_session
from medbridge.state.turn_store import TurnRecord

pytestmark = pytest.mark.postgres

USER_TEXT = "I am on Telma 40 and my BP is 152/97 since Monday, feeling dizzy"
ASSISTANT_TEXT = "A reading of 152/97 is in the stage 2 range [1]. Please contact your doctor."
GATE1 = "Patient reports telmisartan and BP 152/97 with dizziness"
GATE2 = "Evidence covers stage 2 management for this patient"
SECRETS = ("Telma", "152/97", "dizzy", "telmisartan", "stage 2 management", "contact your doctor")


def _record(session_id: uuid.UUID) -> TurnRecord:
    return TurnRecord(
        session_id=session_id,
        user_message=USER_TEXT,
        assistant_message=ASSISTANT_TEXT,
        final_action=ActionEnum.ANSWER,
        citations=[{"marker": "[1]", "chunk_id": "aha_c1", "source": "AHA/ACC 2025",
                    "section": "8.1", "excerpt": "Stage 2 hypertension is ..."}],
        gate_1_action="PROCEED",
        gate_1_rationale=GATE1,
        gate_2_action="ANSWER",
        gate_2_rationale=GATE2,
        evidence_chunk_ids=["aha_c1"],
        latency_ms=1500,
        trace={
            "gate1": {"action": "PROCEED", "rationale": GATE1, "source": "llm"},
            "gate2": {"action": "ANSWER", "rationale": GATE2, "source": "llm"},
            "emergency": {"matched": False, "note": "dizzy but no red flags"},
            "timings": [{"stage": "emergency", "ms": 0.4, "status": "ok"}],
            "total_ms": 1500.0,
            "models": {"gates": "qwen3:8b"},
        },
        delta_events=[
            DeltaEvent(event_type=EventTypeEnum.BP_READING,
                       payload={"systolic": 152, "diastolic": 97}),
            DeltaEvent(event_type=EventTypeEnum.MEDICATION_ADDED,
                       payload={"drug_name": "Telma", "dosage": "40mg"}),
            DeltaEvent(event_type=EventTypeEnum.SYMPTOM_REPORTED, payload={"symptom": "dizzy"}),
        ],
        new_snapshot={"recent_bp_readings": [{"systolic": 152, "diastolic": 97}],
                      "current_medications": [{"drug": "Telma", "dosage": "40mg",
                                               "frequency": ""}],
                      "symptoms": ["dizzy"]},
        soft_ask_count=0,
    )


async def _rows(db: Any, sql: str, sid: uuid.UUID) -> list[Any]:
    return list((await db.execute(text(sql), {"sid": sid})).all())


def _assert_no_plaintext(values: list[Any]) -> None:
    blob = " ".join(str(v) for v in values)
    for secret in SECRETS:
        assert secret not in blob, f"plaintext {secret!r} found at rest"


async def test_persisted_rows_contain_no_plaintext(db_session: Any) -> None:
    sid = await create_session(db_session)
    await turn_store.persist_turn(db_session, _record(sid))

    messages = await _rows(db_session, "SELECT content FROM message_history "
                                       "WHERE session_id = :sid", sid)
    assert len(messages) == 2 and all(m[0].startswith("enc:v1:") for m in messages)

    audits = await _rows(db_session, "SELECT request_message, response_text, gate_1_rationale, "
                                     "gate_2_rationale, trace::text FROM audit_logs "
                                     "WHERE session_id = :sid", sid)
    assert len(audits) == 1
    request, response, r1, r2, trace = audits[0]
    assert all(v.startswith("enc:v1:") for v in (request, response, r1, r2))
    assert '"enc": "v1"' in trace and '"total_ms": 1500.0' in trace

    events = await _rows(db_session, "SELECT payload::text FROM clinical_events "
                                     "WHERE session_id = :sid", sid)
    assert len(events) == 3 and all('"enc": "v1"' in e[0] for e in events)
    snapshot = await _rows(db_session, "SELECT snapshot::text FROM context_snapshots "
                                       "WHERE session_id = :sid", sid)
    assert '"enc": "v1"' in snapshot[0][0]

    _assert_no_plaintext([*messages, *audits, *events, *snapshot])

    # ...while the application reads everything back transparently.
    history = await turn_store.get_history(db_session, sid)
    assert [h.content for h in history] == [USER_TEXT, ASSISTANT_TEXT]
    assert history[1].trace["gate1"]["rationale"] == GATE1
    assert history[1].trace["emergency"]["note"] == "dizzy but no red flags"


async def test_crypto_shredding_preserves_append_only_rows(db_session: Any) -> None:
    sid = await create_session(db_session)
    _, assistant_id = await turn_store.persist_turn(db_session, _record(sid))
    await turn_store.save_feedback(db_session, sid, assistant_id, -1, "my name is Rahul Sharma")
    audit_before = await _rows(db_session, "SELECT audit_id, request_message FROM audit_logs "
                                           "WHERE session_id = :sid", sid)

    await turn_store.erase_session(db_session, sid)

    # The key is gone: nothing can be decrypted any more.
    assert await get_session_cipher(db_session, sid) is None
    assert await _rows(db_session, "SELECT 1 FROM session_keys WHERE session_id = :sid", sid) == []
    state = await turn_store.load_session_state(db_session, sid)
    assert state.erased and state.snapshot == {}
    assert await turn_store.get_history(db_session, sid) == []
    assert await turn_store.list_events(db_session, sid) == []
    assert await turn_store.get_recent_turns(db_session, sid, 4) == []
    assert await turn_store.get_last_user_message(db_session, sid) is None

    # Mutable rows deleted, session stamped, append-only rows untouched (still ciphertext).
    for table in ("message_history", "context_snapshots", "feedback"):
        assert await _rows(db_session, f"SELECT 1 FROM {table} WHERE session_id = :sid", sid) == []
    erased_at = await _rows(db_session, "SELECT erased_at FROM sessions WHERE session_id = :sid",
                            sid)
    assert erased_at[0][0] is not None
    assert await _rows(db_session, "SELECT audit_id, request_message FROM audit_logs "
                                   "WHERE session_id = :sid", sid) == audit_before
    assert len(await _rows(db_session, "SELECT 1 FROM clinical_events "
                                       "WHERE session_id = :sid", sid)) == 3

    # The immutability rules still turn UPDATE/DELETE into no-ops.
    await db_session.execute(text("UPDATE audit_logs SET response_text = 'x' "
                                  "WHERE session_id = :sid"), {"sid": sid})
    await db_session.execute(text("DELETE FROM audit_logs WHERE session_id = :sid"), {"sid": sid})
    await db_session.execute(text("DELETE FROM clinical_events WHERE session_id = :sid"),
                             {"sid": sid})
    assert await _rows(db_session, "SELECT audit_id, request_message FROM audit_logs "
                                   "WHERE session_id = :sid", sid) == audit_before
    assert len(await _rows(db_session, "SELECT 1 FROM clinical_events "
                                       "WHERE session_id = :sid", sid)) == 3

    # Erasure is idempotent and further writes are refused.
    await turn_store.erase_session(db_session, sid)
    with pytest.raises(turn_store.SessionErasedError):
        await turn_store.persist_turn(db_session, _record(sid))
