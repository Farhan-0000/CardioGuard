"""
Concurrency, regenerate and erasure guarantees of the turn store — against the
isolated ``medbridge_test`` database (``postgres`` marker; skipped when down).

Regressions for the finalization review (stream F2):

* persistence-db-1 / invariants-2 / web-ui-2 — overlapping writers of one
  session (a slow turn, the BP form, a second turn, another worker) must never
  drop each other's events from the snapshot: ``snapshot == fold(events)``.
* persistence-db-3 — a regenerated answer is stored only against the re-run
  user message, and refused when a newer turn was stored meanwhile.
* invariants-1 / persistence-db-2 — the counter before a turn is recorded and
  returned for regenerate (first row wins; legacy fallbacks).
* persistence-db-4 — one feedback row per message, even under concurrency.
* privacy-security-3 — legacy plaintext rows are counted at erasure and never
  surface in any reader afterwards (no UPDATE/DELETE on append-only tables).

These tests commit (separate connections must see each other's writes); the
rows stay in the test database, which is never the development database.
"""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.ai.schemas.context_gate import ContextGateOutput
from medbridge.ai.schemas.evidence_gate import EvidenceGateOutput
from medbridge.ai.schemas.extractor import DeltaEvent, ExtractorOutput
from medbridge.api.schemas.enums import ActionEnum, EventTypeEnum
from medbridge.config import get_settings
from medbridge.core.orchestrator import PipelineOrchestrator
from medbridge.db.connection import get_sessionmaker
from medbridge.db.models import AuditLog, ClinicalEvent, Feedback, FinalActionEnum, Session
from medbridge.state import turn_store
from medbridge.state.projector import StateProjector
from medbridge.state.session_manager import create_session
from medbridge.state.turn_store import RegenerateConflictError, TurnRecord

pytestmark = pytest.mark.postgres

AMLODIPINE = DeltaEvent(
    event_type=EventTypeEnum.MEDICATION_ADDED,
    payload={"drug": "amlodipine", "dosage": "5 mg"},
)
AGE_62 = DeltaEvent(event_type=EventTypeEnum.DEMOGRAPHIC, payload={"age": 62})


def bp(systolic: int, diastolic: int) -> DeltaEvent:
    return DeltaEvent(
        event_type=EventTypeEnum.BP_READING,
        payload={"systolic": systolic, "diastolic": diastolic, "source": "manual"},
    )


def record(
    session_id: uuid.UUID,
    events: list[DeltaEvent] | None = None,
    stale_snapshot: dict[str, Any] | None = None,
    **overrides: Any,
) -> TurnRecord:
    events = list(events or [])
    values: dict[str, Any] = dict(
        session_id=session_id,
        user_message="question",
        assistant_message="answer",
        final_action=ActionEnum.GENERALIZE,
        trace={"regenerated": False, "soft_ask_count_before": 0},
        delta_events=events,
        # what the orchestrator projected from the state it loaded at turn start
        new_snapshot=StateProjector().apply_events(stale_snapshot or {}, events) if events else None,
        soft_ask_count=0,
    )
    values.update(overrides)
    return TurnRecord(**values)


@pytest.fixture
async def sid() -> AsyncIterator[uuid.UUID]:
    async with get_sessionmaker()() as db:
        session_id = await create_session(db)
        await db.commit()
    yield session_id


async def committed(work: Any) -> Any:
    """Run ``work(db)`` in its own session and commit."""
    async with get_sessionmaker()() as db:
        result = await work(db)
        await db.commit()
        return result


async def stored_state(session_id: uuid.UUID) -> tuple[dict[str, Any], dict[str, Any], int]:
    """(stored snapshot, fold of the stored events, soft_ask_count)."""
    async with get_sessionmaker()() as db:
        state = await turn_store.load_session_state(db, session_id)
        events = await turn_store.list_events(db, session_id)
    folded = StateProjector().reconstruct_from_events(
        [DeltaEvent(event_type=e.event_type, payload=e.payload) for e in events]
    )
    return state.snapshot, folded, state.soft_ask_count


# ---------------------------------------------------------------------------
# Lost updates (persistence-db-1 / invariants-2 / web-ui-2)
# ---------------------------------------------------------------------------


async def test_bp_form_reading_survives_a_turn_that_loaded_state_earlier(sid: uuid.UUID) -> None:
    """The reported failure: the BP form commits while the turn's LLM work runs."""
    async with get_sessionmaker()() as turn_db:
        stale = await turn_store.load_session_state(turn_db, sid)  # turn starts: S0
        await committed(lambda db: turn_store.append_manual_events(db, sid, [bp(182, 121)]))
        await turn_store.persist_turn(turn_db, record(sid, [AMLODIPINE], stale.snapshot))
        await turn_db.commit()

    snapshot, folded, _ = await stored_state(sid)
    assert snapshot == folded  # the snapshot is still a fold over the event log
    assert snapshot["recent_bp_readings"][0]["systolic"] == 182
    assert snapshot["current_medications"][0]["drug"] == "amlodipine"


async def test_overlapping_writers_are_serialised_and_fold_holds(sid: uuid.UUID) -> None:
    """Five writers load the same state, then all persist at once."""
    loaded = asyncio.Event()
    ready = 0

    async def turn(events: list[DeltaEvent]) -> None:
        nonlocal ready
        async with get_sessionmaker()() as db:
            state = await turn_store.load_session_state(db, sid)
            ready += 1
            if ready == 5:
                loaded.set()
            await loaded.wait()  # every writer holds a stale snapshot now
            await turn_store.persist_turn(db, record(sid, events, state.snapshot))
            await db.commit()

    async def bp_form(reading: DeltaEvent) -> None:
        nonlocal ready
        async with get_sessionmaker()() as db:
            ready += 1
            if ready == 5:
                loaded.set()
            await loaded.wait()
            await turn_store.append_manual_events(db, sid, [reading])
            await db.commit()

    await asyncio.gather(
        turn([AMLODIPINE]), turn([AGE_62]), turn([bp(150, 95)]),
        bp_form(bp(132, 84)), bp_form(bp(128, 80)),
    )
    snapshot, folded, _ = await stored_state(sid)
    assert snapshot == folded
    assert snapshot["current_medications"][0]["drug"] == "amlodipine"
    assert snapshot["demographics"]["age"] == 62
    assert sorted(r["systolic"] for r in snapshot["recent_bp_readings"]) == [128, 132, 150]


async def test_turn_without_gate1_leaves_the_counter_alone(sid: uuid.UUID) -> None:
    await committed(lambda db: turn_store.persist_turn(db, record(sid, soft_ask_count=2)))
    emergency = record(sid, soft_ask_count=0, update_soft_ask_count=False,
                       final_action=ActionEnum.ESCALATE)
    await committed(lambda db: turn_store.persist_turn(db, emergency))
    assert (await stored_state(sid))[2] == 2


# ---------------------------------------------------------------------------
# Regenerate (persistence-db-3, invariants-1)
# ---------------------------------------------------------------------------


async def test_regenerate_is_refused_when_a_newer_turn_was_stored(sid: uuid.UUID) -> None:
    q1, _ = await committed(lambda db: turn_store.persist_turn(db, record(sid)))
    # Stop + reload + Regenerate: the regenerate resolves Q1 ...
    # ... while the stopped turn (Q2) is persisted in the background.
    q2, _ = await committed(lambda db: turn_store.persist_turn(
        db, record(sid, user_message="is 150/95 high?")
    ))
    regen = record(sid, assistant_message="regenerated ibuprofen answer",
                   trace={"regenerated": True, "soft_ask_count_before": 0},
                   regenerated_user_message_id=q1)
    async with get_sessionmaker()() as db:
        with pytest.raises(RegenerateConflictError):
            await turn_store.persist_turn(db, regen)
        await db.rollback()
        history = await turn_store.get_history(db, sid)
    assert [h.role for h in history] == ["user", "assistant", "user", "assistant"]

    regen_q2 = record(sid, assistant_message="regenerated",
                      trace={"regenerated": True, "soft_ask_count_before": 0},
                      regenerated_user_message_id=q2)
    user_id, answer_id = await committed(lambda db: turn_store.persist_turn(db, regen_q2))
    assert user_id == q2
    async with get_sessionmaker()() as db:
        history = await turn_store.get_history(db, sid)
    assert [h.role for h in history][-3:] == ["user", "assistant", "assistant"]
    assert history[-3].message_id == q2 and history[-1].message_id == answer_id
    assert history[-1].trace is not None


async def test_counter_before_is_recorded_first_row_wins_and_recent_turns_collapse(
    sid: uuid.UUID,
) -> None:
    soft_ask = record(sid, final_action=ActionEnum.SOFT_ASK, soft_ask_count=1,
                      trace={"regenerated": False, "soft_ask_count_before": 0})
    user_id, _ = await committed(lambda db: turn_store.persist_turn(db, soft_ask))
    for before in (0, 7):  # a later (wrong) value never replaces the turn's own
        regen = record(sid, final_action=ActionEnum.SOFT_ASK, soft_ask_count=1,
                       trace={"regenerated": True, "soft_ask_count_before": before},
                       regenerated_user_message_id=user_id, assistant_message=f"again {before}")
        await committed(lambda db, regen=regen: turn_store.persist_turn(db, regen))

    async with get_sessionmaker()() as db:
        assert await turn_store.get_soft_ask_count_before(db, sid, user_id) == 0
        turns = await turn_store.get_recent_turns(db, sid, 4)
    # superseded answers are dropped: one question, its latest answer
    assert [(t.role, t.content) for t in turns] == [("user", "question"),
                                                    ("assistant", "again 7")]


async def test_counter_before_legacy_fallbacks(sid: uuid.UUID) -> None:
    async def legacy_row(final_action: FinalActionEnum, trace: dict[str, Any]) -> None:
        async def add(db: AsyncSession) -> None:
            db.add(AuditLog(session_id=sid, request_message="enc:v1:x", response_text="enc:v1:y",
                            final_action=final_action, trace=trace))
        await committed(add)

    unknown = uuid.uuid4()
    async with get_sessionmaker()() as db:
        assert await turn_store.get_soft_ask_count_before(db, sid, unknown) is None
    await legacy_row(FinalActionEnum.GENERALIZE, {"gate2": {"source": "loop_breaker"}})
    async with get_sessionmaker()() as db:
        assert await turn_store.get_soft_ask_count_before(db, sid, unknown) == (
            get_settings().SOFT_ASK_MAX_COUNT
        )
    await committed(lambda db: turn_store.persist_turn(db, record(sid, soft_ask_count=2)))
    await legacy_row(FinalActionEnum.SOFT_ASK, {"total_ms": 1})
    async with get_sessionmaker()() as db:
        assert await turn_store.get_soft_ask_count_before(db, sid, unknown) == 1


# ---------------------------------------------------------------------------
# Feedback (persistence-db-4)
# ---------------------------------------------------------------------------


async def test_concurrent_feedback_leaves_one_row_per_message(sid: uuid.UUID) -> None:
    _, answer = await committed(lambda db: turn_store.persist_turn(db, record(sid)))

    async def rate(rating: int) -> uuid.UUID:
        return await committed(
            lambda db: turn_store.save_feedback(db, sid, answer, rating)  # type: ignore[no-any-return]
        )

    ids = await asyncio.gather(rate(1), rate(-1), rate(1), rate(-1))
    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Feedback).where(Feedback.message_id == answer)
        )).scalars().all()
        assert len(rows) == 1 and rows[0].feedback_id in ids
        with pytest.raises(IntegrityError):  # enforced by the database itself
            await db.execute(text(
                "INSERT INTO feedback (feedback_id, session_id, message_id, rating) "
                "VALUES (gen_random_uuid(), :sid, :mid, 1)"
            ), {"sid": sid, "mid": answer})


# ---------------------------------------------------------------------------
# Legacy plaintext and erasure (privacy-security-3)
# ---------------------------------------------------------------------------


async def test_erasure_marks_legacy_plaintext_rows_and_hides_them(sid: uuid.UUID) -> None:
    await committed(lambda db: turn_store.persist_turn(db, record(sid, [AMLODIPINE])))

    async def add_legacy(db: AsyncSession) -> None:  # rows written before encryption existed
        db.add(AuditLog(session_id=sid, request_message="I'm Asha Rao, 9876543210",
                        response_text="plain answer", final_action=FinalActionEnum.ANSWER))
        db.add(ClinicalEvent(session_id=sid, event_type=EventTypeEnum.DEMOGRAPHIC,
                             payload={"age": 54}))
    await committed(add_legacy)

    async def count(model: Any) -> int:
        async with get_sessionmaker()() as db:
            return int((await db.execute(
                select(func.count()).select_from(model).where(model.session_id == sid)
            )).scalar_one())

    before = (await count(AuditLog), await count(ClinicalEvent))
    async with get_sessionmaker()() as db:
        entries = [e for e in await turn_store.list_audit_entries(db, 500) if e.session_id == sid]
        assert sorted(e.legacy_plaintext for e in entries) == [False, True]

    await committed(lambda db: turn_store.erase_session(db, sid))

    async with get_sessionmaker()() as db:
        session = await db.get(Session, sid)
        assert session is not None and session.erased_at is not None
        assert session.legacy_plaintext_rows == 2
        entries = [e for e in await turn_store.list_audit_entries(db, 500) if e.session_id == sid]
        assert entries and all(e.erased and e.request_text is None and e.response_text is None
                               for e in entries)
        assert await turn_store.list_events(db, sid) == []
        assert await turn_store.get_history(db, sid) == []
        assert await turn_store.get_recent_turns(db, sid, 10) == []
        assert await turn_store.get_last_user_message(db, sid) is None
    assert (await count(AuditLog), await count(ClinicalEvent)) == before  # append-only intact


# ---------------------------------------------------------------------------
# End to end: two concurrent turns + the BP form through the real orchestrator
# ---------------------------------------------------------------------------


def orchestrator_with_fake_llms(release: asyncio.Event, started: asyncio.Event) -> Any:
    async def extract(message: str, snapshot: dict[str, Any], history: Any = None) -> Any:
        if "amlodipine" in message:
            started.set()
            await release.wait()  # slow LLM on the first turn
            return ExtractorOutput(delta_events=[AMLODIPINE], search_query="amlodipine dose",
                                   raw_intent="dose change")
        return ExtractorOutput(delta_events=[AGE_62], search_query="age", raw_intent="age")

    extractor = MagicMock()
    extractor.extract = AsyncMock(side_effect=extract)
    context_gate = MagicMock()
    context_gate.evaluate = AsyncMock(return_value=ContextGateOutput(action="PROCEED",
                                                                     rationale="ok"))
    retriever = MagicMock()
    retriever.hybrid_search = AsyncMock(return_value=[])
    evidence_gate = MagicMock()  # ABSTAIN: template bypass, no generator needed
    evidence_gate.evaluate = AsyncMock(return_value=EvidenceGateOutput(
        action="ABSTAIN", evidence_sufficient=False, rationale="none"))
    settings = get_settings().model_copy(update={
        "SPECULATIVE_RETRIEVAL": False, "SMALL_TALK_ENABLED": False,
        "GROUNDING_VERIFIER_ENABLED": False, "EVIDENCE_GATE_CALIBRATION_ENABLED": False,
    })
    return PipelineOrchestrator(
        emergency_classifier=SimpleNamespace(classify=lambda t: None),
        context_extractor=extractor,
        context_gate=context_gate,
        hybrid_retriever=retriever,
        reranker=MagicMock(),
        evidence_gate=evidence_gate,
        response_generator=MagicMock(),
        settings=settings,
        redactor=lambda t: SimpleNamespace(text=t, count=0, categories={}),
    )


async def test_concurrent_turns_and_bp_form_keep_every_fact(sid: uuid.UUID) -> None:
    release, started = asyncio.Event(), asyncio.Event()
    orchestrator = orchestrator_with_fake_llms(release, started)

    slow = asyncio.create_task(
        orchestrator.process_message(sid, "I take amlodipine 5 mg, can I double it?")
    )
    await started.wait()
    second = asyncio.create_task(orchestrator.process_message(sid, "I'm 62 by the way"))
    await committed(lambda db: turn_store.append_manual_events(db, sid, [bp(182, 121)]))
    release.set()
    first_response, second_response = await asyncio.gather(slow, second)

    assert first_response.message_id is not None and second_response.message_id is not None
    snapshot, folded, _ = await stored_state(sid)
    assert snapshot == folded
    assert snapshot["current_medications"][0]["drug"] == "amlodipine"
    assert snapshot["demographics"]["age"] == 62
    assert snapshot["recent_bp_readings"][0]["systolic"] == 182
