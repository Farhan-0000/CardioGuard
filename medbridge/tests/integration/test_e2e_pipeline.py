"""
End-to-end pipeline tests against live PostgreSQL and Qdrant (TASK-31).

Every LLM-backed component (extractor, both gates, generator) is a
deterministic fake, so no LLM is ever called; the database (encrypted turn
store, separate ``medbridge_test`` database), the emergency classifier, PII
redaction, hybrid retrieval over the indexed corpus and the cross-encoder
reranker are real.

Covers: emergency fast-path, a two-turn SOFT-ASK → ANSWER conversation with
state accumulation, the loop-breaker, ABSTAIN template bypass, regeneration,
and the HTTP JSON + SSE round trip with session tokens.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.ai.schemas.context_gate import ContextGateOutput
from medbridge.ai.schemas.evidence_gate import EvidenceGateOutput
from medbridge.ai.schemas.extractor import DeltaEvent, ExtractorOutput
from medbridge.ai.schemas.response_generator import ResponseGeneratorOutput
from medbridge.api.schemas.enums import ActionEnum, EventTypeEnum
from medbridge.core import emergency_classifier
from medbridge.core.orchestrator import PipelineOrchestrator
from medbridge.core.templates import ABSTAIN_TEMPLATE, LOOP_BREAKER_PREFIX
from medbridge.db.connection import get_sessionmaker
from medbridge.db.models import AuditLog, ContextSnapshot
from medbridge.main import create_app
from medbridge.retrieval.hybrid_retriever import HybridRetriever
from medbridge.state import turn_store
from medbridge.state.session_manager import create_session

pytestmark = [pytest.mark.postgres, pytest.mark.qdrant]

EMERGENCY = (
    "My blood pressure is 220/130 mmHg and I have crushing chest pain, "
    "severe shortness of breath, and visual blurriness."
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session


@pytest.fixture
async def session_id(db_session: AsyncSession) -> uuid.UUID:
    sid = await create_session(db_session)
    await db_session.commit()
    return sid


@pytest.fixture
async def retriever() -> AsyncIterator[HybridRetriever]:
    instance = HybridRetriever()  # fresh client bound to this test's event loop
    yield instance
    await instance.close()


def fake_llm_components() -> dict[str, Any]:
    """Deterministic stand-ins for the four LLM calls (no network)."""
    extractor = MagicMock()
    extractor.extract = AsyncMock(
        return_value=ExtractorOutput(
            search_query="blood pressure target guideline", raw_intent="ask about BP target"
        )
    )
    context_gate = MagicMock()
    context_gate.evaluate = AsyncMock(
        return_value=ContextGateOutput(action="PROCEED", rationale="Context sufficient.")
    )
    evidence_gate = MagicMock()
    evidence_gate.evaluate = AsyncMock(
        return_value=EvidenceGateOutput(
            action="ANSWER", evidence_sufficient=True, rationale="Evidence covers it."
        )
    )
    generator = MagicMock()
    generator.generate = AsyncMock(
        return_value=ResponseGeneratorOutput(
            response_text="Guidelines recommend a target below 130/80 mmHg [1], measured at home [2]."
        )
    )
    return {
        "context_extractor": extractor,
        "context_gate": context_gate,
        "evidence_gate": evidence_gate,
        "response_generator": generator,
    }


async def audit_rows(db: AsyncSession, sid: uuid.UUID) -> list[AuditLog]:
    stmt = select(AuditLog).where(AuditLog.session_id == sid).order_by(AuditLog.created_at.asc())
    return list((await db.execute(stmt)).scalars().all())


# ===========================================================================
# 1. Emergency fast-path
# ===========================================================================


async def test_emergency_fast_path_latency_and_persistence(db_session, session_id, retriever):
    components = fake_llm_components()
    orchestrator = PipelineOrchestrator(hybrid_retriever=retriever, **components)

    start = time.perf_counter()
    assert emergency_classifier.classify(EMERGENCY) == ActionEnum.ESCALATE
    assert (time.perf_counter() - start) * 1000 < 10.0  # deterministic, < 10 ms

    response = await orchestrator.process_message(session_id, EMERGENCY, db=db_session)
    await db_session.commit()

    assert response.action == ActionEnum.ESCALATE
    assert "911" in response.response_text and "112" in response.response_text
    assert response.citations == []
    stage_ms = {t.stage: t.ms for t in response.explanation.timings}
    assert stage_ms["emergency"] < 10.0
    components["context_extractor"].extract.assert_not_called()  # no LLM call at all
    components["context_gate"].evaluate.assert_not_called()
    components["evidence_gate"].evaluate.assert_not_called()
    components["response_generator"].generate.assert_not_called()

    history = await turn_store.get_history(db_session, session_id)
    assert [(h.role, h.action) for h in history] == [("user", None), ("assistant", "ESCALATE")]
    assert history[0].content == EMERGENCY
    assert history[1].message_id == response.message_id
    assert history[1].trace["gate2"]["source"] == "fastpath"

    (audit,) = await audit_rows(db_session, session_id)
    assert audit.final_action.value == "ESCALATE" and audit.gate_2_action.value == "ESCALATE"
    assert audit.gate_1_action is None
    assert EMERGENCY not in json.dumps(audit.trace)  # plaintext trace carries no message text


# ===========================================================================
# 2. SOFT-ASK → ANSWER with live retrieval and state accumulation
# ===========================================================================


async def test_multi_turn_soft_ask_then_answer(db_session, session_id, retriever):
    components = fake_llm_components()
    components["context_extractor"].extract.side_effect = [
        ExtractorOutput(
            delta_events=[
                DeltaEvent(
                    event_type=EventTypeEnum.DEMOGRAPHIC, payload={"age": 58, "sex": "male"}
                ),
                DeltaEvent(
                    event_type=EventTypeEnum.BP_READING, payload={"systolic": 155, "diastolic": 95}
                ),
            ],
            search_query="hypertension treatment stage 2",
            raw_intent="high BP reading",
        ),
        ExtractorOutput(
            delta_events=[
                DeltaEvent(
                    event_type=EventTypeEnum.MEDICATION_ADDED,
                    payload={"drug_name": "Lisinopril", "dosage": "10mg", "frequency": "daily"},
                ),
            ],
            search_query="blood pressure target diabetes hypertension",
            raw_intent="target with diabetes",
        ),
    ]
    components["context_gate"].evaluate.side_effect = [
        ContextGateOutput(
            action="SOFT-ASK",
            rationale="Medications unknown.",
            missing_fields=["current_medications"],
        ),
        ContextGateOutput(action="PROCEED", rationale="Context complete."),
    ]
    components["response_generator"].generate.side_effect = [
        ResponseGeneratorOutput(response_text="Which blood pressure medicines do you take?"),
        ResponseGeneratorOutput(
            response_text="Most guidelines aim for below 130/80 mmHg [1]. Combination therapy is common [2]."
        ),
    ]
    orchestrator = PipelineOrchestrator(hybrid_retriever=retriever, **components)

    first = await orchestrator.process_message(
        session_id, "I'm 58 and my BP was 155/95 this morning.", db=db_session
    )
    await db_session.commit()
    assert first.action == ActionEnum.SOFT_ASK and first.soft_ask_count == 1
    components["evidence_gate"].evaluate.assert_not_called()

    second = await orchestrator.process_message(
        session_id, "I take Lisinopril 10mg daily and I have type 2 diabetes.", db=db_session
    )
    await db_session.commit()
    assert second.action == ActionEnum.ANSWER and second.soft_ask_count == 0
    assert [c.marker for c in second.citations] == ["[1]", "[2]"]
    retrieved_ids = [r.chunk_id for r in second.explanation.retrieval]
    assert len(retrieved_ids) == 5  # live hybrid search + rerank top-5
    assert {c.chunk_id for c in second.citations} <= set(retrieved_ids)
    assert all(c.source and c.excerpt for c in second.citations)
    assert (
        second.explanation.retrieval[0].score_rerank
        >= second.explanation.retrieval[-1].score_rerank
    )

    state = await turn_store.load_session_state(db_session, session_id)
    assert state.soft_ask_count == 0
    assert state.snapshot["demographics"]["age"] == 58
    assert state.snapshot["recent_bp_readings"][0]["systolic"] == 155
    assert any(m["drug"] == "Lisinopril" for m in state.snapshot["current_medications"])

    events = await turn_store.list_events(db_session, session_id)
    assert sorted(e.event_type for e in events) == ["BP_READING", "DEMOGRAPHIC", "MEDICATION_ADDED"]

    history = await turn_store.get_history(db_session, session_id)
    assert [h.action for h in history if h.role == "assistant"] == ["SOFT-ASK", "ANSWER"]

    audits = await audit_rows(db_session, session_id)
    assert [(a.gate_1_action.value, a.final_action.value) for a in audits] == [
        ("SOFT-ASK", "SOFT-ASK"),
        ("PROCEED", "ANSWER"),
    ]
    assert audits[1].evidence_chunk_ids == retrieved_ids


# ===========================================================================
# 3. Loop-breaker (soft_ask_count >= 2 → GENERALIZE)
# ===========================================================================


async def test_loop_breaker_forces_generalize(db_session, session_id, retriever):
    snapshot = await db_session.get(ContextSnapshot, session_id)
    snapshot.soft_ask_count = 2
    await db_session.commit()

    components = fake_llm_components()
    components["context_gate"].evaluate.return_value = ContextGateOutput(
        action="SOFT-ASK", rationale="Still no reading.", missing_fields=["blood_pressure_reading"]
    )
    components["response_generator"].generate.return_value = ResponseGeneratorOutput(
        response_text="Regular home monitoring helps [1]."
    )
    orchestrator = PipelineOrchestrator(hybrid_retriever=retriever, **components)
    response = await orchestrator.process_message(
        session_id, "What should I do about my condition?", db=db_session
    )
    await db_session.commit()

    assert response.action == ActionEnum.GENERALIZE
    assert response.response_text.startswith(LOOP_BREAKER_PREFIX)
    assert response.soft_ask_count == 0
    components["evidence_gate"].evaluate.assert_not_called()
    assert response.explanation.gate2.source == "loop_breaker"
    assert (await turn_store.load_session_state(db_session, session_id)).soft_ask_count == 0
    (audit,) = await audit_rows(db_session, session_id)
    assert audit.final_action.value == "GENERALIZE" and audit.gate_2_action.value == "GENERALIZE"


# ===========================================================================
# 4. ABSTAIN template bypass
# ===========================================================================


async def test_insufficient_evidence_abstains_without_generator(db_session, session_id, retriever):
    components = fake_llm_components()
    components["evidence_gate"].evaluate.return_value = EvidenceGateOutput(
        action="ABSTAIN", evidence_sufficient=False, rationale="Dermatology is out of scope."
    )
    orchestrator = PipelineOrchestrator(hybrid_retriever=retriever, **components)
    response = await orchestrator.process_message(
        session_id, "What dose of topical steroid should I use for eczema?", db=db_session
    )
    await db_session.commit()

    assert response.action == ActionEnum.ABSTAIN
    assert response.response_text == ABSTAIN_TEMPLATE
    components["response_generator"].generate.assert_not_called()
    (audit,) = await audit_rows(db_session, session_id)
    assert audit.gate_2_action.value == "ABSTAIN" and audit.final_action.value == "ABSTAIN"


# ===========================================================================
# 5. Regenerate
# ===========================================================================


async def test_regenerate_adds_answer_without_new_user_message(db_session, session_id, retriever):
    components = fake_llm_components()
    components["context_extractor"].extract.return_value = ExtractorOutput(
        delta_events=[
            DeltaEvent(
                event_type=EventTypeEnum.BP_READING, payload={"systolic": 140, "diastolic": 90}
            )
        ],
        search_query="blood pressure target",
        raw_intent="target",
    )
    orchestrator = PipelineOrchestrator(hybrid_retriever=retriever, **components)
    first = await orchestrator.process_message(
        session_id, "My BP is 140/90, what target?", db=db_session
    )
    await db_session.commit()
    again = await orchestrator.process_message(session_id, "", db=db_session, regenerate=True)
    await db_session.commit()

    assert again.explanation.regenerated is True
    assert again.user_message_id == first.user_message_id
    assert again.message_id != first.message_id
    history = await turn_store.get_history(db_session, session_id)
    assert [h.role for h in history] == ["user", "assistant", "assistant"]
    assert len(await turn_store.list_events(db_session, session_id)) == 1  # no duplicate event
    assert components["context_extractor"].extract.await_args_list[1].args[0] == (
        "My BP is 140/90, what target?"
    )


# ===========================================================================
# 6. HTTP round trip (JSON + SSE) with session tokens
# ===========================================================================


def parse_sse(body: str) -> list[tuple[str, Any]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_fastapi_rest_and_sse_round_trip():
    with TestClient(create_app()) as client:
        created = client.post("/api/sessions")
        assert created.status_code == 201
        body = created.json()
        sid, token = body["session_id"], body["session_token"]
        headers = {"Authorization": f"Bearer {token}"}

        unauthorised = client.post(f"/api/sessions/{sid}/messages", json={"message": EMERGENCY})
        assert unauthorised.status_code == 401

        reply = client.post(
            f"/api/sessions/{sid}/messages", json={"message": EMERGENCY}, headers=headers
        )
        assert reply.status_code == 200
        data = reply.json()
        assert data["action"] == "ESCALATE" and data["session_id"] == sid
        assert data["explanation"]["gate2"]["source"] == "fastpath"

        stream = client.post(
            f"/api/sessions/{sid}/messages/stream",
            json={"message": "I have crushing chest pain right now"},
            headers=headers,
        )
        assert stream.status_code == 200
        events = parse_sse(stream.text)
        kinds = [k for k, _ in events]
        assert kinds[0] == "stage" and kinds[-1] == "done" and "meta" in kinds and "token" in kinds
        assert events[-1][1]["action"] == "ESCALATE"

        history = client.get(f"/api/sessions/{sid}/history", headers=headers)
        assert history.status_code == 200
        assert [m["role"] for m in history.json()["messages"]] == ["user", "assistant"] * 2

        health = client.get("/health")
        assert health.json()["postgres_connected"] is True
        assert health.json()["qdrant_connected"] is True
        assert health.json()["status"] in ("healthy", "degraded")


# ===========================================================================
# 7. Live turn with the real LLM (opt-in: pytest -m live)
# ===========================================================================


@pytest.mark.live
@pytest.mark.ollama
async def test_live_turn_with_real_llm(db_session, session_id, retriever):
    """One real turn through every component; checks shape, never the clinical wording."""
    orchestrator = PipelineOrchestrator(hybrid_retriever=retriever)
    response = await orchestrator.process_message(
        session_id,
        "I'm 62, my BP is around 150/95 and I take amlodipine 5 mg. What target should I aim for?",
        db=db_session,
    )
    await db_session.commit()
    assert response.action in set(ActionEnum)
    assert response.response_text
    assert response.message_id is not None
    exp = response.explanation
    assert exp.gate1 is not None and exp.gate1.source in ("llm", "fallback")
    assert exp.llm_calls, "LLM call statistics were not recorded"
    if response.action == ActionEnum.ANSWER:
        assert response.citations, "an ANSWER must carry at least one valid citation"
