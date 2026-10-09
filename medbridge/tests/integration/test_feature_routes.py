"""
Feature routes: feedback, context card, manual BP entry, clinician summary.

Hermetic tests mock the turn store; the ``postgres`` test drives the full
create → BP → context → feedback → summary → erase flow over HTTP against the
isolated test database (one shared, rolled-back DB session).
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.api.middleware.error_handler import register_error_handlers
from medbridge.api.routes.features import features_router
from medbridge.api.routes.sessions import sessions_router
from medbridge.api.schemas.enums import ActionEnum
from medbridge.db.connection import get_session
from medbridge.privacy.tokens import issue_session_token
from medbridge.state import turn_store
from medbridge.state.session_manager import MessageNotFoundError
from medbridge.state.turn_store import EventRow, SessionState, TurnRecord


def _auth(session_id: uuid.UUID) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_session_token(session_id)[0]}"}


def _app(db: Any) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(sessions_router, prefix="/api")
    app.include_router(features_router, prefix="/api")

    async def override() -> AsyncIterator[Any]:
        yield db

    app.dependency_overrides[get_session] = override
    return app


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=_app(MagicMock()))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Hermetic
# ---------------------------------------------------------------------------

async def test_all_feature_routes_require_the_session_token(client: httpx.AsyncClient) -> None:
    sid = uuid.uuid4()
    for method, path in (("POST", "feedback"), ("GET", "context"), ("POST", "bp"),
                         ("GET", "summary")):
        resp = await client.request(method, f"/api/sessions/{sid}/{path}", json={})
        assert resp.status_code == 401, path


async def test_feedback(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    sid, mid, fid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    save = AsyncMock(return_value=fid)
    monkeypatch.setattr(turn_store, "save_feedback", save)
    resp = await client.post(f"/api/sessions/{sid}/feedback", headers=_auth(sid),
                             json={"message_id": str(mid), "rating": -1, "comment": "too long"})
    assert resp.status_code == 201 and resp.json() == {"feedback_id": str(fid)}
    assert save.await_args.args[1:] == (sid, mid, -1, "too long")

    bad = await client.post(f"/api/sessions/{sid}/feedback", headers=_auth(sid),
                            json={"message_id": str(mid), "rating": 5})
    assert bad.status_code == 422

    save.side_effect = MessageNotFoundError(mid)
    missing = await client.post(f"/api/sessions/{sid}/feedback", headers=_auth(sid),
                                json={"message_id": str(mid), "rating": 1})
    assert missing.status_code == 404 and missing.json()["error_code"] == "MESSAGE_NOT_FOUND"


async def test_context(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = uuid.uuid4()
    now = datetime.now(UTC)
    events = [
        EventRow(uuid.uuid4(), "BP_READING", {"systolic": 150, "diastolic": 95}, now),
        EventRow(uuid.uuid4(), "MEDICATION_ADDED", {"drug_name": "Telma"}, now),
    ]
    monkeypatch.setattr(turn_store, "load_session_state",
                        AsyncMock(return_value=SessionState(snapshot={"symptoms": ["x"]})))
    monkeypatch.setattr(turn_store, "list_events", AsyncMock(return_value=events))
    resp = await client.get(f"/api/sessions/{sid}/context", headers=_auth(sid))
    assert resp.status_code == 200
    body = resp.json()
    assert body["snapshot"] == {"symptoms": ["x"]}
    assert body["bp_readings"][0]["category"] == "stage2"
    assert body["bp_readings"][0]["systolic"] == 150
    assert body["counts"]["BP_READING"] == 1 and body["counts"]["total_events"] == 2
    assert body["counts"]["LAB_RESULT"] == 0
    assert [c["key"] for c in body["categories"]][-1] == "crisis"


async def test_context_profile_and_medications(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = uuid.uuid4()
    snapshot = {
        "demographics": {"age": 64, "pregnancy_status": "not pregnant",
                         "conditions": ["diabetes"]},
        "current_medications": [{"drug": "telmisartan", "dosage": "80 mg", "frequency": "",
                                 "brand_name": "Telma", "drug_class": "ARB",
                                 "history": [{"dosage": "40 mg", "frequency": ""}]}],
        "discontinued_medications": [{"drug": "lisinopril", "dosage": "10 mg",
                                      "reason": "dry cough"}],
    }
    monkeypatch.setattr(turn_store, "load_session_state",
                        AsyncMock(return_value=SessionState(snapshot=snapshot)))
    monkeypatch.setattr(turn_store, "list_events", AsyncMock(return_value=[]))
    body = (await client.get(f"/api/sessions/{sid}/context", headers=_auth(sid))).json()
    assert body["snapshot"] == snapshot
    assert body["profile"] == {"age": 64, "sex": None, "pregnancy_status": "not pregnant",
                               "conditions": ["diabetes"]}
    assert body["medications"] == [{
        "drug": "telmisartan", "brand_name": "Telma", "drug_class": "ARB", "dosage": "80 mg",
        "frequency": None, "history": [{"dosage": "40 mg"}], "reason": None,
    }]
    assert body["discontinued_medications"][0]["reason"] == "dry cough"


async def test_bp_entry_uses_bp_parse_plausibility(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from medbridge.api.routes import features as features_module

    sid = uuid.uuid4()
    monkeypatch.setattr(features_module, "is_plausible_bp", lambda s, d: False)
    resp = await client.post(f"/api/sessions/{sid}/bp", headers=_auth(sid),
                             json={"systolic": 128, "diastolic": 82})
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"systolic": 59, "diastolic": 40},
        {"systolic": 301, "diastolic": 90},
        {"systolic": 120, "diastolic": 29},
        {"systolic": 120, "diastolic": 201},
        {"systolic": 90, "diastolic": 90},
        {"systolic": 120, "diastolic": 80, "pulse": 300},
        {"systolic": 120, "diastolic": 80,
         "measured_at": (datetime.now(UTC) + timedelta(days=1)).isoformat()},
    ],
)
async def test_bp_entry_validation(client: httpx.AsyncClient, payload: dict[str, Any]) -> None:
    sid = uuid.uuid4()
    resp = await client.post(f"/api/sessions/{sid}/bp", headers=_auth(sid), json=payload)
    assert resp.status_code == 422


async def test_bp_entry_crisis_flag(client: httpx.AsyncClient,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    sid = uuid.uuid4()
    append = AsyncMock(return_value={"recent_bp_readings": [{"systolic": 185}]})
    monkeypatch.setattr(turn_store, "append_manual_events", append)
    resp = await client.post(f"/api/sessions/{sid}/bp", headers=_auth(sid),
                             json={"systolic": 185, "diastolic": 110, "pulse": 90,
                                   "measured_at": "2026-10-09T07:00:00"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["crisis"] is True and body["reading"]["category"] == "crisis"
    assert body["reading"]["recorded_at"] == "2026-10-09T07:00:00+00:00"
    assert set(body["emergency_numbers"]) == {"emergency", "ambulance", "region"}
    assert body["emergency_numbers"]["emergency"] in body["advice"]
    event = append.await_args.args[2][0]
    assert event.event_type == "BP_READING"
    assert event.payload == {"systolic": 185, "diastolic": 110, "pulse": 90,
                             "measured_at": "2026-10-09T07:00:00+00:00", "source": "manual"}

    append.return_value = {}
    normal = await client.post(f"/api/sessions/{sid}/bp", headers=_auth(sid),
                               json={"systolic": 118, "diastolic": 76})
    assert normal.json()["crisis"] is False and normal.json()["advice"] is None


# ---------------------------------------------------------------------------
# PostgreSQL end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture
async def db() -> AsyncIterator[AsyncSession]:
    from medbridge.db.connection import get_sessionmaker

    async with get_sessionmaker()() as session:
        yield session
        await session.rollback()


@pytest.mark.postgres
async def test_feature_flow_end_to_end(db: AsyncSession) -> None:
    transport = httpx.ASGITransport(app=_app(db))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        created = (await http.post("/api/sessions")).json()
        sid = uuid.UUID(created["session_id"])
        auth = {"Authorization": f"Bearer {created['session_token']}"}

        _, answer_id = await turn_store.persist_turn(db, TurnRecord(
            session_id=sid, user_message="What is a normal BP?",
            assistant_message="Below 120/80 mmHg is normal [1].", final_action=ActionEnum.ANSWER,
            citations=[{"marker": "[1]", "chunk_id": "c1", "source": "AHA/ACC 2025",
                        "section": "Table 4", "excerpt": "...", "page": 18}],
            trace={"gate1": {"action": "PROCEED", "rationale": "general question"}},
        ))

        bp = await http.post(f"/api/sessions/{sid}/bp", headers=auth,
                             json={"systolic": 146, "diastolic": 92, "pulse": 74})
        assert bp.status_code == 201 and bp.json()["reading"]["category"] == "stage2"

        context = (await http.get(f"/api/sessions/{sid}/context", headers=auth)).json()
        assert [r["source"] for r in context["bp_readings"]] == ["manual"]
        assert context["snapshot"]["recent_bp_readings"][0]["systolic"] == 146

        feedback = await http.post(f"/api/sessions/{sid}/feedback", headers=auth,
                                   json={"message_id": str(answer_id), "rating": 1})
        assert feedback.status_code == 201

        history = (await http.get(f"/api/sessions/{sid}/history", headers=auth)).json()
        assert history["messages"][1]["message_id"] == str(answer_id)
        assert history["messages"][1]["explanation"]["gate1"]["rationale"] == "general question"

        md = await http.get(f"/api/sessions/{sid}/summary", headers=auth)
        assert md.status_code == 200 and md.headers["content-type"].startswith("text/markdown")
        assert "attachment;" in md.headers["content-disposition"]
        assert "146/92" in md.text and "Table 4, p. 18" in md.text
        summary = (await http.get(f"/api/sessions/{sid}/summary?format=json",
                                  headers=auth)).json()
        assert summary["exchanges"][0]["action"] == "ANSWER"

        assert (await http.delete(f"/api/sessions/{sid}", headers=auth)).status_code == 204
        for path in ("history", "context", "summary"):
            gone = await http.get(f"/api/sessions/{sid}/{path}", headers=auth)
            assert gone.status_code == 410, path
        late_bp = await http.post(f"/api/sessions/{sid}/bp", headers=auth,
                                  json={"systolic": 130, "diastolic": 85})
        assert late_bp.status_code == 410
