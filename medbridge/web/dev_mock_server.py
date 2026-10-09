"""
Development mock of the MedBridge HTTP API for building and screenshotting the web UI.

Run:  python -m medbridge.web.dev_mock_server   (serves http://127.0.0.1:8100)

It implements the contract in docs/finalization_plan.md §3.7 with canned,
in-memory data and mounts the real SPA via ``mount_web``. Nothing here calls
an LLM, PostgreSQL or Qdrant. ALL content is synthetic placeholder text for
UI development — it is not evidence, not a result and must never be quoted
in the report. Responses are keyword-triggered:

* "chest pain"         -> ESCALATE (emergency fast-path, template, no LLM stages)
* "french revolution"  -> ABSTAIN  (out of scope, template)
* "dose"               -> SOFT-ASK (Gate 1 asks for missing context)
* "diet"               -> GENERALIZE
* "rate limit"         -> HTTP 429 with ``Retry-After: 8`` (exercises the countdown)
* anything else        -> ANSWER with two citations and a full explanation

Like the real server, a streamed turn keeps running and is saved when the
client stops reading (Stop button, closed tab), so the UI's "finishing the
previous answer" lock can be exercised.

Admin login: username ``admin`` / password ``admin`` (mock only). The same
strict Content-Security-Policy as production is applied to every response so
CSP violations surface during development.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import secrets
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from medbridge.web import mount_web

# Same policy as medbridge.privacy.middleware.CONTENT_SECURITY_POLICY.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; font-src 'self' data:; "
    "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)
STAGE_LABELS: dict[str, str] = {
    "privacy": "Protecting your privacy…",
    "emergency": "Checking for emergencies…",
    "load_state": "Loading your conversation…",
    "small_talk": "Reading your message…",
    "extraction": "Understanding your context…",
    "context_gate": "Checking what I know about you…",
    "retrieval": "Searching guidelines…",
    "rerank": "Ranking the evidence…",
    "evidence_gate": "Verifying evidence…",
    "generation": "Writing answer…",
    "verification": "Double-checking claims…",
    "persistence": "Saving securely…",
}
STAGES = list(STAGE_LABELS)
ADMIN_TOKEN = "mock-admin-token"
MODELS = {"gates": "qwen3:8b", "generator": "qwen3:8b"}

ESCALATE_TEXT = (
    "⚠️ IMPORTANT: Based on what you've described, this situation may require immediate "
    "medical attention. Please contact your healthcare provider, visit your nearest emergency "
    "room, or call emergency services right away.\n\nThis tool provides informational guidance "
    "only and cannot assess or manage medical emergencies."
)
ABSTAIN_TEXT = (
    "I don't have sufficient information in my clinical guidelines to safely address this "
    "specific question. It may be outside the scope of cardiovascular and blood pressure care.\n\n"
    "I recommend discussing this question directly with your healthcare provider."
)

DOCS: list[dict[str, Any]] = [
    {"guideline_id": "mock_aha_acc_2025", "title": "Sample guideline A — high blood pressure in adults",
     "org": "AHA/ACC", "year": 2025, "section": "BP classification", "page": 14},
    {"guideline_id": "mock_esc_2024", "title": "Sample guideline B — elevated BP and hypertension",
     "org": "ESC", "year": 2024, "section": "Out-of-office measurement", "page": 31},
    {"guideline_id": "mock_nice_ng136", "title": "Sample guideline C — hypertension in adults",
     "org": "NICE", "year": 2023, "section": "Lifestyle interventions", "page": 9},
    {"guideline_id": "mock_who_2021", "title": "Sample guideline D — pharmacological treatment",
     "org": "WHO", "year": 2021, "section": "Recommendations", "page": 22},
    {"guideline_id": "mock_esc_2024", "title": "Sample guideline B — elevated BP and hypertension",
     "org": "ESC", "year": 2024, "section": "Treatment targets", "page": 44},
]
EXCERPTS = [
    "[Synthetic placeholder excerpt] Blood pressure categories are defined by systolic and "
    "diastolic thresholds; stage 2 corresponds to a systolic value of 140 mmHg or higher or a "
    "diastolic value of 90 mmHg or higher.",
    "[Synthetic placeholder excerpt] A diagnosis should be based on repeated measurements, "
    "preferably confirmed with home or ambulatory blood pressure monitoring.",
    "[Synthetic placeholder excerpt] Lifestyle measures such as reducing salt, regular "
    "physical activity and limiting alcohol are recommended for people with raised blood pressure.",
    "[Synthetic placeholder excerpt] Medicine choice depends on age, other conditions and "
    "individual factors discussed with a clinician.",
    "[Synthetic placeholder excerpt] Treatment targets are individualised.",
]
SUGGESTED_PROMPTS = [
    "My blood pressure was 145/92 this morning. Is that high?",
    "What diet changes help lower blood pressure?",
    "What is the usual dose of amlodipine for me?",
    "I have chest pain and my BP is 190/125",
    "How should I measure my blood pressure at home?",
    "Tell me about the French Revolution",
]
CRISIS_ADVICE = (
    "[Mock] This reading is in the severe range. If you have symptoms such as chest pain or "
    "shortness of breath, call 112 now; otherwise re-measure after resting and contact your doctor."
)


@dataclass
class MockSession:
    """In-memory session."""

    token: str
    created_at: datetime
    messages: list[dict[str, Any]] = field(default_factory=list)
    readings: list[dict[str, Any]] = field(default_factory=list)
    medications: list[dict[str, str]] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    soft_ask_count: int = 0


SESSIONS: dict[str, MockSession] = {}
FEEDBACK: list[dict[str, Any]] = []
AUDIT: list[dict[str, Any]] = []


class MessageIn(BaseModel):
    """Chat message body."""

    message: str = Field(min_length=1, max_length=2000)


class FeedbackIn(BaseModel):
    """Feedback body."""

    message_id: str
    rating: int = Field(ge=-1, le=1)
    comment: str | None = Field(default=None, max_length=1000)


class BpIn(BaseModel):
    """Manual BP reading (same limits as the real BPEntryRequest)."""

    systolic: int = Field(ge=60, le=300)
    diastolic: int = Field(ge=30, le=200)
    pulse: int | None = Field(default=None, ge=20, le=250)
    measured_at: datetime | None = None


class LoginIn(BaseModel):
    """Admin credentials."""

    username: str
    password: str


def _now() -> datetime:
    return datetime.now(UTC)


BP_LABELS = {
    "normal": "Normal",
    "elevated": "Elevated",
    "stage1": "Stage 1 hypertension",
    "stage2": "Stage 2 hypertension",
    "crisis": "Severe / hypertensive crisis range",
}


def bp_category(systolic: int, diastolic: int) -> str:
    """ACC/AHA category key (same thresholds as medbridge.features.bp)."""
    if systolic >= 180 or diastolic >= 120:
        return "crisis"
    if systolic >= 140 or diastolic >= 90:
        return "stage2"
    if systolic >= 130 or diastolic >= 80:
        return "stage1"
    if systolic >= 120:
        return "elevated"
    return "normal"


def _reading_out(reading: dict[str, Any]) -> dict[str, Any]:
    """A stored reading in the ContextResponse / BPEntryResponse shape."""
    key = bp_category(reading["systolic"], reading["diastolic"])
    return {**reading, "category": key, "category_label": BP_LABELS[key]}


def _session(session_id: str, authorization: str | None) -> MockSession:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if authorization != f"Bearer {session.token}":
        raise HTTPException(status_code=401, detail="Invalid or missing session token")
    return session


def _snapshot(session: MockSession) -> dict[str, Any]:
    """Projected snapshot in the event-sourced shape used by the real projector."""
    return {
        "recent_bp_readings": session.readings[-5:],
        "current_medications": session.medications,
        "symptoms": session.symptoms,
        "demographics": {"age": 54} if session.readings else {},
        "lab_results": [],
    }


def _citation(i: int) -> dict[str, Any]:
    doc = DOCS[i]
    return {
        "marker": f"[{i + 1}]",
        "chunk_id": f"{doc['guideline_id']}_c{i + 1}",
        "source": f"{doc['org']} {doc['year']}",
        "section": doc["section"],
        "excerpt": EXCERPTS[i],
        "guideline_id": doc["guideline_id"],
        "title": doc["title"],
        "org": doc["org"],
        "year": doc["year"],
        "page": doc["page"],
        "url": None,
    }


def _retrieval(scores: list[float], cited: set[int]) -> list[dict[str, Any]]:
    return [
        {
            "rank": i + 1,
            "chunk_id": f"{doc['guideline_id']}_c{i + 1}",
            "guideline_id": doc["guideline_id"],
            "title": doc["title"],
            "org": doc["org"],
            "year": doc["year"],
            "section": doc["section"],
            "page": doc["page"],
            "score_rrf": round(1 / (60 + i + 1), 4),
            "score_rerank": scores[i],
            "excerpt": EXCERPTS[i][:160],
            "cited": i in cited,
        }
        for i, doc in enumerate(DOCS)
    ]


def _timings(plan: dict[str, str]) -> list[dict[str, Any]]:
    base = {
        "privacy": 1.2, "emergency": 0.4, "load_state": 6.0, "small_talk": 0.3,
        "extraction": 820.0, "context_gate": 610.0, "retrieval": 95.0, "rerank": 140.0,
        "evidence_gate": 690.0, "generation": 2350.0, "verification": 380.0, "persistence": 14.0,
    }
    return [
        {"stage": s, "ms": base[s] if status == "ok" else 0.0, "status": status}
        for s, status in plan.items()
    ]


def build_turn(message: str, session: MockSession) -> tuple[dict[str, Any], dict[str, str]]:
    """Return (MessageResponse-like dict without ids, stage plan) for a message."""
    text = message.lower()
    redactions = len(re.findall(r"\b\d{10}\b", message)) + len(re.findall(r"my name is \w+", text))
    plan = dict.fromkeys(STAGES, "ok")
    emergency: dict[str, Any] = {"matched": False, "rules": [], "latency_ms": 0.4}
    gate3: dict[str, Any] | None = None
    citations: list[dict[str, Any]] = []
    retrieval: list[dict[str, Any]] = []
    gate1: dict[str, Any] | None = None
    gate2: dict[str, Any] | None = None

    if "chest pain" in text:
        action = "ESCALATE"
        reply = ESCALATE_TEXT
        emergency = {"matched": True, "rules": ["chest_pain", "bp_crisis_with_symptoms"], "latency_ms": 0.6}
        plan.update(dict.fromkeys(STAGES[2:-1], "skipped"))
    elif "french revolution" in text:
        action = "ABSTAIN"
        reply = ABSTAIN_TEXT
        gate1 = {"action": "PROCEED", "rationale": "General question; no personal context is needed to search.", "source": "llm"}
        retrieval = _retrieval([-7.9, -8.4, -9.1, -9.6, -10.2], set())
        gate2 = {"action": "ABSTAIN", "rationale": "Retrieved passages are unrelated to the question; the topic is outside the cardiovascular scope.", "source": "llm"}
        plan.update(dict.fromkeys(["generation", "verification"], "skipped"))
    elif "dose" in text and session.soft_ask_count < 2:
        action = "SOFT-ASK"
        session.soft_ask_count += 1
        reply = (
            "I can share what the guidelines say, but to keep it safe I need a little more "
            "information first:\n\n1. Which **other medicines** do you take?\n2. Do you have any "
            "**kidney or heart conditions**?\n3. What is your **age**?"
        )
        gate1 = {
            "action": "SOFT-ASK",
            "rationale": "Dosing depends on age, kidney function and current medicines, which are not yet known.",
            "missing_fields": ["current_medications", "comorbidities", "age"],
            "source": "llm",
        }
        plan.update(dict.fromkeys(["retrieval", "rerank", "evidence_gate", "verification"], "skipped"))
    elif "diet" in text:
        action = "GENERALIZE"
        session.soft_ask_count = 0
        reply = (
            "Here is general guidance that applies to most adults with raised blood pressure [1]:\n\n"
            "- **Reduce salt** in cooking and processed foods\n"
            "- Eat plenty of **vegetables, fruit and whole grains**\n"
            "- Keep **alcohol** within recommended limits\n\n"
            "Your clinician can tailor these to you."
        )
        citations = [_citation(2)]
        citations[0]["marker"] = "[1]"
        gate1 = {"action": "PROCEED", "rationale": "Lifestyle question answerable without personal details.", "source": "llm"}
        retrieval = _retrieval([5.2, 3.1, 1.4, -0.8, -2.6], {2})
        gate2 = {"action": "GENERALIZE", "rationale": "Evidence supports general lifestyle advice but not a personalised plan.", "source": "llm"}
    else:
        action = "ANSWER"
        session.soft_ask_count = 0
        reading = re.search(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b", message)
        if reading:
            sys_v, dia_v = int(reading.group(1)), int(reading.group(2))
            session.readings.append({"systolic": sys_v, "diastolic": dia_v, "pulse": None,
                                     "recorded_at": _now().isoformat(), "source": "chat"})
            category = BP_LABELS[bp_category(sys_v, dia_v)].lower()
            lead = f"A reading of **{sys_v}/{dia_v} mmHg** falls in the **{category}** range [1]."
        else:
            lead = "Blood pressure categories are defined by systolic and diastolic thresholds [1]."
        reply = (
            f"{lead} A single reading is not a diagnosis — guidelines recommend confirming it with "
            "repeated measurements, ideally with home or ambulatory monitoring [2].\n\n"
            "**What you can do next**\n\n"
            "1. Measure again on several days, seated and rested\n"
            "2. Keep a log and share it with your clinician\n"
            "3. Seek care urgently if you develop symptoms such as chest pain or weakness"
        )
        citations = [_citation(0), _citation(1)]
        gate1 = {"action": "PROCEED", "rationale": "The question concerns interpreting a reading the patient provided.", "source": "llm"}
        retrieval = _retrieval([6.84, 5.12, 2.31, -0.42, -2.08], {0, 1})
        gate2 = {"action": "ANSWER", "rationale": "Passages 1 and 2 directly define the categories and how a diagnosis is confirmed.", "source": "llm"}
        gate3 = {
            "action": "ANSWER", "n_claims": 3, "n_unsupported": 0, "unsupported_fraction": 0.0,
            "sentences": [{"text": "Category statement", "cited": True, "supported": True, "entailment": 0.93, "markers": ["[1]"]}],
        }

    for drug in ("amlodipine", "lisinopril", "losartan", "metoprolol"):
        if drug in text and all(m["drug"] != drug for m in session.medications):
            session.medications.append({"drug": drug, "dosage": "5 mg", "frequency": "once daily"})
    if "headache" in text and "headache" not in session.symptoms:
        session.symptoms.append("headache")

    timings = _timings(plan)
    explanation = {
        "gate1": gate1,
        "gate2": gate2,
        "gate3": gate3,
        "emergency": emergency,
        "retrieval": retrieval,
        "timings": timings,
        "total_ms": round(sum(t["ms"] for t in timings), 1),
        "speculative_retrieval": (
            {"launched": True, "used": action != "SOFT-ASK", "discarded": action == "SOFT-ASK"} if gate1 else None
        ),
        "citation_validation": {"valid": len(citations), "stripped": 0} if citations else None,
        "redactions": redactions,
        "redaction_categories": {"PHONE": redactions} if redactions else {},
        "models": MODELS if action not in ("ESCALATE",) else {},
        "llm_calls": (
            [
                {"call_name": "context_extractor", "model": MODELS["gates"], "latency_ms": 820, "prompt_tokens": 612, "completion_tokens": 88, "cached": False},
                {"call_name": "context_gate", "model": MODELS["gates"], "latency_ms": 610, "prompt_tokens": 540, "completion_tokens": 41, "cached": False},
            ]
            if gate1
            else []
        ),
        "regenerated": False,
    }
    response = {
        "response_text": reply,
        "action": action,
        "citations": citations,
        "soft_ask_count": session.soft_ask_count,
        "redaction_count": redactions,
        "explanation": explanation,
    }
    return response, plan


def _record(session_id: str, session: MockSession, message: str, turn: dict[str, Any], *, regenerate: bool) -> dict[str, Any]:
    """Persist a turn in memory and return the full MessageResponse dict."""
    now = _now()
    user_id = str(uuid.uuid4())
    assistant_id = str(uuid.uuid4())
    if regenerate and session.messages and session.messages[-1]["role"] == "assistant":
        session.messages.pop()
    elif not regenerate:
        session.messages.append({"role": "user", "content": message, "action": None, "citations": [], "timestamp": now.isoformat(), "message_id": user_id})
    turn["explanation"]["regenerated"] = regenerate
    full = {
        "session_id": session_id,
        "timestamp": now.isoformat(),
        "message_id": assistant_id,
        "user_message_id": user_id,
        **turn,
    }
    session.messages.append({
        "role": "assistant", "content": turn["response_text"], "action": turn["action"],
        "citations": turn["citations"], "timestamp": now.isoformat(), "message_id": assistant_id,
        "explanation": turn["explanation"],
    })
    expl = turn["explanation"]
    AUDIT.append({
        "audit_id": str(uuid.uuid4()), "session_ref": session_id[:8],
        "created_at": now.isoformat(), "final_action": turn["action"],
        "gate_1_action": (expl["gate1"] or {}).get("action"), "gate_2_action": (expl["gate2"] or {}).get("action"),
        "latency_ms": round(expl["total_ms"]), "evidence_chunks": len(expl["retrieval"]),
        "request_preview": message[:160], "response_preview": turn["response_text"][:160],
        "erased": False, "timings": expl["timings"],
    })
    return full


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# Strong references to turns still running after their client disconnected.
_RUNNING_TURNS: set[asyncio.Task[None]] = set()
RATE_LIMIT_TRIGGER = "rate limit"
RATE_LIMIT_RETRY_AFTER = 8


def _rate_limit_check(message: str) -> None:
    """429 + Retry-After for messages containing the mock trigger phrase."""
    if RATE_LIMIT_TRIGGER in message.lower():
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Please wait {RATE_LIMIT_RETRY_AFTER} s and try again.",
            headers={"Retry-After": str(RATE_LIMIT_RETRY_AFTER)},
        )


async def stream_turn(session_id: str, session: MockSession, message: str, *, regenerate: bool, delay: float) -> AsyncIterator[str]:
    """Stream a turn that, like the real server, finishes and is saved even if the client leaves."""
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def produce() -> None:
        try:
            async for event in _turn_events(session_id, session, message, regenerate=regenerate, delay=delay):
                queue.put_nowait(event)
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(produce())
    _RUNNING_TURNS.add(task)
    task.add_done_callback(_RUNNING_TURNS.discard)
    while (event := await queue.get()) is not None:
        yield event


async def _turn_events(session_id: str, session: MockSession, message: str, *, regenerate: bool, delay: float) -> AsyncIterator[str]:
    """Emit stage events, the answer as token chunks, then ``done``."""
    turn, plan = build_turn(message, session)
    for stage, status in plan.items():
        if stage == "persistence":
            yield _sse("stage", {"stage": stage, "status": "start", "label": STAGE_LABELS[stage]})
            full = _record(session_id, session, message, turn, regenerate=regenerate)
            yield _sse("stage", {"stage": stage, "status": "end", "label": STAGE_LABELS[stage]})
            continue
        if status == "skipped":
            yield _sse("stage", {"stage": stage, "status": "skipped", "label": STAGE_LABELS[stage]})
            continue
        yield _sse("stage", {"stage": stage, "status": "start", "label": STAGE_LABELS[stage]})
        await asyncio.sleep(delay * (4 if stage in ("extraction", "context_gate", "evidence_gate", "generation") else 1))
        yield _sse("stage", {"stage": stage, "status": "end", "label": STAGE_LABELS[stage]})
    yield _sse("meta", {"action": full["action"], "message_id": full["message_id"],
                        "redaction_count": full["redaction_count"]})
    for chunk in re.findall(r"\S+\s*|\s+", full["response_text"]):
        yield _sse("token", {"text": chunk})
        await asyncio.sleep(delay / 6)
    yield _sse("done", full)


def _seed_audit() -> None:
    """Synthetic audit rows so the dashboard has something to draw (mock only)."""
    rng = random.Random(42)
    actions = ["ANSWER"] * 9 + ["GENERALIZE"] * 5 + ["SOFT-ASK"] * 4 + ["ABSTAIN"] * 3 + ["ESCALATE"] * 2
    start = _now() - timedelta(hours=6)
    for i in range(46):
        action = rng.choice(actions)
        latency = {"ESCALATE": 12, "ABSTAIN": 2600, "SOFT-ASK": 2900}.get(action, 4800) * rng.uniform(0.7, 1.5)
        AUDIT.append({
            "audit_id": str(uuid.UUID(int=rng.getrandbits(128))),
            "session_ref": f"{rng.getrandbits(32):08x}",
            "created_at": (start + timedelta(minutes=7 * i)).isoformat(),
            "final_action": action,
            "gate_1_action": None if action == "ESCALATE" else ("SOFT-ASK" if action == "SOFT-ASK" else "PROCEED"),
            "gate_2_action": None if action in ("ESCALATE", "SOFT-ASK") else action,
            "latency_ms": round(latency),
            "evidence_chunks": 0 if action in ("ESCALATE", "SOFT-ASK") else 5,
            "request_preview": "[synthetic] My BP was 150/95, my name is [NAME], what should I do?",
            "response_preview": f"[synthetic] {action.title()} response preview…",
            "erased": i % 17 == 0,
            "timings": [],
        })
    for i in range(14):
        FEEDBACK.append({"rating": 1 if i % 5 else -1, "comment": None})


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 1)


def create_mock_app(*, stream_delay: float = 0.12) -> FastAPI:
    """Build the mock application (``stream_delay`` = seconds per pipeline stage)."""
    app = FastAPI(title="MedBridge UI mock", docs_url=None, redoc_url=None, openapi_url=None)
    if not AUDIT:
        _seed_audit()

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Response:
        response: Response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/health")
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "healthy", "postgres_connected": True, "qdrant_connected": True,
                "timestamp": _now().isoformat(), "ollama_reachable": True, "local_only_mode": True}

    @app.get("/api/config/public")
    async def public_config() -> dict[str, Any]:
        return {
            "app_name": "MedBridge",
            "version": "mock",
            "emergency_number": "112",
            "ambulance_number": "108",
            "emergency_region_label": "India",
            "local_only_mode": True,
            "models": MODELS,
            "disclaimer": "MedBridge gives general, guideline-based information (mock).",
            "suggested_prompts": SUGGESTED_PROMPTS,
            "corpus": {"documents": len(DOCS), "items": [
                {"title": d["title"], "org": d["org"], "year": d["year"]} for d in DOCS
            ]},
        }

    @app.post("/api/sessions", status_code=201)
    async def create_session() -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        now = _now()
        SESSIONS[session_id] = MockSession(token=secrets.token_urlsafe(24), created_at=now)
        return {"session_id": session_id, "created_at": now.isoformat(),
                "session_token": SESSIONS[session_id].token,
                "expires_at": (now + timedelta(days=7)).isoformat()}

    @app.get("/api/sessions/{session_id}/history")
    async def history(session_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        session = _session(session_id, authorization)
        return {"session_id": session_id, "messages": session.messages}

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def erase(session_id: str, authorization: str | None = Header(default=None)) -> Response:
        _session(session_id, authorization)
        del SESSIONS[session_id]
        return Response(status_code=204)

    @app.post("/api/sessions/{session_id}/messages")
    async def send(session_id: str, body: MessageIn, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        session = _session(session_id, authorization)
        _rate_limit_check(body.message)
        turn, _plan = build_turn(body.message, session)
        return _record(session_id, session, body.message, turn, regenerate=False)

    @app.post("/api/sessions/{session_id}/messages/stream")
    async def send_stream(session_id: str, body: MessageIn, authorization: str | None = Header(default=None)) -> StreamingResponse:
        session = _session(session_id, authorization)
        _rate_limit_check(body.message)
        return StreamingResponse(
            stream_turn(session_id, session, body.message, regenerate=False, delay=stream_delay),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/sessions/{session_id}/regenerate", response_model=None)
    async def regenerate(session_id: str, stream: bool = Query(default=False),
                         authorization: str | None = Header(default=None)) -> StreamingResponse | dict[str, Any]:
        session = _session(session_id, authorization)
        last = next((m["content"] for m in reversed(session.messages) if m["role"] == "user"), None)
        if last is None:
            raise HTTPException(status_code=409, detail="Nothing to regenerate")
        if stream:
            return StreamingResponse(stream_turn(session_id, session, last, regenerate=True, delay=stream_delay),
                                     media_type="text/event-stream")
        turn, _plan = build_turn(last, session)
        return _record(session_id, session, last, turn, regenerate=True)

    @app.post("/api/sessions/{session_id}/feedback", status_code=201)
    async def feedback(session_id: str, body: FeedbackIn, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _session(session_id, authorization)
        FEEDBACK.append({"rating": body.rating, "comment": body.comment})
        return {"feedback_id": str(uuid.uuid4())}

    @app.get("/api/sessions/{session_id}/context")
    async def context(session_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        session = _session(session_id, authorization)
        return {"session_id": session_id, "snapshot": _snapshot(session),
                "bp_readings": [_reading_out(r) for r in session.readings],
                "counts": {"BP_READING": len(session.readings), "total_events": len(session.readings)}}

    @app.post("/api/sessions/{session_id}/bp", status_code=201)
    async def add_bp(session_id: str, body: BpIn, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        session = _session(session_id, authorization)
        if body.systolic <= body.diastolic:
            raise HTTPException(status_code=422, detail="Systolic must be greater than diastolic")
        when = body.measured_at or _now()
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        reading = {"systolic": body.systolic, "diastolic": body.diastolic, "pulse": body.pulse,
                   "recorded_at": when.isoformat(), "source": "manual"}
        session.readings.append(reading)
        session.readings.sort(key=lambda r: datetime.fromisoformat(r["recorded_at"]))
        out = _reading_out(reading)
        crisis = out["category"] == "crisis"
        return {"snapshot": _snapshot(session), "reading": out, "crisis": crisis,
                "emergency_numbers": {"emergency": "112", "ambulance": "108", "region": "India"},
                "advice": CRISIS_ADVICE if crisis else None}

    @app.get("/api/sessions/{session_id}/summary", response_model=None)
    async def summary(session_id: str, format: str = Query(default="md"),
                      authorization: str | None = Header(default=None)) -> PlainTextResponse | dict[str, Any]:
        session = _session(session_id, authorization)
        lines = ["# Clinician summary (mock)", "", "_Synthetic development data._", "", "## Blood pressure"]
        lines += [f"- {r['recorded_at'][:16]}: {r['systolic']}/{r['diastolic']} mmHg" for r in session.readings] or ["- none"]
        lines += ["", "## Questions and actions"]
        lines += [f"- {m['action']}: {m['content'][:80]}" for m in session.messages if m["role"] == "assistant"] or ["- none"]
        text = "\n".join(lines) + "\n"
        if format == "json":
            return {"markdown": text}
        return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")

    @app.post("/api/admin/login")
    async def admin_login(body: LoginIn) -> dict[str, str]:
        if body.username != "admin" or body.password != "admin":
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"token": ADMIN_TOKEN, "expires_at": (_now() + timedelta(hours=8)).isoformat()}

    def _require_admin(authorization: str | None) -> None:
        if authorization != f"Bearer {ADMIN_TOKEN}":
            raise HTTPException(status_code=401, detail="Admin token required")

    @app.get("/api/metrics")
    async def metrics(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _require_admin(authorization)
        latencies = [float(a["latency_ms"]) for a in AUDIT]
        by_action: dict[str, list[float]] = {}
        gate1: dict[str, int] = {}
        gate2: dict[str, int] = {}
        for row in AUDIT:
            by_action.setdefault(row["final_action"], []).append(float(row["latency_ms"]))
            if row["gate_1_action"]:
                gate1[row["gate_1_action"]] = gate1.get(row["gate_1_action"], 0) + 1
            if row["gate_2_action"]:
                gate2[row["gate_2_action"]] = gate2.get(row["gate_2_action"], 0) + 1
        stage_base = _timings(dict.fromkeys(STAGES, "ok"))
        up = sum(1 for f in FEEDBACK if f["rating"] == 1)
        down = sum(1 for f in FEEDBACK if f["rating"] == -1)

        def stats(values: list[float]) -> dict[str, Any]:
            return {"count": len(values), "p50_ms": _percentile(values, 0.5), "p95_ms": _percentile(values, 0.95)}

        return {
            "generated_at": _now().isoformat(),
            "turns": len(AUDIT),
            "window": 500,
            "final_actions": {k: len(v) for k, v in by_action.items()},
            "gate_1": gate1,
            "gate_2": gate2,
            "gate_sources": {"gate1": {"llm": sum(gate1.values()) - 1, "fallback": 1}, "gate2": {"llm": sum(gate2.values())}},
            "latency": stats(latencies),
            "latency_by_action": {k: stats(v) for k, v in by_action.items()},
            "stage_latency": {t["stage"]: {"count": len(AUDIT), "p50_ms": t["ms"], "p95_ms": round(t["ms"] * 1.6, 1)} for t in stage_base},
            "fallbacks": {"gate1": 1, "gate2": 0, "generator": 0, "extractor": 0},
            "routing": {"loop_breaker": 2, "fastpath": len(by_action.get("ESCALATE", [])), "small_talk": 3},
            "citations": {"downgraded_to_generalize": 1, "stripped_markers": 2},
            "speculative_retrieval": {"launched": 40, "discarded": 6},
            "redaction": {"turns_with_redactions": 5, "items": 9},
            "feedback": {"count": up + down, "thumbs_up": up, "thumbs_down": down,
                         "positive_rate": round(up / (up + down), 3) if up + down else None},
        }

    @app.get("/api/admin/audit")
    async def audit(limit: int = Query(default=50, ge=1, le=500), authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _require_admin(authorization)
        rows = [{k: v for k, v in row.items() if k != "timings"} for row in reversed(AUDIT[-limit:])]
        return {"items": rows, "count": len(rows)}

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, headers=exc.headers, content={
            "error_code": "RATE_LIMITED" if exc.status_code == 429 else f"HTTP_{exc.status_code}",
            "message": str(exc.detail),
            "safe_fallback": "This request could not be completed. Please start a new conversation or try again.",
        })

    mount_web(app)
    return app


def main() -> None:
    """Serve the mock on 127.0.0.1:8100."""
    import uvicorn

    uvicorn.run(create_mock_app(), host="127.0.0.1", port=8100, log_level="warning")


if __name__ == "__main__":
    main()
