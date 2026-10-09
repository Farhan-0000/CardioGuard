"""A small in-memory imitation of the MedBridge HTTP API for script tests (httpx.MockTransport)."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

_SESSION_PATH = re.compile(r"^/api/sessions/(?P<sid>[^/]+)(?P<rest>/.*)?$")


@dataclass
class FakeMedBridge:
    """Routes requests like the real API; behaviour knobs let tests inject faults."""

    actions: dict[str, str] = field(
        default_factory=lambda: {
            "chest pain": "ESCALATE",
            "French Revolution": "ABSTAIN",
            "athlete": "ABSTAIN",
            "higher dose": "SOFT-ASK",
            "lisinopril 10 mg": "ANSWER",
            "dietary": "GENERALIZE",
            "lifestyle": "GENERALIZE",
            "hello": "GENERALIZE",
        }
    )
    redact: bool = True
    health_status: str = "healthy"
    session_429_first: int = 0
    message_429_every: int = 0
    stream_ok: bool = True
    explanation: bool = True
    sessions: dict[str, str] = field(default_factory=dict)
    erased: set[str] = field(default_factory=set)
    requests: list[tuple[str, str]] = field(default_factory=list)
    message_count: int = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def _action(self, message: str) -> str:
        return next((a for k, a in self.actions.items() if k.lower() in message.lower()), "ANSWER")

    def _reply(self, sid: str, message: str) -> dict[str, Any]:
        action = self._action(message)
        body: dict[str, Any] = {
            "session_id": sid,
            "response_text": f"reply to {message[:20]}",
            "action": action,
            "citations": [{"marker": "[1]", "source": "AHA/ACC 2025"}]
            if action == "ANSWER"
            else [],
            "soft_ask_count": 1 if action == "SOFT-ASK" else 0,
            "timestamp": "2026-10-09T00:00:00Z",
            "message_id": str(uuid.uuid4()),
            "redaction_count": 2 if (self.redact and "John Smith" in message) else 0,
        }
        if self.explanation:
            body["explanation"] = {
                "timings": [
                    {"stage": "privacy", "ms": 0.4, "status": "ok"},
                    {"stage": "emergency", "ms": 1.1, "status": "ok"},
                    {"stage": "extraction", "ms": 900.0, "status": "ok"},
                    {"stage": "retrieval", "ms": 80.0, "status": "discarded"},
                    {"stage": "generation", "ms": 0.0, "status": "skipped"},
                    {"stage": "persistence", "ms": 5.0, "status": "ok"},
                ],
                "total_ms": 1010.5,
                "llm_calls": [{"call_name": "context_extractor"}],
            }
        return body

    def handle(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        self.requests.append((method, path))
        if path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": self.health_status,
                    "postgres_connected": True,
                    "qdrant_connected": True,
                },
            )
        if path in ("/api/sessions", "/api/sessions/") and method == "POST":
            if self.session_429_first > 0:
                self.session_429_first -= 1
                return httpx.Response(429, headers={"Retry-After": "2"}, json={"error": "rate"})
            sid = str(uuid.uuid4())
            token = f"tok-{sid}"
            self.sessions[sid] = token
            return httpx.Response(
                201,
                json={
                    "session_id": sid,
                    "session_token": token,
                    "created_at": "2026-10-09T00:00:00Z",
                },
            )
        match = _SESSION_PATH.match(path)
        if not match:
            return httpx.Response(404, json={"detail": "not found"})
        sid, rest = match.group("sid"), match.group("rest") or ""
        auth = request.headers.get("authorization", "")
        if not auth:
            return httpx.Response(401, json={"detail": "token required"})
        if auth != f"Bearer {self.sessions.get(sid)}":
            return httpx.Response(403, json={"detail": "forbidden"})
        if sid in self.erased:
            return httpx.Response(410, json={"error_code": "SESSION_ERASED"})
        if method == "DELETE" and rest == "":
            self.erased.add(sid)
            return httpx.Response(204)
        if method == "POST" and rest == "/messages":
            self.message_count += 1
            if self.message_429_every and self.message_count % self.message_429_every == 0:
                return httpx.Response(429, headers={"Retry-After": "30"}, json={"error": "rate"})
            message = json.loads(request.content)["message"]
            return httpx.Response(200, json=self._reply(sid, message))
        if method == "POST" and rest == "/messages/stream":
            if not self.stream_ok:
                return httpx.Response(500, json={"error": "boom"})
            message = json.loads(request.content)["message"]
            done = self._reply(sid, message)
            text = "".join(
                [
                    'event: stage\ndata: {"stage": "privacy", "status": "start"}\n\n',
                    'event: stage\ndata: {"stage": "privacy", "status": "end"}\n\n',
                    f"event: meta\ndata: {json.dumps({'action': done['action']})}\n\n",
                    'event: token\ndata: {"text": "reply "}\n\n',
                    f"event: done\ndata: {json.dumps(done)}\n\n",
                ]
            )
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        if method == "POST" and rest == "/feedback":
            return httpx.Response(201, json={"feedback_id": str(uuid.uuid4())})
        if method == "GET" and rest == "/context":
            return httpx.Response(200, json={"session_id": sid, "snapshot": {}, "bp_readings": []})
        if method == "GET" and rest.startswith("/summary"):
            return httpx.Response(200, json={"session_ref": sid[:8]})
        if method == "GET" and rest == "/history":
            return httpx.Response(200, json={"session_id": sid, "messages": []})
        return httpx.Response(404, json={"detail": "not found"})
