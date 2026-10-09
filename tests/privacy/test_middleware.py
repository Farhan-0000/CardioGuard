"""Security middleware: headers/CSP, CORS, body-size limit, rate limits, SSE passthrough."""
from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.types import Message

from medbridge.api.middleware.error_handler import register_error_handlers
from medbridge.config import Settings
from medbridge.privacy.middleware import (
    CONTENT_SECURITY_POLICY,
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RateRule,
    register_security_middleware,
)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "CORS_ORIGINS": ["http://127.0.0.1:8000"],
        "MAX_REQUEST_BYTES": 1024,
        "RATE_LIMIT_PER_MINUTE": 3,
    }
    base.update(overrides)
    return Settings(**base)


def _build_app(settings: Settings) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    register_security_middleware(app, settings)

    @app.get("/api/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/page")
    async def page() -> dict[str, str]:
        return {"page": "yes"}

    @app.post("/api/echo")
    async def echo(request: Request) -> dict[str, int]:
        return {"bytes": len(await request.body())}

    @app.post("/api/sessions")
    async def create() -> dict[str, str]:
        return {"session": "new"}

    @app.post("/api/sessions/{sid}/messages")
    async def message(sid: str) -> dict[str, str]:
        return {"sid": sid}

    @app.post("/api/admin/login")
    async def login() -> dict[str, str]:
        return {"token": "t"}

    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_build_app(_settings()), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Headers & CORS
# ---------------------------------------------------------------------------

def test_security_headers_on_every_response(client: TestClient) -> None:
    for path in ("/api/ping", "/page", "/does-not-exist"):
        resp = client.get(path)
        assert resp.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["referrer-policy"] == "no-referrer"
        assert resp.headers["x-frame-options"] == "DENY"
        assert resp.headers["permissions-policy"] == "camera=(), microphone=(), geolocation=()"
        assert resp.headers["cross-origin-opener-policy"] == "same-origin"
    assert "frame-ancestors 'none'" in CONTENT_SECURITY_POLICY
    assert "script-src 'self';" in CONTENT_SECURITY_POLICY


def test_api_responses_are_not_cached(client: TestClient) -> None:
    assert client.get("/api/ping").headers["cache-control"] == "no-store"
    assert "cache-control" not in client.get("/page").headers


def test_docs_get_a_docs_specific_csp(client: TestClient) -> None:
    csp = client.get("/docs").headers["content-security-policy"]
    assert "https://cdn.jsdelivr.net" in csp and "frame-ancestors 'none'" in csp


def test_cors_allows_only_configured_origins(client: TestClient) -> None:
    preflight = {
        "Origin": "http://127.0.0.1:8000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Authorization, Content-Type",
    }
    ok = client.options("/api/echo", headers=preflight)
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "http://127.0.0.1:8000"
    assert "access-control-allow-credentials" not in ok.headers
    assert "DELETE" in ok.headers["access-control-allow-methods"]
    evil = client.options("/api/echo", headers={**preflight, "Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in evil.headers


# ---------------------------------------------------------------------------
# Body size limit
# ---------------------------------------------------------------------------

def test_body_size_limit_returns_413_json(client: TestClient) -> None:
    resp = client.post("/api/echo", content=b"x" * 2048,
                       headers={"Content-Type": "application/octet-stream"})
    assert resp.status_code == 413
    body = resp.json()
    assert body["error_code"] == "PAYLOAD_TOO_LARGE"
    assert body["safe_fallback"]
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert client.post("/api/echo", content=b"x" * 512).json() == {"bytes": 512}


async def _run_asgi(app: Any, scope: dict[str, Any], chunks: list[bytes]) -> list[Message]:
    messages = [
        {"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
        for i, c in enumerate(chunks)
    ]
    sent: list[Message] = []

    async def receive() -> Message:
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope, receive, send)
    return sent


def _scope(method: str, path: str, headers: list[tuple[bytes, bytes]] | None = None,
           client: tuple[str, int] = ("10.0.0.1", 1234)) -> dict[str, Any]:
    return {"type": "http", "method": method, "path": path, "headers": headers or [],
            "query_string": b"", "client": client, "scheme": "http", "server": ("t", 80),
            "http_version": "1.1", "root_path": ""}


async def test_chunked_body_over_limit_is_rejected_and_small_body_is_replayed() -> None:
    received: list[bytes] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        received.append(body)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    limited = BodySizeLimitMiddleware(app, max_bytes=10)
    sent = await _run_asgi(limited, _scope("POST", "/api/x"), [b"12345", b"67890", b"1"])
    assert sent[0]["status"] == 413 and received == []
    sent = await _run_asgi(limited, _scope("POST", "/api/x"), [b"1234", b"5678"])
    assert sent[0]["status"] == 200 and received == [b"12345678"]


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_session_creation_is_limited_to_10_per_minute(client: TestClient) -> None:
    statuses = [client.post("/api/sessions").status_code for _ in range(11)]
    assert statuses[:10] == [200] * 10 and statuses[10] == 429


def test_admin_login_is_limited_to_5_per_minute(client: TestClient) -> None:
    statuses = [client.post("/api/admin/login").status_code for _ in range(6)]
    assert statuses == [200] * 5 + [429]


def test_message_posts_use_configured_limit(client: TestClient) -> None:
    statuses = [client.post("/api/sessions/abc/messages").status_code for _ in range(4)]
    assert statuses == [200, 200, 200, 429]
    resp = client.post("/api/sessions/abc/messages")
    assert resp.json()["error_code"] == "RATE_LIMITED"
    assert 1 <= int(resp.headers["retry-after"]) <= 60
    # Reads are never rate limited.
    assert all(client.get("/api/ping").status_code == 200 for _ in range(10))


async def test_sliding_window_and_per_client_isolation() -> None:
    now = [1000.0]

    async def app(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    rule = RateRule("message", "POST", re.compile(r"^/m$"), limit=2, window_seconds=60)
    limiter = RateLimitMiddleware(app, [rule], clock=lambda: now[0])

    async def status(client_ip: str) -> int:
        sent = await _run_asgi(limiter, _scope("POST", "/m", client=(client_ip, 1)), [b""])
        return sent[0]["status"]

    assert [await status("a"), await status("a"), await status("a")] == [200, 200, 429]
    assert await status("b") == 200  # other client unaffected
    now[0] += 61
    assert await status("a") == 200  # window slid


def test_rate_limit_can_be_disabled_for_messages() -> None:
    client = TestClient(_build_app(_settings(RATE_LIMIT_PER_MINUTE=0)))
    assert all(client.post("/api/sessions/x/messages").status_code == 200 for _ in range(20))


# ---------------------------------------------------------------------------
# SSE passthrough
# ---------------------------------------------------------------------------

async def test_streaming_response_is_not_buffered() -> None:
    """The second chunk is only produced after the first one reached the client."""
    app = FastAPI()
    register_security_middleware(app, _settings())
    first_delivered = asyncio.Event()

    async def events() -> AsyncIterator[bytes]:
        yield b"event: stage\ndata: {}\n\n"
        await asyncio.wait_for(first_delivered.wait(), timeout=5)
        yield b"event: done\ndata: {}\n\n"

    @app.post("/api/sessions/s/messages/stream")
    async def stream() -> StreamingResponse:
        return StreamingResponse(events(), media_type="text/event-stream")

    bodies: list[bytes] = []
    start: list[Message] = []

    async def receive() -> Message:
        await asyncio.sleep(10)
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            start.append(message)
        elif message.get("body"):
            bodies.append(message["body"])
            first_delivered.set()

    scope = _scope("POST", "/api/sessions/s/messages/stream",
                   headers=[(b"content-length", b"0")])
    await asyncio.wait_for(app(scope, receive, send), timeout=5)
    assert bodies == [b"event: stage\ndata: {}\n\n", b"event: done\ndata: {}\n\n"]
    headers = dict(start[0]["headers"])
    assert headers[b"content-type"].startswith(b"text/event-stream")
    assert headers[b"cache-control"] == b"no-store"
