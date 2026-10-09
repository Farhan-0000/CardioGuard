"""
HTTP security middleware (ADL-034).

``register_security_middleware(app, settings)`` installs, outermost first:

1. ``SecurityHeadersMiddleware`` — CSP, nosniff, no-referrer, frame denial,
   permissions policy, COOP on every response; ``Cache-Control: no-store`` on ``/api/*``.
2. ``CORSMiddleware`` — only ``CORS_ORIGINS``, no credentials, the methods and
   headers the SPA actually uses.
3. ``RateLimitMiddleware`` — in-memory sliding window per client IP:
   message/stream/regenerate POSTs (``RATE_LIMIT_PER_MINUTE``), session creation
   (10/min) and admin login (5/min) → 429 + ``Retry-After``.
4. ``BodySizeLimitMiddleware`` — bodies larger than ``MAX_REQUEST_BYTES`` → 413.

All custom middlewares are *pure ASGI* (not ``BaseHTTPMiddleware``) so
Server-Sent-Events responses stream through unbuffered.
"""
from __future__ import annotations

import math
import re
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from medbridge.api.middleware.error_handler import SAFE_FALLBACK
from medbridge.api.schemas.responses import ErrorResponse
from medbridge.config import Settings

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; font-src 'self' data:; "
    "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)
# FastAPI's interactive docs load Swagger UI / ReDoc from a CDN with inline scripts.
DOCS_CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https://fastapi.tiangolo.com; connect-src 'self'; "
    "font-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)
DOCS_PATHS = ("/docs", "/redoc")

SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("X-Frame-Options", "DENY"),
    ("Permissions-Policy", "camera=(), microphone=(), geolocation=()"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
)

SESSION_CREATE_PER_MINUTE = 10
ADMIN_LOGIN_PER_MINUTE = 5


def _error_response(status_code: int, error_code: str, message: str,
                    headers: dict[str, str] | None = None) -> JSONResponse:
    body = ErrorResponse(error_code=error_code, message=message, safe_fallback=SAFE_FALLBACK)
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"),
                        headers=headers)


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------

class SecurityHeadersMiddleware:
    """Adds security headers to every HTTP response start message."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        csp = DOCS_CONTENT_SECURITY_POLICY if path.startswith(DOCS_PATHS) else (
            CONTENT_SECURITY_POLICY
        )
        is_api = path.startswith("/api/") or path == "/api"

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if "content-security-policy" not in headers:
                    headers.append("Content-Security-Policy", csp)
                for name, value in SECURITY_HEADERS:
                    if name.lower() not in headers:
                        headers.append(name, value)
                if is_api:
                    headers["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_with_headers)


# ---------------------------------------------------------------------------
# Request body size limit
# ---------------------------------------------------------------------------

class BodySizeLimitMiddleware:
    """Rejects request bodies larger than ``max_bytes`` with 413 (JSON ErrorResponse)."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    def _too_large(self) -> JSONResponse:
        return _error_response(
            413, "PAYLOAD_TOO_LARGE", f"Request body exceeds the {self.max_bytes}-byte limit."
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            if declared.isdigit() and int(declared) > self.max_bytes:
                await self._too_large()(scope, receive, send)
                return
            await self.app(scope, receive, send)
            return

        # No Content-Length (chunked): buffer up to the limit, then replay.
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body: bytes = message.get("body", b"")
            total += len(body)
            if total > self.max_bytes:
                await self._too_large()(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break
        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RateRule:
    """``limit`` requests per ``window_seconds`` for ``method`` + ``path_pattern``, per client."""

    name: str
    method: str
    path_pattern: re.Pattern[str]
    limit: int
    window_seconds: float = 60.0


def default_rate_rules(settings: Settings) -> list[RateRule]:
    rules = [
        RateRule("session_create", "POST", re.compile(r"^/api/sessions/?$"),
                 SESSION_CREATE_PER_MINUTE),
        RateRule("admin_login", "POST", re.compile(r"^/api/admin/login/?$"),
                 ADMIN_LOGIN_PER_MINUTE),
    ]
    if settings.RATE_LIMIT_PER_MINUTE > 0:
        rules.append(RateRule(
            "message", "POST",
            re.compile(r"^/api/sessions/[^/]+/(?:messages(?:/stream)?|regenerate)/?$"),
            settings.RATE_LIMIT_PER_MINUTE,
        ))
    return rules


class RateLimitMiddleware:
    """In-memory sliding-window limiter keyed by (rule, client IP).

    State is per process; suitable for the single-node local deployment.
    """

    def __init__(self, app: ASGIApp, rules: Sequence[RateRule], max_keys: int = 10_000,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.app = app
        self.rules = list(rules)
        self.max_keys = max_keys
        self.clock = clock
        self._hits: dict[tuple[str, str], deque[float]] = {}

    def _match(self, method: str, path: str) -> RateRule | None:
        for rule in self.rules:
            if rule.method == method and rule.path_pattern.match(path):
                return rule
        return None

    def _prune(self, now: float) -> None:
        horizon = max((r.window_seconds for r in self.rules), default=60.0)
        for key in [k for k, q in self._hits.items() if not q or now - q[-1] >= horizon]:
            del self._hits[key]
        while len(self._hits) > self.max_keys:
            del self._hits[next(iter(self._hits))]

    def hit(self, rule: RateRule, client: str) -> float | None:
        """Record a request; return seconds to wait if the limit is exceeded."""
        now = self.clock()
        key = (rule.name, client)
        hits = self._hits.get(key)
        if hits is None:
            if len(self._hits) >= self.max_keys:
                self._prune(now)
            hits = self._hits[key] = deque()
        while hits and now - hits[0] >= rule.window_seconds:
            hits.popleft()
        if len(hits) >= rule.limit:
            return rule.window_seconds - (now - hits[0])
        hits.append(now)
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            rule = self._match(scope["method"], scope.get("path", ""))
            if rule is not None:
                client = scope.get("client")
                retry_after = self.hit(rule, client[0] if client else "unknown")
                if retry_after is not None:
                    seconds = max(1, math.ceil(retry_after))
                    response = _error_response(
                        429, "RATE_LIMITED",
                        f"Too many requests. Please wait {seconds} s and try again.",
                        headers={"Retry-After": str(seconds)},
                    )
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_security_middleware(app: FastAPI, settings: Settings) -> None:
    """Install CORS, security headers, rate limiting and the body size limit."""
    # Starlette wraps in reverse order: the last middleware added is outermost.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.MAX_REQUEST_BYTES)
    app.add_middleware(RateLimitMiddleware, rules=default_rate_rules(settings))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Session-Token"],
        expose_headers=["Retry-After"],
        max_age=600,
    )
    app.add_middleware(SecurityHeadersMiddleware)
