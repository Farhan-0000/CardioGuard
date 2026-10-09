"""
FastAPI dependencies enforcing session and admin tokens (ADL-033, ADL-034).

* ``require_session_access`` — when ``AUTH_REQUIRED`` the request must carry the
  session's signed token as ``Authorization: Bearer <token>`` or
  ``X-Session-Token: <token>``. Missing → 401, invalid/expired/other session → 403.
* ``require_session_erasure_access`` — the same check for the right-to-erasure
  route (``DELETE /api/sessions/{id}``) only, except that a correctly signed
  token for *this* session is still accepted after it expired, for a bounded
  grace period (:func:`erasure_token_grace_seconds`). Session tokens cannot be
  refreshed, while the data they protect lives until the retention purge
  (``DATA_RETENTION_DAYS`` after the last activity); without the grace a patient
  could not erase their data between token expiry and the purge. Possessing a
  validly signed token for the session is the proof of ownership; every other
  route keeps strict expiry.
* ``require_admin`` — the dashboard is disabled (403) unless ``ADMIN_PASSWORD``
  is set; otherwise a bearer admin token from ``POST /api/admin/login`` is required.

Malformed tokens (including non-ASCII header values) are rejected as invalid
(403), never with a server error.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException, Request, status

from medbridge.config import get_settings
from medbridge.privacy.tokens import decode_token, verify_admin_token, verify_session_token

_BEARER_HEADERS = {"WWW-Authenticate": "Bearer"}
_SECONDS_PER_DAY = 86_400

# Minimum grace for erasing with an expired session token. It is at least
# DATA_RETENTION_DAYS (the longest the data can outlive the token: activity
# stops when the token expires, and the purge runs DATA_RETENTION_DAYS after
# the last activity), so the bound never cuts the erasure window short.
ERASURE_TOKEN_GRACE_DAYS = 90


def erasure_token_grace_seconds() -> int:
    """How long after expiry a session token may still erase its own session."""
    days = max(ERASURE_TOKEN_GRACE_DAYS, get_settings().DATA_RETENTION_DAYS)
    return days * _SECONDS_PER_DAY


def extract_token(request: Request) -> str | None:
    """Bearer token from ``Authorization`` or ``X-Session-Token`` (``None`` if absent)."""
    auth = request.headers.get("authorization", "")
    scheme, _, value = auth.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    header = request.headers.get("x-session-token", "").strip()
    return header or None


def _check_session_token(session_id: UUID, request: Request, *, grace_seconds: int) -> None:
    if not get_settings().AUTH_REQUIRED:
        return
    token = extract_token(request)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token required.",
            headers=_BEARER_HEADERS,
        )
    if not verify_session_token(token, session_id, expired_grace_seconds=grace_seconds):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session token is invalid, expired or belongs to another session.",
        )


async def require_session_access(session_id: UUID, request: Request) -> None:
    """Allow the request only with a valid, unexpired token for ``session_id``."""
    _check_session_token(session_id, request, grace_seconds=0)


async def require_session_erasure_access(session_id: UUID, request: Request) -> None:
    """Erasure only: also accept a correctly signed but expired token for ``session_id``.

    The grace is bounded by :func:`erasure_token_grace_seconds`; a token for
    another session, a bad signature or a token older than the grace → 403.
    """
    _check_session_token(session_id, request, grace_seconds=erasure_token_grace_seconds())


async def require_admin(request: Request) -> str:
    """Return the admin username for a valid admin token (401/403 otherwise)."""
    if not get_settings().ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin dashboard disabled (ADMIN_PASSWORD is not set).",
        )
    token = extract_token(request)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin token required.",
            headers=_BEARER_HEADERS,
        )
    if not verify_admin_token(token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin token is invalid or expired.",
        )
    payload = decode_token(token) or {}
    return str(payload.get("sub", ""))
