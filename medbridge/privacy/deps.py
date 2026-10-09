"""
FastAPI dependencies enforcing session and admin tokens (ADL-033, ADL-034).

* ``require_session_access`` — when ``AUTH_REQUIRED`` the request must carry the
  session's signed token as ``Authorization: Bearer <token>`` or
  ``X-Session-Token: <token>``. Missing → 401, invalid/expired/other session → 403.
* ``require_admin`` — the dashboard is disabled (403) unless ``ADMIN_PASSWORD``
  is set; otherwise a bearer admin token from ``POST /api/admin/login`` is required.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException, Request, status

from medbridge.config import get_settings
from medbridge.privacy.tokens import decode_token, verify_admin_token, verify_session_token

_BEARER_HEADERS = {"WWW-Authenticate": "Bearer"}


def extract_token(request: Request) -> str | None:
    """Bearer token from ``Authorization`` or ``X-Session-Token`` (``None`` if absent)."""
    auth = request.headers.get("authorization", "")
    scheme, _, value = auth.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    header = request.headers.get("x-session-token", "").strip()
    return header or None


async def require_session_access(session_id: UUID, request: Request) -> None:
    """Allow the request only with a valid token for ``session_id``."""
    if not get_settings().AUTH_REQUIRED:
        return
    token = extract_token(request)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token required.",
            headers=_BEARER_HEADERS,
        )
    if not verify_session_token(token, session_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session token is invalid, expired or belongs to another session.",
        )


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
