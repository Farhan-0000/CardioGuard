"""
Signed, expiring bearer tokens (ADL-033).

Format: ``v1.<payload_b64url>.<signature_b64url>`` where the signature is
HMAC-SHA256 over ``v1.<payload_b64url>`` with ``SESSION_TOKEN_SECRET``.

* Session token payload: ``{"typ": "session", "sid": "<uuid>", "exp": <unix>}``
  — grants access to exactly one session (the UUID alone is no longer enough).
* Admin token payload:   ``{"typ": "admin", "role": "admin", "sub": "<user>", "exp": <unix>}``

Signatures are compared in constant time. Tokens are stateless; they expire
after ``SESSION_TOKEN_TTL_SECONDS`` (session) or ``ADMIN_TOKEN_TTL_SECONDS``.
If ``SESSION_TOKEN_SECRET`` is empty an ephemeral per-process secret is used
(with a loud warning): tokens then stop working after a restart.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog

from medbridge.config import get_settings

logger = structlog.get_logger(__name__)

TOKEN_VERSION = "v1"
ADMIN_TOKEN_TTL_SECONDS = 8 * 3600

_ephemeral_secret: bytes | None = None


def _secret() -> bytes:
    global _ephemeral_secret
    configured = get_settings().SESSION_TOKEN_SECRET
    if configured:
        return configured.encode("utf-8")
    if _ephemeral_secret is None:
        _ephemeral_secret = secrets.token_bytes(32)
        logger.warning(
            "session_token_secret_missing",
            message="!!! SESSION_TOKEN_SECRET is empty: using an EPHEMERAL secret. Issued "
            "tokens become invalid after a restart. Run `python scripts/setup_env.py`. !!!",
        )
    return _ephemeral_secret


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _sign(signing_input: str) -> str:
    digest = hmac.new(_secret(), signing_input.encode("ascii"), hashlib.sha256).digest()
    return _b64encode(digest)


def _encode(payload: dict[str, Any]) -> str:
    body = _b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signing_input = f"{TOKEN_VERSION}.{body}"
    return f"{signing_input}.{_sign(signing_input)}"


def decode_token(token: str) -> dict[str, Any] | None:
    """Verify signature and expiry; return the payload or ``None``."""
    if not token or token.count(".") != 2:
        return None
    version, body, signature = token.split(".")
    if version != TOKEN_VERSION:
        return None
    expected = _sign(f"{version}.{body}")
    if not hmac.compare_digest(expected.encode("ascii"), signature.encode("ascii", "replace")):
        return None
    try:
        payload = json.loads(_b64decode(body))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    exp = payload.get("exp")
    if not isinstance(exp, int) or exp <= int(time.time()):
        return None
    return payload


def issue_session_token(session_id: uuid.UUID | str) -> tuple[str, datetime]:
    """Issue a bearer token for one session. Returns ``(token, expires_at)``."""
    exp = int(time.time()) + get_settings().SESSION_TOKEN_TTL_SECONDS
    token = _encode({"typ": "session", "sid": str(session_id), "exp": exp})
    return token, datetime.fromtimestamp(exp, tz=UTC)


def verify_session_token(token: str, session_id: uuid.UUID | str) -> bool:
    """True when ``token`` is valid, unexpired and was issued for ``session_id``."""
    payload = decode_token(token)
    if payload is None or payload.get("typ") != "session":
        return False
    return hmac.compare_digest(str(payload.get("sid", "")), str(session_id))


def issue_admin_token_with_expiry(username: str) -> tuple[str, datetime]:
    """Issue a dashboard bearer token; returns ``(token, expires_at)``."""
    exp = int(time.time()) + ADMIN_TOKEN_TTL_SECONDS
    token = _encode({"typ": "admin", "role": "admin", "sub": username, "exp": exp})
    return token, datetime.fromtimestamp(exp, tz=UTC)


def issue_admin_token(username: str) -> str:
    """Issue a dashboard bearer token (valid for ``ADMIN_TOKEN_TTL_SECONDS``)."""
    return issue_admin_token_with_expiry(username)[0]


def verify_admin_token(token: str) -> bool:
    """True for a valid, unexpired admin token while the dashboard is enabled."""
    settings = get_settings()
    if not settings.ADMIN_PASSWORD:
        return False
    payload = decode_token(token)
    return bool(
        payload
        and payload.get("typ") == "admin"
        and payload.get("role") == "admin"
        and payload.get("sub") == settings.ADMIN_USERNAME
    )
