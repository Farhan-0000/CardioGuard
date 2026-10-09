"""
Privacy & security layer (ADL-030 … ADL-035).

* ``redaction`` — deterministic PII/PHI redaction before any LLM call or write.
* ``crypto``    — envelope encryption at rest + crypto-shredding (right to erasure).
* ``tokens``    — HMAC-signed, expiring session and admin tokens.
* ``deps``      — FastAPI dependencies enforcing those tokens.
* ``middleware``— CORS, security headers/CSP, body-size and rate limits (pure ASGI).
* ``log_scrub`` — structlog processor removing PII from log events.
* ``retention`` — TTL job that crypto-shreds inactive sessions.

Submodules are imported explicitly by callers; this package has no side effects.
"""
