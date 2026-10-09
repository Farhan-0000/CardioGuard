# Privacy & security (`medbridge.privacy`)

Everything that keeps patient data on the machine, out of logs and
unreadable once a patient asks for erasure. Submodules are imported
explicitly; the package has no import side effects.

| Module | Responsibility |
|---|---|
| `redaction.py` | `redact(text) -> RedactionResult(text, count, categories)`. Deterministic regex/rule-based PII/PHI redaction run on every message before any LLM call and before persistence: names (only after strong cues), phones, e-mails, addresses, PIN/ZIP codes, Aadhaar, PAN, SSN, MRN/UHID/ABHA, insurance ids, URLs/IPs, exact dates of birth (→ `[AGE: N]`). BP readings, labs, doses, drug names (incl. brands), stated ages and emergency numbers are preserved. Owned by the privacy work-stream. |
| `crypto.py` | Envelope encryption at rest: `ENCRYPTION_MASTER_KEY` (KEK, environment only) wraps one random Fernet data key per session (`session_keys`). `SessionCipher.encrypt/decrypt` (text: `enc:v1:<token>`) and `encrypt_json/decrypt_json` (JSONB: `{"enc": "v1", "ct": …}`); `create_session_key`, `get_session_cipher` (`None` = erased), `shred_session_key`. Readers accept legacy plaintext. |
| `tokens.py` | Stateless HMAC-SHA256 tokens `v1.<payload>.<signature>`: session tokens bound to one session id (`SESSION_TOKEN_TTL_SECONDS`) and admin tokens; constant-time verification. Malformed or non-ASCII tokens are simply invalid (never a server error). `verify_session_token(..., expired_grace_seconds=N)` accepts a correctly signed, expired token for at most `N` seconds — used only for erasure. |
| `deps.py` | FastAPI dependencies: `require_session_access` (401 missing / 403 invalid or expired, from `Authorization: Bearer` or `X-Session-Token`) when `AUTH_REQUIRED`; `require_session_erasure_access` (only `DELETE /api/sessions/{id}`: also accepts the session's own correctly signed token after expiry, for `max(90, DATA_RETENTION_DAYS)` days); `require_admin` (dashboard disabled unless `ADMIN_PASSWORD` is set). |
| `middleware.py` | `register_security_middleware(app, settings)`: security headers + CSP (`Cache-Control: no-store` on `/api/*`), strict CORS (`CORS_ORIGINS`), per-IP sliding-window rate limits (messages `RATE_LIMIT_PER_MINUTE`, session creation, admin login → 429 + `Retry-After`), body-size limit (`MAX_REQUEST_BYTES` → 413). Pure ASGI, so SSE streams are not buffered. |
| `log_scrub.py` | `scrub_pii_processor`: structlog processor that runs `redact()` over every string value of a log event (bounded size and depth; UUIDs and structural keys kept). Installed by `medbridge.logging_config` next to the secret-masking processor. |
| `retention.py` | `purge_expired_sessions(db, older_than_days)` and `retention_loop(stop_event)`: crypto-shreds sessions inactive for `DATA_RETENTION_DAYS` (started by the app lifespan, hourly, never raises). |
| `evaluation.py` | Span-level precision/recall of the redactor on a labelled set (used by `scripts/pii_eval.py` and `tests/privacy/test_redaction.py`). |

## Data flow guarantees

* **LLM prompts** only ever contain the redacted message; with
  `LOCAL_ONLY_MODE=true` the LLM wrapper refuses remote providers.
* **At rest**, message text, audit request/response text, gate rationales,
  clinical event payloads, the snapshot, feedback comments and the private part
  of the audit trace are encrypted with the session data key. Only actions,
  chunk ids, latency and the PHI-free trace (`state.turn_store.split_trace`)
  stay in plaintext for `/api/metrics`.
* **Erasure** (`DELETE /api/sessions/{id}`, retention job): the data key and the
  mutable rows are deleted and `sessions.erased_at` is set. Rows in the
  append-only `clinical_events`/`audit_logs` tables remain but are permanently
  unreadable. Afterwards the history, context, summary, BP, feedback, message,
  stream and regenerate routes answer `410 SESSION_ERASED` (the message routes
  check this before any LLM work); a repeated `DELETE` is idempotent.
* **Erasure after the token expired.** Session tokens are stateless and cannot
  be refreshed, but the data they protect lives until the retention purge
  (`DATA_RETENTION_DAYS` after the last activity, and activity stops when the
  token expires). `DELETE /api/sessions/{id}` therefore accepts a token whose
  signature is valid *for that session* even after `exp`, for a bounded grace of
  `max(90 days, DATA_RETENTION_DAYS)` — possession of the signed token is the
  proof of ownership. A token for another session, a bad signature or a token
  older than the grace is still refused (403), and every other route keeps
  strict expiry. Clients must treat only 404/410 as "the server no longer has
  this conversation"; 401/403 mean erasure was *not* confirmed.
* **Legacy plaintext rows (known limitation).** Rows written before the
  encryption migration (`g1a7c3e9d2b4`) or while `ENCRYPTION_ENABLED=false` are
  plaintext at rest. Readers accept them; crypto-shredding cannot cover them.
  For the mutable tables erasure deletes them anyway, but rows in the
  append-only `audit_logs` / `clinical_events` can never be rewritten or deleted
  (the immutability rules are a non-negotiable invariant and are *not*
  disabled to encrypt them). Instead:
  * `erase_session` (user erasure and the retention purge) counts them into
    `sessions.legacy_plaintext_rows` and logs `erasure_legacy_plaintext_retained`;
  * every reader hides an erased session completely — history, recent turns,
    events, context, summary and the admin audit view (`[erased]`, with a
    `legacy_plaintext` flag instead of any text);
  * the admin audit view flags legacy rows of live sessions (`legacy_plaintext`).
  A database that ever held real data before encryption must be reset (drop and
  recreate it, then `alembic upgrade head`; with Docker `docker compose down -v`)
  — that is the only way to remove such rows. All project data is synthetic.
* **Logs** never contain request bodies; validation errors are logged without
  the offending input and every event passes the PII scrubber.

## Tests

`tests/privacy/` (redaction quality gate, crypto, encryption at rest, tokens and
dependencies, middleware, log scrubbing and retention),
`medbridge/tests/integration/test_message_routes.py` (410 for erased sessions),
`medbridge/tests/unit/test_privacy_erasure_tokens.py` (non-ASCII tokens → 403,
erasure with an expired token) and
`medbridge/tests/integration/test_turn_concurrency.py` (legacy plaintext rows
are counted at erasure and never surface afterwards).
