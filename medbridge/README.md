# `medbridge` — application package

The FastAPI back end of MedBridge (CardioGuard): a local-first patient
assistant for blood pressure and heart health built around the **Dual
Answerability Gate** — Gate 1 (Context Gate) decides whether personal facts
must be asked first, Gate 2 (Evidence Gate) decides what the retrieved
guideline evidence allows. Contracts: `docs/finalization_plan.md` §3;
invariants: `docs/project_constitution.md` §12.

## Layout

| Package / module | Responsibility | README |
|---|---|---|
| `main.py` | `create_app()`: logging, security middleware, error handlers, `/api` + `/health` routers, web UI mount; lifespan (DB pool, background model warm-up, retention job, clean shutdown). | — |
| `config.py` | `Settings` (pydantic-settings, `.env`); local-first defaults (`LLM_PROVIDER=ollama`, `LOCAL_ONLY_MODE=true`). | — |
| `logging_config.py` | structlog → stdlib JSON/console logging with secret masking and PII scrubbing. | — |
| `core/` | Orchestrator (one turn = traced stages), emergency fast-path, citations, small talk, templates, shared pipeline types. | [core/README.md](core/README.md) |
| `ai/` | Resilient LLM wrapper (Ollama structured outputs), extractor, Gate 1, Gate 2, generator, V1 calibrated gate, V2 grounding verifier, prompt isolation. | [ai/README.md](ai/README.md) |
| `nlp/` | Deterministic helpers: drug brand → generic normaliser, lay → clinical query expansion, BP mention parser. | — |
| `retrieval/` | Hybrid dense + BM25 search in Qdrant (RRF k = 60), cross-encoder rerank, caches. | [retrieval/README.md](retrieval/README.md) |
| `state/` | Event-sourced patient state: pure projector, encrypted single-flush turn store, session manager. | [state/README.md](state/README.md) |
| `privacy/` | PII redaction, envelope encryption + crypto-shredding, signed tokens, auth dependencies, security middleware, log scrubbing, retention. | [privacy/README.md](privacy/README.md) |
| `db/` | SQLAlchemy 2.0 async models and engine/session factory (migrations live in the repository's `alembic/`). | — |
| `api/` | Routes (`sessions`, `messages` incl. SSE + regenerate, `features`, `admin`, `metrics`, `config`, `health`), request/response schemas, global error handlers. | — |
| `features/` | BP categories (ACC/AHA) for the BP log and the clinician summary export. | — |
| `ingestion/` | Offline corpus → Qdrant indexer (the only writer of the collection). | [ingestion/README.md](ingestion/README.md) |
| `web/` | Single-page chat UI + admin dashboard (vanilla JS), mounted at `/` and `/dashboard`. | — |
| `frontend/` | Legacy Streamlit client (HTTP only). | [frontend/README.md](frontend/README.md) |
| `tests/` | `unit/` and `integration/` suites (hermetic by default). | — |

## One patient turn

```
raw message ─▶ 0 privacy (redact) ─▶ 1 emergency fast-path on the RAW text (no LLM) ─▶ ESCALATE template
            ─▶ 2 load state (snapshot, soft-ask counter, recent turns)
            ─▶ 2b small talk (deterministic template)
            ─▶ 3 extraction (LLM 1: delta events, search query, intent) ─▶ in-memory projection
            ─▶ 4 Gate 1 SOFT-ASK | PROCEED (LLM 2, T=0, seed 42; loop-breaker at 2)
                 ‖ speculative retrieval (discarded on SOFT-ASK)
            ─▶ 5 retrieval: dense ‖ BM25 → RRF(k=60) top-20 → cross-encoder top-5
            ─▶ 6 Gate 2 ANSWER | GENERALIZE | ABSTAIN | ESCALATE (LLM 3, T=0, seed 42; V1 optional)
            ─▶ 7 ABSTAIN/ESCALATE → templates; ANSWER/GENERALIZE → generator (LLM 4, T=0.3)
                 → citation validation (ANSWER without a valid citation → GENERALIZE)
                 → optional Gate 3 grounding verifier (V2)
            ─▶ 8 persistence: one turn_store.persist_turn() flush (encrypted text, PHI-free trace)
```

The redacted message is what every stage after the emergency check sees. Gate
2 and the generator receive both the patient's own (redacted) words
(`message=`, preferred, so false premises and unknown drug names are judged on
what the patient wrote) and the extractor's clinical search query (`query=`).
If extraction fails, retrieval uses `nlp.expand_query(redacted message)`.

## Invariants (enforced in code and tests)

* Five actions only: `SOFT-ASK`, `ANSWER`, `GENERALIZE`, `ABSTAIN`, `ESCALATE`.
* Deterministic emergency fast-path before any LLM call.
* Gate 1 and Gate 2 are separate decisions at temperature 0, seed 42.
* Loop-breaker: the third consecutive SOFT-ASK becomes GENERALIZE (count reset).
* Template bypass: ABSTAIN and ESCALATE never call the generator.
* XML isolation: untrusted text sits in `<untrusted_user_input>`; tag-like text
  in any untrusted slot is escaped (`ai/prompt_utils.neutralize_tags`).
* Append-only `clinical_events` and `audit_logs`.
* The orchestrator never raises; every failure degrades to a pre-vetted response.

## Running

```bash
.venv-mac/bin/python -m uvicorn medbridge.main:app --host 127.0.0.1 --port 8000
.venv-mac/bin/python -m pytest medbridge -q            # hermetic; postgres/qdrant/ollama tests auto-skip
.venv-mac/bin/python -m ruff check medbridge
.venv-mac/bin/python -m mypy medbridge
```

Tests never touch the development database: the root `conftest.py` rewrites
`DATABASE_URL` to the sibling `medbridge_test` database.
