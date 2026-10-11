# MedBridge v3 — Project Constitution

> **Status:** Active & Permanent  
> **Authority:** Principal Software Architect & Engineering Lead  
> **Applies to:** All Developers, AI Coding Agents, Reviewers, and Architects  
> **Scope:** Entire MedBridge v3 Codebase, Infrastructure, and Lifecycle  

---

# 1. Project Mission

* **The Problem:** Hypertension patients frequently receive complex, generic, or conflicting medical advice and lack accessible, evidence-grounded guidance for interpreting readings, managing medications, and identifying acute cardiovascular risks.
* **Intended Users:** Adult hypertension patients seeking informational guidance on blood pressure management, prescribed medications, lifestyle modifications, and symptom triage.
* **Primary Objective:** Deliver a safe, deterministic, multi-turn clinical conversational agent that provides evidence-grounded hypertension guidance with verbatim citations, enforces strict emergency triage, tracks patient context through deterministic event sourcing, and prevents clinical hallucinations.

---

# 2. System Vision

MedBridge v3 is an intelligent clinical decision-support conversational application. The system ingests and indexes peer-reviewed clinical hypertension guidelines (AHA/ACC, ESC/ESH, MedlinePlus) into a hybrid dense-sparse vector store. At runtime, the application evaluates patient queries through a multi-stage guarded pipeline: it executes deterministic emergency triage, extracts structured medical facts into an append-only event stream, evaluates context sufficiency via deterministic classification gates, retrieves and reranks verified guideline evidence, and generates plain-language patient responses with inline clinical citations.

Expected outcomes include zero out-of-domain hallucinations, sub-5ms emergency crisis escalation, mathematically deterministic state tracking across conversation turns, and guaranteed patient safety via fail-safe fallback boundaries.

---

# 3. Architecture Principles

All development must strictly adhere to the following enforceable architectural rules:

1. **Separation of Concerns:** The presentation layer, backend orchestration, AI inference, state management, and persistence tiers must remain strictly decoupled into isolated modules.
2. **Single Responsibility:** Each module, service, class, and function must have exactly one reason to change and one clearly defined owner.
3. **Deterministic State Projection:** Patient state tracking must follow an Event Sourcing pattern. Snapshots are mathematical projections of immutable event logs and must never be updated directly by generative LLM output.
4. **Guarded Pipeline Execution:** All generative AI stages must be preceded by deterministic or zero-temperature classification gates. No evidence generation may proceed without evidence validation.
5. **Fail-Safe Robustness:** The system must never throw an unhandled 500 error to a patient. Every failure (timeout, parse error, API limit) must degrade deterministically to a pre-vetted clinical fallback template.
6. **Explicit & Typed Interfaces:** All inter-module communication must use strictly typed Pydantic models, SQLAlchemy ORM models, or typed Python signatures. Untyped dictionaries across module boundaries are prohibited.

---

# 4. Final System Components

* **Presentation Layer (`frontend/`):** A lightweight Streamlit interface responsible for capturing patient input, rendering multi-turn message history, displaying colored routing action badges, and presenting collapsible citation panels. It contains zero clinical or AI logic and communicates exclusively with the backend via HTTP.
* **API Gateway & Middleware (`api/`):** A FastAPI application exposing session management and messaging endpoints, enforcing request validation, managing CORS for loopback traffic, and implementing global exception filters that map failures to clinically safe responses.
* **Core Orchestrator & State Machine (`core/`):** The central deterministic runtime engine that coordinates the 8-stage pipeline, evaluates emergency triage regex patterns, executes the Context Gate loop-breaker state machine, coordinates retrieval, routes gate actions, and persists audit trails.
* **State Management Engine (`state/`):** A pure, deterministic Python projector and session manager that applies typed delta events to patient context snapshots and persists events atomically to PostgreSQL.
* **AI & Retrieval Pipeline (`ai/`, `retrieval/`):** A sequential pipeline consisting of a Resilient LLM Wrapper, Context Extractor (LLM 1), Context Gate (LLM 2), Qdrant Hybrid Retriever (Dense + Sparse + RRF), Cross-Encoder Reranker, Evidence Gate (LLM 3), and Response Generator (LLM 4).
* **Storage Tier (`db/`, Qdrant):** PostgreSQL 16 providing immutable append-only storage for clinical events and audit logs with materialized snapshots, and Qdrant providing hybrid dense-sparse vector indexing for chunked clinical guidelines.
* **Ingestion Pipeline (`ingestion/`):** An offline CLI tool that parses clinical PDFs with PyMuPDF, chunks text using section-aware boundaries (512/64 tokens), computes dense and sparse vectors, validates schema compliance, and upserts points to Qdrant.

---

# 5. Approved Technology Stack

Only the following approved technologies, frameworks, and libraries may be used. Introducing alternative frameworks is strictly prohibited without formal ADL approval.

| Layer | Approved Technology |
| :--- | :--- |
| **Frontend** | Streamlit ($\ge 1.35$), `httpx` (sync/async client) |
| **Backend Framework** | FastAPI ($\ge 0.111$), Uvicorn ($\ge 0.29$), Pydantic v2 ($\ge 2.7$), `pydantic-settings` |
| **Database & ORM** | PostgreSQL 16, SQLAlchemy 2.0 (async), `asyncpg`, Alembic ($\ge 1.13$), `psycopg2-binary` |
| **Vector Store** | Qdrant ($\ge 1.9$) running via Docker container |
| **Embedding Models** | `fastembed` ($\ge 0.3$) utilizing `BAAI/bge-small-en-v1.5` (Dense) and `Qdrant/bm25` (Sparse) |
| **Reranker Model** | `sentence-transformers` ($\ge 3.0$) utilizing `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Primary LLM Provider** | Groq Cloud API (`llama-3.1-8b-instant`) via HTTPS REST |
| **Fallback LLM Provider** | OpenAI API (`gpt-4o-mini` or equivalent) via HTTPS REST |
| **Document Processing** | PyMuPDF / `fitz` ($\ge 1.24$) |
| **Logging & Telemetry** | `structlog` (structured JSON logging) |
| **Testing Suite** | `pytest`, `pytest-asyncio`, `pytest-cov`, `httpx` test client |
| **Infrastructure** | Docker Engine ($\ge 24.0$), Docker Compose v3.9 |

---

# 6. Data Ownership Rules

1. **Frontend Isolation:** The Streamlit frontend owns only local presentation state (`st.session_state`). It is **forbidden** from directly accessing PostgreSQL, Qdrant, or external LLM APIs.
2. **State Ownership:** The `state/` module owns the schema and business logic for mutating context snapshots and appending to `clinical_events`. No other module may write to these tables.
3. **Audit Log Immutability:** The `audit_logs` and `clinical_events` tables are write-once, append-only data stores. Database `RULE` triggers must reject all `UPDATE` and `DELETE` queries.
4. **Vector Store Ownership:** The `ingestion/` module owns write access (collection creation, vector upsert) to Qdrant. The `retrieval/` module has strictly read-only search access.
5. **Direct AI Bypass Prohibited:** AI modules (`ai/`) may not query the database or vector store directly; they must receive inputs from and return outputs to the Core Orchestrator.

---

# 7. AI System Rules

1. **XML Boundary Isolation:** All prompts containing untrusted user text must encapsulate the input within `<untrusted_user_input>` tags to neutralize prompt injection attacks. System prompts, context, and evidence must be strictly partitioned in distinct XML blocks (`<system_instructions>`, `<patient_context>`, `<clinical_evidence>`, `<user_query>`).
2. **Deterministic Gate Classification:** Context Gate (LLM 2) and Evidence Gate (LLM 3) must always execute with `temperature: 0.0` and `seed: 42`. Non-zero temperature on safety gates is strictly prohibited.
3. **Generator Temperature Constraint:** Response Generator (LLM 4) must execute with `temperature: 0.3` to balance clinical adherence with linguistic fluency.
4. **Template Bypass on Rejection:** If the Evidence Gate or Emergency Classifier evaluates to `ABSTAIN` or `ESCALATE`, the Response Generator (LLM 4) **must not be invoked**. Pre-vetted deterministic templates from `core/templates.py` must be returned immediately.
5. **Loop-Breaker Threshold:** If `soft_ask_count >= 2`, the orchestrator must override the Context Gate action to `GENERALIZE`, inject the loop-breaker transition prefix, and force progression to retrieval.
6. **Hallucination Zero-Tolerance:** The Response Generator must generate claims grounded strictly in provided `<clinical_evidence>`. Uncited clinical claims are treated as critical defects.
7. **Threadpool Offloading:** The Cross-Encoder Reranker is a CPU-bound operation and must always be invoked via `asyncio.to_thread()` to prevent blocking the FastAPI event loop.

---

# 8. Security Principles

1. **Loopback Binding:** All local service ports (FastAPI `:8000`, PostgreSQL `:5432`, Qdrant `:6333`, Streamlit `:8501`) must bind strictly to `127.0.0.1`. Binding to `0.0.0.0` in production without an authenticating reverse proxy is prohibited.
2. **Secret Management:** API keys, database credentials, and secret tokens must be loaded through `pydantic-settings` from environment variables. Hardcoding credentials in code, configuration files, or Dockerfiles is strictly forbidden.
3. **No Plaintext PHI in Prompts:** Real patient Protected Health Information (PHI) must never be sent to external LLM APIs. All testing, benchmarking, and development must use synthetic clinical vignettes.
4. **Structured Logging Sanitization:** Authorization headers, bearer tokens, API keys, and raw database passwords must be scrubbed automatically by `structlog` log processors prior to log emission.
5. **Input Length Constraints:** The API layer must reject messages exceeding 2,000 characters with HTTP 422 to prevent denial-of-service and buffer exhaustion.

---

# 9. Coding Standards

1. **Strict Type Hinting:** All Python code must include complete type annotations (arguments and return types). Code must pass `mypy --strict` without errors.
2. **Linting & Formatting:** All code must conform to standard PEP 8 rules, enforced via `ruff` with line-length 100.
3. **Error Handling Protocol:** Never use bare `except:` clauses. Catch specific exceptions (`httpx.TimeoutException`, `ValidationError`, `json.JSONDecodeError`).
4. **Async/Await Integrity:** Avoid blocking calls in async route handlers or orchestrator logic. Use async drivers (`asyncpg`, `httpx.AsyncClient`) and offload heavy CPU work to threadpools.
5. **Testing Thresholds:** Every module must maintain $\ge 85\%$ unit test coverage. All core state transitions and emergency classifier patterns must have $100\%$ branch coverage.

---

# 10. Integration Rules

```
ALLOWED:
Streamlit ──(HTTP)──▶ FastAPI Route ──▶ Orchestrator ──▶ State Projector ──▶ PostgreSQL
                                                │ ──▶ Emergency Classifier
                                                │ ──▶ Resilient LLM Wrapper ──▶ Groq Cloud API
                                                │ ──▶ Hybrid Retriever ──▶ Qdrant
                                                │ ──▶ Cross-Encoder (Threadpool)

PROHIBITED:
Streamlit ──(SQL)──❌──▶ PostgreSQL
Streamlit ──(gRPC)─❌──▶ Qdrant
AI Module ──(SQL)──❌──▶ PostgreSQL
Retriever ──(HTTP)─❌──▶ Groq Cloud API
FastAPI   ──(Raw)──❌──▶ Qdrant (bypassing Hybrid Retriever)
```

1. All external network traffic must pass through the `ResilientLLMWrapper` (for LLMs) or `HybridRetriever` (for Qdrant).
2. The orchestrator is the sole coordinator. Peer-to-peer communication between sub-pipelines (e.g. Extractor calling Retriever directly) is forbidden.

---

# 11. Development Rules

1. **Architecture Decision Records:** Any proposed change that alters component boundaries, introduces new dependencies, modifies database schemas, or changes the AI pipeline flow must be documented and approved in the `architecture_decision_log.md` before implementation.
2. **Module Modification Boundaries:** Changes to a module must not alter its public interface without updating its corresponding specification in `module_specifications.md` and downstream callers.
3. **Deterministic Testing:** Automated tests must not make live calls to external LLM APIs by default. Unit and integration tests must use deterministic mocks or recorded responses. Live API tests must be segregated into optional benchmark suites.
4. **Database Migrations:** Schema changes must be enacted exclusively through forward-compatible Alembic migration scripts. Direct schema modification via SQL console is prohibited.

---

# 12. Non-Negotiable Constraints

The following constraints are absolute and non-negotiable across the entire project lifecycle:

* **[NON-NEGOTIABLE 1] Deterministic Fast-Path Triage:** The Emergency Classifier must evaluate messages before any LLM is called and return `ESCALATE` in $<5$ms on match.
* **[NON-NEGOTIABLE 2] Append-Only Event Log:** Clinical events and audit logs must remain permanently immutable; no updates or deletes are permitted.
* **[NON-NEGOTIABLE 3] Pure Zero-Temperature Gates:** Context Gate and Evidence Gate must execute at `temperature: 0.0` with `seed: 42`.
* **[NON-NEGOTIABLE 4] Loop-Breaker Limit:** The `soft_ask_count` limit is strictly 2. Upon reaching 2, the system must force `GENERALIZE`.
* **[NON-NEGOTIABLE 5] XML Input Isolation:** Patient input must always be delimited by `<untrusted_user_input>` XML tags in all prompts.
* **[NON-NEGOTIABLE 6] Template Bypass:** `ABSTAIN` and `ESCALATE` decisions must never call the LLM Generator; they must return deterministic templates.
* **[NON-NEGOTIABLE 7] No HTTP 500 to Patients:** The backend must handle all internal exceptions gracefully and return a valid, safe fallback response.

---

# 13. Definition of Done

A task, feature, or pull request is complete and accepted into the codebase only when all of the following criteria are satisfied:

- [ ] **Functional Compliance:** Implements all acceptance criteria defined in the Development Backlog task.
- [ ] **Architecture Adherence:** Respects all data ownership rules, integration constraints, and non-negotiable boundaries.
- [ ] **Unit Tests Passing:** Complete unit test suite implemented with $\ge 85\%$ line coverage and zero failures.
- [ ] **Integration Verified:** Verified against running PostgreSQL and Qdrant instances with clean transaction commits and rollbacks.
- [ ] **Type & Lint Cleanliness:** Passes `ruff check .` and `mypy --strict` with zero warnings or errors.
- [ ] **Deterministic Behavior:** Verified that safety gates and emergency checks operate deterministically.
- [ ] **Documentation Updated:** Module specifications, API schemas, and runbooks updated to reflect any authorized modifications.
- [ ] **Security Verified:** No hardcoded secrets, input length validated, and XML isolation tags present on all prompt templates.

---

# 14. Amendments (2026-10, approved via ADL)

> **Date:** 2026-10-10 · **Approval path:** Development Rule §11.1 — each amendment is backed by entries in `docs/architecture_decision_log.md` (ADL-027 – ADL-058), whose scope was approved beforehand in `docs/finalization_plan.md`.
> **Precedence:** Sections 1 – 13 above are kept unchanged as the historical record. Where an amendment below differs from earlier text, the amendment governs. **No Non-Negotiable Constraint of §12 is relaxed** (see A-9).

## A-1. Local Ollama inference is the default; Groq and OpenAI become optional (amends §5, §8.3, §10)

*ADL-039, ADL-040, ADL-057*

1. The **primary LLM provider** is a local **Ollama** server reached through its native `/api/chat` endpoint with JSON-schema structured output. The application model is chosen by a recorded experiment (`evaluation/results/model_selection_round1_prefix/`; `qwen3.5:4b` for the hallucination study) and is configured with `LLM_MODEL`, optionally split per role with `LLM_MODEL_GATES` and `LLM_MODEL_GENERATOR`.
2. `LOCAL_ONLY_MODE=true` is the default. In this mode the Resilient LLM Wrapper builds only the Ollama provider, accepts only a loopback Ollama host, refuses cloud-offloaded model tags, and sends nothing if the configuration violates these rules (the spec fallbacks then apply).
3. Groq and OpenAI remain **optional** providers, usable only with `LOCAL_ONLY_MODE=false`; their rows in the §5 table now read "optional".
4. §8.3 is strengthened: in local-only mode no patient text leaves the machine. The obligations to never send real PHI to external APIs and to use synthetic data for all testing remain.
5. §10 is read with "Resilient LLM Wrapper ──▶ Ollama (loopback)" in place of "Groq Cloud API"; a remote provider may appear only behind the same wrapper and only with local-only mode switched off. Rule §10.1 (all LLM traffic through the wrapper) is unchanged.

## A-2. FastAPI-served web SPA alongside the Streamlit client (amends §4, §5, §6.1, §8.1)

*ADL-053, ADL-054, ADL-055*

1. The **presentation layer** consists of a static single-page application (`medbridge/web/`: HTML, CSS and ES modules served by FastAPI at `/`, `/dashboard` and `/static/*`) and the existing **Streamlit** client (`medbridge/frontend/`), which remains supported as a legacy client. No Node.js build step and no third-party JavaScript are used at runtime; Node.js is permitted only for optional JavaScript unit tests.
2. §6.1 applies to both clients: each owns only presentation state (the SPA keeps its conversation list and session tokens in browser storage; Streamlit uses `st.session_state`), contains no clinical or AI logic, and talks to the backend over HTTP only.
3. Streaming uses Server-Sent Events from the same FastAPI process. Text is released only after the turn has been validated, so streaming never bypasses a gate, citation validation or Gate 3.
4. §8.1 is read as follows: the SPA is served on the FastAPI port (8000); Streamlit uses 8501; Docker Compose maps PostgreSQL to host port 5433 and Qdrant to 6333/6334. Every port remains bound to `127.0.0.1`.

## A-3. Privacy & security layer (adds to §4, §6, §8)

*ADL-030 – ADL-035, ADL-038*

1. A new component **Privacy & Security Layer** (`medbridge/privacy/`) is part of the approved architecture: deterministic PII/PHI redaction before any LLM call, log line or database write (only the deterministic emergency classifier sees the raw message), per-session envelope encryption at rest, crypto-shredding erasure, signed expiring session tokens, admin authentication, HTTP security middleware, PII log scrubbing and a data-retention purge.
2. §6.2 is clarified: the `state/` package (`state/turn_store.py` with the projector) is the only writer of `clinical_events`, `context_snapshots` and `audit_logs`. Snapshot writers lock the snapshot row and re-project their own events, so the snapshot always equals the fold of the event log (§3.3).
3. §6.3 is clarified: **right to erasure is implemented by destroying the session's data key (crypto-shredding)**. UPDATE and DELETE on `clinical_events` and `audit_logs` remain forbidden, session rows are never deleted, and only mutable rows (`message_history`, `context_snapshots`, `feedback`, `session_keys`) may be deleted.
4. §8.4 is extended: log processors also remove PII from every string value of a log event, and the database engine hides bound SQL parameters in errors.
5. §8.5 is extended: in addition to the 2,000-character message limit (HTTP 422), bodies above `MAX_REQUEST_BYTES` are rejected with 413 and per-client rate limits answer 429 with `Retry-After`.
6. New §8.6: when `AUTH_REQUIRED` is true (the default), every `/api/sessions/{id}/*` route requires a bearer token signed for that session; the admin dashboard and metrics require a separate admin token and are disabled while `ADMIN_PASSWORD` is empty.

## A-4. Evaluation package (adds to §4, extends §11.3)

*ADL-050, ADL-056, ADL-058*

1. The **evaluation package** (`evaluation/`) and the benchmark, adversarial and operations harnesses (`tests/benchmark/`, `tests/adversarial/`, `scripts/`) are approved components. They may run the pipeline only through `PipelineOrchestrator.process_message(..., ablation=AblationConfig.preset(...))`. Ablation switches are never exposed through the public API, and production always uses `AblationConfig.full()`.
2. §11.3 is extended: live LLM evaluation runs are optional suites that write only to a dedicated evaluation database (`EVAL_DATABASE_URL`); the default test suite uses the isolated `medbridge_test` database and mocks every LLM call.
3. Thresholds may be calibrated only on the held-out development split; reported results come from the test split. Degraded turns (any fallback or failure path) are flagged and excluded from headline metrics, with a sensitivity analysis that includes them.
4. Every number published in the README, report, paper or slides must be generated by a script from `evaluation/results/` or `reports/`. Evaluation data remain synthetic (ADL-017).

## A-5. Approved technology additions (amends §5)

*ADL-030 – ADL-035, ADL-039, ADL-052, ADL-056*

| Layer | Added technology |
| :--- | :--- |
| **LLM runtime** | Ollama (local, Metal/CUDA/CPU), native `/api/chat` API |
| **Encryption** | `cryptography` (Fernet envelope encryption) |
| **NLI models** | `sentence-transformers` cross-encoders: `cross-encoder/nli-MiniLM2-L6-H768` (V2 grounding verifier), `cross-encoder/nli-deberta-v3-base` (evaluation judge only) |
| **Presentation** | Static HTML/CSS/ES modules served by FastAPI (no CDN, no bundler); Node.js optional for JS unit tests |
| **Evaluation & figures** | `numpy`, `scipy`, `matplotlib`, `openpyxl` |
| **Report generation** | `python-docx`, `python-pptx`, `playwright` (screenshots) |
| **Quality tooling** | `ruff`, `mypy`, `pytest-timeout` |

Exact pinned versions are in `requirements.txt` and `pyproject.toml`. The embedding models (`BAAI/bge-small-en-v1.5`, `Qdrant/bm25`) and the reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) are unchanged (ADL-050).

## A-6. Pipeline additions that may only make a response more conservative (amends §4 "Core Orchestrator", §7)

*ADL-027 – ADL-029, ADL-036, ADL-043, ADL-051, ADL-052, ADL-061*

1. The orchestrator runs explicit, traced stages: privacy → emergency → load state → small talk → extraction → Context Gate (with optional speculative retrieval) → retrieval and rerank → Evidence Gate → generation → Gate 3 verification (on by default since the study) → citation validation → single-flush persistence.
2. Approved additions, each of which can only keep or lower the action and never bypasses a gate: deterministic small-talk replies (GENERALIZE, no clinical claims); speculative retrieval whose results are discarded on SOFT-ASK; server-side citation validation (an ANSWER with no valid citation becomes GENERALIZE); the optional **V1 calibrated Evidence Gate** (permutation votes, each at temperature 0.0 and seed 42; thresholds can only make Gate 2 more conservative); the **V2 grounding verifier (Gate 3)**, which may remove unsupported sentences or downgrade to GENERALIZE/ABSTAIN, and withdraws a draft whose every claim is unsupported (ADL-061). Both were evaluated in the hallucination study: V2 (Gate 3) significantly reduced hallucination and is enabled by default with dev-calibrated thresholds (ADL-052); V1 showed no significant effect and stays disabled (ADL-051).
3. After a SOFT-ASK, Gate 1, Gate 2 and the generator judge the original question together with the follow-up details ("effective question"); the emergency classifier and the extractor still see only the current message.
4. §7.3 is clarified: the generator runs at temperature 0.3 with top_p 0.95 and the shared seed 42, so evaluation runs are reproducible.
5. §7.7 is extended: every CPU- or GPU-bound model inference in the request path (cross-encoder reranker, NLI verifier) runs in a worker thread, on the best available device (CUDA, then MPS, then CPU).

## A-7. Knowledge base and ingestion (amends §2, §4 "Ingestion Pipeline")

*ADL-046, ADL-047*

1. The knowledge base consists of exactly the documents listed in `data/corpus_manifest.json`, each with organisation, year, canonical URL, licence text and a SHA-256 pin (27 documents at the time of this amendment, `reports/corpus_stats.md`). Scraping of unvetted sources is prohibited, and sources whose licence forbids text-and-data mining are excluded.
2. Chunks are at most 400 tokens with 60 tokens of overlap, measured with the embedding model's tokenizer; tables become separate chunks; ingestion is incremental and fingerprinted. The ingestion module remains the only writer to Qdrant (§6.4).

## A-8. Domain scope (amends §1, §2)

The intended users and scope widen from adult hypertension patients to **cardiovascular care centred on hypertension** (hypertension and hypertensive emergencies, heart failure, atrial fibrillation, coronary disease, dyslipidaemia and cardiovascular risk, CKD-related blood pressure, diabetes-related cardiovascular risk, hypertensive disorders of pregnancy, lifestyle and prevention). Non-cardiovascular medicine and non-medical topics are out of domain. The system still does not diagnose or prescribe.

## A-9. Non-Negotiable Constraints reaffirmed

| # | Constraint | Status after the 2026-10 changes |
| :--- | :--- | :--- |
| 1 | Deterministic fast-path triage before any LLM call | Kept and extended (natural language, negation, typos, Hinglish; ADL-045). Still deterministic, on the raw message, before any LLM call; tests enforce under 5 ms on typical messages. |
| 2 | Append-only event log and audit log | Kept. Erasure destroys keys, never rows (ADL-032); snapshot writers lock and re-project (ADL-038). |
| 3 | Zero-temperature gates (0.0, seed 42) | Kept. Native structured output keeps the same options (ADL-039); V1 votes are each zero-temperature, seed-42 calls (ADL-051). |
| 4 | Loop-breaker limit of 2 → forced GENERALIZE | Kept, and now also correct for regenerate (ADL-037) and for follow-up answers (ADL-043). |
| 5 | XML input isolation in all prompts | Kept and strengthened: the intent summary and earlier turns are isolated too, and tag-like text is neutralised (ADL-041). |
| 6 | Template bypass for ABSTAIN and ESCALATE | Kept. Gate 3 abstentions also use the template (ADL-052). |
| 7 | No HTTP 500 to patients | Kept for the pipeline. The orchestrator never raises: for an accepted request the message and stream routes return a `MessageResponse` (a pre-vetted template on any pipeline failure), and SSE failures end with a safe `done` event (ADL-036, ADL-054). Requests refused before the pipeline receive structured 4xx errors (401, 403, 404, 410, 413, 422, 429). An unexpected exception outside the pipeline is still mapped by the global handler to a structured `ErrorResponse` with safe fallback text and no stack trace, carrying status 500 as in the original error-handler design. |

## A-10. Documentation references (amends §11.2, §13)

Module boundaries and interfaces are documented in `docs/module_map.md`, the interface contract in `docs/finalization_plan.md` §3 and the README of each package. `module_specifications.md`, referenced in §11.2, is not part of this repository; §11.2 and the "Documentation Updated" item of §13 refer to these documents instead.
