# MedBridge — Finalization Plan (Review 4 / Review 5)

Status: active · Owner: finalization work-stream · Started 2026-10-09

This plan turns the Review-3 prototype into a finished, measured, documented
system without changing its architecture. The **Dual Answerability Gate**
(Gate 1 Context Gate → retrieval → Gate 2 Evidence Gate) and every invariant in
`docs/project_constitution.md` §12 stay intact. Every change gets an ADL entry
(ADL-027 onward).

## 0. Baseline (recorded before any change)

| Item | Value | Evidence |
|---|---|---|
| Code | commit `fad6fea` (team WIP baseline on top of `d2b1a3f`) | git log |
| Tests | 480 passed, 5 failed (4 guideline-id mismatches, 1 live-LLM loop-breaker test) | `reports/baseline/pytest_baseline.txt` |
| Corpus | 3 PDFs, 905 chunks (AHA/ACC 2025, ESC 2024, NICE NG136) | Qdrant |
| Model | `gemma3:4b` via Ollama OpenAI-compatible endpoint | `.env` |
| Latency | cold + warm per action, original code | `reports/performance/before_gemma3-4b.json` |

## 1. Phases & checkpoints

| Phase | Content | Checkpoint |
|---|---|---|
| P0 | Baseline, env, contracts (this doc) | baseline numbers saved |
| P1 | Parallel implementation (work-streams A–G below) | each WS: own tests green |
| P2 | Integration, full test suite, adversarial code review | suite green, review findings fixed |
| P3 | Corpus ≥15 docs ingested, retrieval evaluation, HalluBench authored | corpus manifest + retrieval metrics |
| P4 | Model-selection experiment → choose application + judge models | `evaluation/results/model_selection/` |
| P5 | Hallucination study C0–C5 + V1 + V2 (live, local) + judges + stats | `evaluation/results/` + figures |
| P6 | Live AQ benchmark, RABBITS, adversarial, load test, smoke test, perf after | `reports/` |
| P7 | Report DOCX/PDF, paper DOCX+LaTeX, PPTX, README, HOW_TO_RUN, checklist, demo script, CHANGELOG, ADL | `report/`, `docs/` |
| P8 | Fresh-clone dry run following HOW_TO_RUN.md | dry-run log |

Compute budget for P4–P6: ~12–18 h of local inference on the Apple M5 (24 GB).
Identical LLM requests are memoised during evaluation (content-addressed cache),
so paired conditions differ only in pipeline structure and shared calls are not
recomputed.

## 2. Work-streams and file ownership (P1)

Agents work in the same tree; **only touch files you own**. Shared contract files
(`medbridge/config.py`, `medbridge/core/pipeline_types.py`,
`medbridge/api/schemas/responses.py`, `medbridge/state/turn_store.py` signatures)
are frozen — extend only by adding optional fields, never by renaming.

| WS | Scope | Owns |
|---|---|---|
| A core | orchestrator refactor, speculative retrieval, tracing, citations, small talk, SSE, regenerate, metrics, health, warm-up, retrieval caching | `medbridge/core/orchestrator.py`, `core/citations.py`, `core/smalltalk.py`, `core/templates.py`, `medbridge/retrieval/*`, `api/routes/messages.py`, `api/routes/health.py`, `api/routes/metrics.py`, `api/routes/config.py`, `api/routes/__init__.py`, `api/schemas/requests.py`, `medbridge/main.py`, related tests |
| B ai | Ollama-native wrapper, structured outputs, prompts + few-shot, extractor history, generator modes, V1 calibrated gate, V2 grounding verifier, drug normaliser, query expansion | `medbridge/ai/**`, `medbridge/nlp/**`, related tests |
| C emergency | deterministic NL emergency classifier (colloquial, negation, typos) | `medbridge/core/emergency_classifier.py`, its tests |
| D privacy+persistence | redaction, encryption, crypto-shredding, tokens, admin auth, rate limit, headers, log scrubbing, retention; DB models + Alembic; turn store; session/feature/admin routes | `medbridge/privacy/**`, `medbridge/db/**`, `alembic/versions/*`, `medbridge/state/**`, `medbridge/logging_config.py`, `api/middleware/**`, `api/routes/sessions.py`, `api/routes/features.py`, `api/routes/admin.py`, `medbridge/features/**`, `tests/privacy/**` |
| E web | ChatGPT-style SPA + dashboard (vanilla JS, no CDN, no third-party JS), Streamlit legacy kept working | `medbridge/web/**`, `medbridge/frontend/**` |
| F corpus | ≥15 open-access documents, manifest, ingestion upgrades, re-index, stats | `medbridge/ingestion/**`, `data/**`, `scripts/download_corpus.py` |
| G evaluation | HalluBench, runner, judges, stats, figures, retrieval eval, model selection, annotation sheet | `evaluation/**` |

## 3. Interface contract

### 3.1 Request flow (production, `AblationConfig.full()`)

```
0 privacy      redact(message)  → redacted text used by EVERYTHING below except stage 1
1 emergency    classify(raw message)  (deterministic, <10 ms, before any LLM)  → ESCALATE template
2 load_state   turn_store.load_session_state + get_recent_turns(EXTRACTOR_HISTORY_TURNS)
2b small_talk  is_small_talk(redacted) → deterministic GENERALIZE template, no clinical claims
3 extraction   ContextExtractor.extract(redacted, snapshot, history)  → delta events (+ drug normalisation)
               StateProjector.apply_events(snapshot, events)  (pure, in memory)
4 context gate ContextGate.evaluate(new_snapshot, redacted, raw_intent)
               ‖ speculative retrieval task (SPECULATIVE_RETRIEVAL) started right after extraction
               SOFT-ASK & count < 2  → cancel/discard retrieval, generator(SOFT-ASK), count += 1
               SOFT-ASK & count ≥ 2  → loop-breaker: force GENERALIZE (Gate 2 skipped), count = 0
               PROCEED                → count = 0
5 retrieval    hybrid (dense ‖ sparse) → RRF k=60 top-20 → cross-encoder top-5 (thread)
6 evidence gate EvidenceGate.evaluate → ANSWER | GENERALIZE | ABSTAIN | ESCALATE
               (V1 when enabled: calibrated_evidence_decision)
7 generation   ABSTAIN/ESCALATE → templates (no LLM)
               ANSWER/GENERALIZE → ResponseGenerator.generate → validate_citations
               ANSWER with 0 valid citations → downgrade to GENERALIZE
               (V2 when enabled: GroundingVerifier.verify → strip / downgrade / abstain)
8 persistence  turn_store.persist_turn(TurnRecord) — ONE flush; trace → audit_logs.trace
```

LLM failure fallbacks (unchanged): extractor → raw message + no events;
Gate 1 → PROCEED; Gate 2 → GENERALIZE; generator → pre-vetted GENERALIZE
template. The orchestrator never raises.

### 3.2 Orchestrator (WS-A)

```python
class PipelineOrchestrator:
    async def process_message(self, session_id, message: str, db: AsyncSession | None = None, *,
                              on_event: EventCallback | None = None,
                              ablation: AblationConfig | None = None,
                              regenerate: bool = False) -> MessageResponse
    async def warmup(self) -> dict[str, Any]
```
`regenerate=True`: re-run the last user message; extraction runs for the search
query but no new clinical events are persisted and no new user message is stored.

Stage events: `PipelineEvent(type="stage", stage=StageName.X, status="start"|"end"|"skipped", label=STAGE_LABELS[X])`.
`MessageResponse.explanation` is always filled (`Explanation` model).

### 3.3 AI layer (WS-B)

```python
# medbridge/ai/llm_wrapper.py
class ResilientLLMWrapper:
    async def call(self, call_name, system_prompt, user_prompt, output_schema, temperature=0.0,
                   max_tokens=512, *, model: str | None = None, seed: int | None = None) -> T | None
    async def warmup(self, models: list[str] | None = None) -> dict[str, bool]
    async def health(self) -> dict[str, Any]   # {"provider", "reachable", "models": {name: present}}
    async def close(self) -> None
def get_llm_wrapper() -> ResilientLLMWrapper
@contextmanager def llm_call_log() -> Iterator[list[LLMCallStat]]   # contextvar-scoped per turn
@dataclass class LLMCallStat: call_name, model, provider, latency_ms, prompt_tokens, completion_tokens, attempts, success, cached
def set_request_cache(cache: RequestCache | None) -> None           # evaluation memoisation hook
class RequestCache(Protocol): get(key: str) -> str | None; set(key: str, value: str) -> None

# medbridge/ai/context_extractor.py
async def extract(self, message, snapshot, history: list[ChatTurn] | None = None) -> ExtractorOutput | None
# medbridge/ai/context_gate.py      (signature unchanged)
async def evaluate(self, snapshot, message, raw_intent) -> ContextGateOutput | None
# medbridge/ai/evidence_gate.py     (signature unchanged)
async def evaluate(self, chunks, snapshot, query=None, message=None) -> EvidenceGateOutput | None
# medbridge/ai/response_generator.py
async def generate(self, action, chunks, snapshot, query=None, message=None, *,
                   mode: GeneratorMode = "grounded", history: list[ChatTurn] | None = None,
                   missing_fields: list[str] | None = None) -> ResponseGeneratorOutput | None
# chunks are rendered as <chunk index="1" source=... section=... page=...>; the model cites [1]..[5]
# medbridge/ai/evidence_calibration.py   (V1)
def evidence_features(query: str, chunks: list[RetrievedChunk], snapshot: dict) -> dict[str, float]
async def calibrated_evidence_decision(gate, chunks, snapshot, query, *, votes: int,
        settings: Settings) -> CalibratedDecision   # .action .rationale .votes .features .overridden
# medbridge/ai/grounding_verifier.py     (V2 / Gate 3)
class GroundingVerifier:
    async def verify(self, response_text: str, chunks: list[RetrievedChunk]) -> GroundingResult
    # GroundingResult: sentences[{text, cited, supported, entailment, markers}], unsupported_fraction,
    #                  cleaned_text, n_claims, n_unsupported
def get_grounding_verifier() -> GroundingVerifier
# medbridge/nlp/drugs.py
def normalize_drug(name: str) -> DrugInfo | None      # DrugInfo(generic, brand, drug_class)
def find_drugs(text: str) -> list[DrugInfo]
BRAND_TO_GENERIC: dict[str, str]                      # ≥ 60 cardiovascular brands incl. Indian
# medbridge/nlp/query_expansion.py
def expand_query(text: str) -> str                    # lay → clinical terms appended
```

### 3.4 Citations & small talk (WS-A)

```python
# medbridge/core/citations.py
@dataclass class CitationValidation: text: str; citations: list[CitationResponse];
                                     valid: int; stripped: int; cited_chunk_ids: list[str]
def validate_citations(text: str, chunks: list[RetrievedChunk]) -> CitationValidation
# markers renumbered in order of first appearance; markers not mapping to a chunk removed
# medbridge/core/smalltalk.py
def is_small_talk(text: str) -> str | None           # returns template key or None
```

### 3.5 Retrieval (WS-A) — Qdrant payload (WS-F writes, WS-A reads)

Payload keys per point: `chunk_id, chunk_text, guideline_id, doc_title, org, year,
section_title, page_number, page_end, source_url, topics (list[str]),
doc_sha256, chunk_index, content_type ("text"|"table")`.
`RetrievedChunk` gains optional `doc_title, org, year, page_end, topics, score_rrf, score_rerank`.
Retriever API unchanged (`hybrid_search(query, top_k)`), plus
`async def warmup() -> None` and in-process TTL/LRU caches for query embeddings
and results. Variant search for evaluation: `search(query, top_k, mode="hybrid"|"dense"|"sparse")`.

### 3.6 Privacy & persistence (WS-D)

```python
# medbridge/privacy/redaction.py
@dataclass class RedactionResult: text: str; count: int; categories: dict[str, int]
def redact(text: str) -> RedactionResult        # names, phones, emails, addresses/PIN, Aadhaar/PAN/SSN/MRN/
                                                # insurance ids, exact DOB (→ "[AGE: 54]"), URLs/IPs.
                                                # Never alters BP values, labs, drug names, doses.
# medbridge/privacy/crypto.py
class SessionCipher: encrypt(str) -> str; decrypt(str) -> str; encrypt_json(Any) -> dict; decrypt_json(dict) -> Any
async def create_session_key(db, session_id) -> None
async def get_session_cipher(db, session_id) -> SessionCipher | None   # None → erased
async def shred_session_key(db, session_id) -> None
# medbridge/privacy/tokens.py
def issue_session_token(session_id) -> tuple[str, datetime]
def verify_session_token(token: str, session_id) -> bool
def issue_admin_token(username: str) -> str
def verify_admin_token(token: str) -> bool
# medbridge/privacy/deps.py (FastAPI dependencies)
async def require_session_access(session_id: UUID, request: Request) -> None   # 401/403
async def require_admin(request: Request) -> str
# medbridge/privacy/middleware.py
def register_security_middleware(app: FastAPI, settings: Settings) -> None   # CORS, rate limit, size limit, headers/CSP
# medbridge/privacy/log_scrub.py
def scrub_pii_processor(logger, method_name, event_dict) -> dict
# medbridge/privacy/retention.py
async def purge_expired_sessions(db, older_than_days: int) -> int
async def retention_loop(stop_event: asyncio.Event) -> None
```

DB changes (one Alembic migration `g1…_privacy_trace_feedback`):
`audit_logs.trace JSONB` (plaintext, PHI-free), `sessions.erased_at`,
`session_keys(session_id PK/FK, wrapped_key TEXT, created_at)` (deletable),
`feedback(feedback_id, session_id, message_id, rating SMALLINT, comment TEXT enc, created_at)`.
Encrypted text columns store `enc:v1:<fernet token>`; encrypted JSONB stores
`{"enc": "v1", "ct": "<fernet token>"}`. Rationales are encrypted; actions,
chunk ids, latency and trace stay plaintext for metrics. Sessions are never
deleted (FK + append-only rules); erasure = shred key + delete mutable rows
(message_history, context_snapshots, feedback) + set `erased_at`.

### 3.7 HTTP API (all session routes require `Authorization: Bearer <session_token>` when `AUTH_REQUIRED`)

| Method & path | Owner | Notes |
|---|---|---|
| `POST /api/sessions` | D | 201 `{session_id, created_at, session_token, expires_at}` |
| `GET /api/sessions/{id}/history` | D | messages incl. `message_id`, `explanation` |
| `DELETE /api/sessions/{id}` | D | crypto-shred, 204 |
| `POST /api/sessions/{id}/messages` | A | `MessageResponse` (backwards compatible) |
| `POST /api/sessions/{id}/messages/stream` | A | SSE: `stage`, `meta`, `token`, `done`, `error` |
| `POST /api/sessions/{id}/regenerate` | A | SSE or JSON (`?stream=true`) |
| `POST /api/sessions/{id}/feedback` | D | `{message_id, rating: 1|-1, comment?}` → 201 |
| `GET /api/sessions/{id}/context` | D | snapshot + BP series with ACC/AHA categories |
| `POST /api/sessions/{id}/bp` | D | manual BP entry → BP_READING event via projector |
| `GET /api/sessions/{id}/summary?format=md|json` | D | clinician summary (cited) |
| `GET /api/config/public` | A | app name, emergency numbers, disclaimer, local-only flag, models, suggestions |
| `GET /health`, `GET /api/health` | A | + `ollama_reachable`, `models_loaded`, `qdrant_points` |
| `GET /api/metrics` | A | admin-only; p50/p95 per action & stage, action/gate counts, feedback stats |
| `POST /api/admin/login` | D | `{username, password}` → `{token}` |
| `GET /api/admin/audit?limit=` | D | recent audit rows (decrypted+redacted, or `[erased]`) |
| `GET /`, `/dashboard`, `/static/*` | E (mounted by A) | `medbridge.web.mount_web(app)` |

SSE wire format: `event: <type>\ndata: <json>\n\n`. `done` carries the full
`MessageResponse` JSON. The SSE route opens its own DB session (not the request
dependency) and commits it.

## 4. Quality bar
- Tests are hermetic by default: no live LLM, Postgres or Qdrant unless marked
  `@pytest.mark.live` / `integration` (skipped automatically when the service is down).
- `ruff` (line length 100) + `mypy` configured; new code fully typed.
- No fabricated numbers: every figure in README/report/paper is produced by a script
  from `evaluation/results/` or `reports/`.
