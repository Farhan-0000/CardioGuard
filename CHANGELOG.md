# Changelog

All changes made to MedBridge since the version the team handed over for finalisation. That baseline is commit `fad6fea` ("team wip baseline"), package version 3.0.0. The current package version is 4.0.0 (`pyproject.toml`).

Entries are grouped by area. Each one names the commits that carry it and links the matching entry in the [architecture decision log](docs/architecture_decision_log.md). ADL ids are filled in by `scripts/render_docs_results.py` from the log's headings, so the links stay correct while the log is being extended. "ADL entry pending" means the log has no matching entry yet.

Nothing here changes the architecture. The eight-step workflow, the five actions, both gates as separate decision points at temperature 0.0 / seed 42, the loop-breaker at 2, template bypass for ABSTAIN and ESCALATE, XML isolation, append-only `clinical_events` / `audit_logs` and the never-throw orchestrator are all kept (constitution §12).

## Baseline as received (`fad6fea`)

For comparison, the received version had:

- one monolithic `_execute_pipeline` method of about 390 lines (in a 539-line module), several database flushes per turn spread over the message, projector and audit helpers, and lazy model loading on the first query (Review-3 latencies of 25–36 s for ABSTAIN/GENERALIZE);
- a Streamlit UI, sessions identified by a bare UUID, no PII redaction and no encryption at rest;
- 3 guideline PDFs (AHA/ACC 2025, ESC 2024, NICE NG136) in 905 chunks, many longer than the embedder's 512-token window;
- `gemma3:4b` through Ollama's OpenAI-compatible endpoint;
- a MedBridge-AQ benchmark report generated in mock mode (12 vignettes, no LLM called), no ablation study and no baselines;
- 480 passing and 5 failing tests (`reports/baseline/pytest_baseline.txt`);
- a README with Windows-only absolute links and outdated paths.

## Planning and contracts

- Finalisation plan with phases, work-stream file ownership and frozen interface contracts (`docs/finalization_plan.md`). Local-first configuration defaults, an isolated `medbridge_test` database for the test suite, and a baseline latency probe recorded before any change (`reports/performance/before_gemma3-4b.json`). Commit `e6c04db`.

## Core pipeline (`medbridge/core`, `medbridge/retrieval`, API routes)

- The orchestrator is split into traced stage methods over a turn context (privacy → emergency → load state → small talk → extraction → Context Gate → retrieval/rerank → Evidence Gate → generation → verification → persistence), with one persistence call and one flush per turn. Same order, same outputs, same fallbacks. Commit `02a547f`; <!-- ADL:Orchestrator Decomposition -->[ADL-036](docs/architecture_decision_log.md)<!-- /ADL -->.
- Speculative retrieval (`SPECULATIVE_RETRIEVAL`, on by default) overlaps retrieval with Gate 1 and discards the result on SOFT-ASK, so Gate 1 still controls whether evidence is used. Commit `02a547f`; <!-- ADL:Speculative Retrieval -->[ADL-027](docs/architecture_decision_log.md)<!-- /ADL -->.
- Server-side citation validation: markers that do not map to a retrieved chunk are removed, markers are renumbered, citations are rebuilt from chunk metadata, and an ANSWER with no valid citation becomes GENERALIZE. Commit `02a547f`; <!-- ADL:Citation Validation -->[ADL-028](docs/architecture_decision_log.md)<!-- /ADL -->.
- Deterministic small-talk handling (greetings, thanks, "who are you") with claim-free GENERALIZE templates and no LLM call. Commit `02a547f`; <!-- ADL:Small-Talk -->[ADL-029](docs/architecture_decision_log.md)<!-- /ADL -->.
- Explicit reciprocal rank fusion with k = 60 (`RrfQuery(rrf=Rrf(k=60))`). The old call used Qdrant's un-parameterised RRF fusion, whose constant is not 60, so the documented k = 60 was not what ran. Commit `02a547f`; <!-- ADL:RRF -->[ADL-048](docs/architecture_decision_log.md)<!-- /ADL -->.
- LRU/TTL caches for query embeddings and fused results, concurrent dense and sparse encoding, automatic reranker device selection (CUDA, then MPS, then CPU), and warm-up of embedders, reranker and Ollama models in the FastAPI lifespan with readiness reported by `/health`. Commit `02a547f`; <!-- ADL:Caches|Warm-Up -->[ADL-049](docs/architecture_decision_log.md)<!-- /ADL -->.
- Server-sent events endpoints for messages and regenerate (`stage`, `meta`, `token`, `done`, `error`); regenerate; `/api/metrics` (latency percentiles, action and gate counts, Gate 3 and feedback statistics); `/api/config/public`; a per-stage trace in each audit row. Commit `02a547f`; <!-- ADL:Server-Sent|Streaming|SSE -->[ADL-054](docs/architecture_decision_log.md)<!-- /ADL -->.
- Regenerate restarts Gate 1 from the loop-breaker counter recorded before the original turn, so repeated regenerates neither trip nor undo the loop-breaker. Commit `10aef5e`; <!-- ADL:Regenerate -->[ADL-037](docs/architecture_decision_log.md)<!-- /ADL -->.
- Per-session turn serialisation, plus row locking and re-projection of the snapshot in the turn store, so the snapshot always equals the fold of its events under concurrent writers. Emergency turns are never queued behind another turn. Commit `10aef5e`; <!-- ADL:Row Locking|Turn Serialisation -->[ADL-038](docs/architecture_decision_log.md)<!-- /ADL -->.
- Multi-turn follow-ups: after a SOFT-ASK, both gates and the generator judge the original question plus the patient's follow-up details. Commit `10aef5e`; <!-- ADL:Effective Question -->[ADL-043](docs/architecture_decision_log.md)<!-- /ADL -->.
- Retrieval outages are reported (`retrieval_status = failed`) instead of being treated as an empty result; Gate 2 still decides on empty evidence. Erased sessions return 410 before any pipeline work. Commits `02cf985`, `10aef5e`.

## AI layer and deterministic NLP (`medbridge/ai`, `medbridge/nlp`)

- The LLM wrapper calls Ollama's native `/api/chat` with a JSON-schema `format` (grammar-constrained output), seed and context size, `keep_alive`, and thinking control for models that support it. Each call names its model, so the gates and the generator can use different models (`LLM_MODEL_GATES`, `LLM_MODEL_GENERATOR`). Commit `1f88b75`; <!-- ADL:Structured Outputs -->[ADL-039](docs/architecture_decision_log.md)<!-- /ADL -->.
- `LOCAL_ONLY_MODE` (default on) fails closed: only a loopback Ollama host and non-cloud model tags are allowed, and anything else blocks the call so the specification's fallbacks apply. Commits `1f88b75`, `10aef5e`; <!-- ADL:LOCAL_ONLY|Local Inference -->[ADL-040](docs/architecture_decision_log.md)<!-- /ADL -->.
- Prompts rewritten for cardiology centred on hypertension, with ordered rules and few-shot examples for false premises, invented drugs, doses the guidelines do not state, children and pregnancy. The few-shot examples were later decontaminated against HalluBench, and a test enforces this. Commits `1f88b75`, `10aef5e`; <!-- ADL:Prompt Redesign|Decontamination -->[ADL-041](docs/architecture_decision_log.md)<!-- /ADL -->.
- Evidence is rendered as numbered chunks and the model cites `[n]`; the server rebuilds the citations. Commit `1f88b75`; <!-- ADL:Index-Based Citation -->[ADL-042](docs/architecture_decision_log.md)<!-- /ADL -->.
- Drug normaliser (189 brands, 65 Indian, with fuzzy matching), lay-to-clinical query expansion, and a BP parser that records only first-person measured readings, never targets or hypothetical values. Commits `1f88b75`, `10aef5e`; <!-- ADL:Drug Normalisation|Query Expansion|BP-Reading -->[ADL-044](docs/architecture_decision_log.md)<!-- /ADL -->.
- V1 calibrated Evidence Gate (permutation voting over evidence orderings plus calibrated reranker-score and entity-coverage thresholds) and V2 grounding verifier ("Gate 3", local NLI check of every cited claim). Both are optional, off by default, and can only make a reply more conservative. Commits `1f88b75`, `02cf985`, `10aef5e`; <!-- ADL:Calibrated Evidence Gate -->[ADL-051](docs/architecture_decision_log.md)<!-- /ADL --> and <!-- ADL:Grounding Verifier|Gate 3|Triple -->[ADL-052](docs/architecture_decision_log.md)<!-- /ADL -->.
- Generator modes `vanilla` and `prompt_safety` so the study's C1 and C2 baselines are real baselines rather than the production prompt. Truncated model output is retried with a larger budget and never repaired or cached. Tag escaping also covers unterminated tags, look-alike brackets and invisible characters. Commits `02cf985`, `10aef5e`.

## Emergency fast-path (`medbridge/core/emergency_classifier.py`)

- Rebuilt as a deterministic rules engine for colloquial phrasing ("150 over 95", "my chest feels heavy", "can't move my arm"), negation scope ("no chest pain"), typo tolerance, and suppression of purely informational or historical mentions. `classify_detailed()` explains every decision. Still no LLM, no network and no randomness, and still before any LLM call. Commit `461d3ea`; <!-- ADL:Emergency Classifier -->[ADL-045](docs/architecture_decision_log.md)<!-- /ADL -->.
- Roman-script Hindi / Indian-English red flags ("seene mein dard", "saans nahi aa rahi") with Hindi post-position negation. English behaviour is unchanged. Commit `420ca26`.

## Privacy and persistence (`medbridge/privacy`, `medbridge/state`, `medbridge/db`, `alembic`)

- Deterministic PII/PHI redaction before any LLM call, log line or write; a date of birth becomes an age, and clinical values are protected. Commits `89d3693`, `9717a2a`, `10aef5e`; <!-- ADL:Redaction -->[ADL-030](docs/architecture_decision_log.md)<!-- /ADL -->.
- Envelope encryption at rest with one data key per session. Commit `89d3693`; <!-- ADL:Envelope Encryption -->[ADL-031](docs/architecture_decision_log.md)<!-- /ADL -->.
- Right to erasure by crypto-shredding: the key and the mutable rows are deleted, the append-only rows stay untouched but unreadable. Commit `89d3693`; <!-- ADL:Crypto-Shredding -->[ADL-032](docs/architecture_decision_log.md)<!-- /ADL -->.
- Signed, expiring session tokens on every session route. The erasure route accepts an expired but correctly signed token for the same session so a user can always delete their data. Commits `89d3693`, `10aef5e`; <!-- ADL:Session Tokens -->[ADL-033](docs/architecture_decision_log.md)<!-- /ADL -->.
- Admin login for the dashboard, rate and request-size limits, security headers and a strict CSP, tightened CORS. Commit `89d3693`; <!-- ADL:Admin Authentication|Security Middleware -->[ADL-034](docs/architecture_decision_log.md)<!-- /ADL -->.
- PII scrubbing in logs and an inactivity retention job. Commit `89d3693`; <!-- ADL:Log Scrubbing|Retention -->[ADL-035](docs/architecture_decision_log.md)<!-- /ADL -->.
- Migrations `g1a7c3e9d2b4` (trace column, session keys, feedback, erasure marker) and `h2b8d4f6a1c3` (one feedback row per message, legacy-plaintext marker). The append-only rules are unchanged. Commits `89d3693`, `10aef5e`.
- New feature routes: feedback, patient context with BP categories, manual BP entry, clinician summary, used by the BP log, the summary download and the admin dashboard. Commit `89d3693`; <!-- ADL:BP Log -->[ADL-055](docs/architecture_decision_log.md)<!-- /ADL -->.
- PII redaction evaluated on a development set and once on a blind held-out set written without access to the detector (`reports/privacy_redaction_metrics.json`). Commit `9717a2a`.

## Web app (`medbridge/web`, `medbridge/frontend`)

- New single-page app and admin dashboard served by FastAPI at `/` and `/dashboard`: plain HTML/CSS/ES modules, no build step, no third-party scripts, an escape-first markdown renderer under a strict CSP, live pipeline progress over SSE, action badges, emergency card, citation drawer, "Why this answer?" panel, patient context card with erasure, BP trend chart, feedback, regenerate, export and print. Commit `f9eee6e`; <!-- ADL:Web SPA -->[ADL-053](docs/architecture_decision_log.md)<!-- /ADL -->.
- Streamlit kept as a legacy client, now using session tokens. Commit `f9eee6e`.
- Safer error handling: only 404/410 count as "erased", 429 shows a retry countdown, one write at a time per conversation, and the tab title never contains message text. Commit `10aef5e`.

## Knowledge base (`medbridge/ingestion`, `data`)

- Corpus grown from 3 to 27 curated documents (guidelines, society statements, one open-access review, US government patient pages), each pinned by SHA-256 with its licence in `data/corpus_manifest.json`, plus a downloader. Commit `d1bcb10`; <!-- ADL:Corpus Expansion -->[ADL-046](docs/architecture_decision_log.md)<!-- /ADL -->.
- Chunks measured with the embedder's own tokenizer (400 tokens, 60 overlap), table-aware chunking for PDF and HTML, removal of headers, footers and reference lists, de-duplication, incremental hash-based ingestion and a `--stats` report. Commit `d1bcb10`; <!-- ADL:400-Token|Table-Aware|Incremental Ingestion -->[ADL-047](docs/architecture_decision_log.md)<!-- /ADL -->.
- Retrieval evaluation set (70 labelled queries) and a reranker ablation. The default cross-encoder was kept. Commits `d1bcb10`, `02cf985`; <!-- ADL:Reranker Retained -->[ADL-050](docs/architecture_decision_log.md)<!-- /ADL -->.

## Evaluation (`evaluation/`)

- MedBridge-HalluBench v1 (247 synthetic items, 298 turns, six categories, stratified dev/test split with seed 42), ablation presets C0–C5 and variants V1/V2/V12 run through the real orchestrator, an LLM judge from a different model family, a deterministic NLI judge, a blinded 50-item human-annotation sheet, statistics (Wilson, bootstrap, exact McNemar with Holm, Cochran's Q, effect sizes), figures, model selection and a before/after latency table. Commit `4924f9a`; <!-- ADL:Evaluation Harness|HalluBench -->[ADL-056](docs/architecture_decision_log.md)<!-- /ADL -->.
- Validity safeguards from the code review: degraded-turn accounting with a 5 % headline limit, calibration-gated V1/V2 runs, a request cache namespaced by model digest, a narrower non-clinical claim filter, citation metrics that check the cited chunk, and item-level pairing only for complete items. Commit `10aef5e`.
- Application model chosen by a dev-split model-selection experiment on the full pipeline (`evaluation/model_selection.py`); the choice stays configurable through `.env`. The first round's results are in `evaluation/results/model_selection_round1_prefix/` and a re-run with the current prompts is pending. Commit `4924f9a` (harness), results in the working tree; <!-- ADL:Model Selection -->[ADL-057](docs/architecture_decision_log.md)<!-- /ADL -->.
- Tests always use a separate `medbridge_test` database, and live evaluation runners require a dedicated evaluation database. Commits `e6c04db`, `0a6dbc5`; <!-- ADL:Database Isolation -->[ADL-058](docs/architecture_decision_log.md)<!-- /ADL -->.
- `scripts/run_study.sh` runs the full protocol resumably, and `scripts/study_ollama.sh` starts a dedicated Ollama server for it (working tree).

## Benchmarks, security tests and operations scripts

- MedBridge-AQ rebuilt with 275 vignettes and a live mode that seeds each vignette's context as typed events; mock mode is labelled as a routing simulator, not a result. RABBITS over all 189 brands. A live adversarial runner with 19 injection and jailbreak payloads and deterministic scoring. Live runners refuse the development database. Commit `0a6dbc5`; <!-- ADL:Benchmark|RABBITS|Adversarial -->ADL entry pending<!-- /ADL -->.
- `scripts/smoke_test.py` (every action path, PII, SSE, auth, erasure), `scripts/load_test.py` (1/5/10 users), `scripts/latency_probe.py`, `scripts/pii_eval.py`, `scripts/setup_env.py` (generates keys and passwords). Commits `e6c04db`, `89d3693`, `0a6dbc5`, `9717a2a`.
- `medbridge/cli.py`: cross-platform launcher (`doctor`, `up`, `ingest`, `smoke`) for Windows 11 and macOS (working tree).

## Fixed

- Two integration rounds and an adversarial code review: 37 confirmed findings, all fixed in `10aef5e`. They cover regenerate and the loop-breaker counter, concurrent snapshot writes, degraded turns in the study, retrieval outages, V1 calibration order, the C1 baseline prompt, the non-clinical claim filter, address redaction leaks, erasure after token expiry, non-ASCII tokens, BP targets recorded as readings, Gate 3 skipping bold lines, truncated output, few-shot contamination, raw text outside the isolation block, drug-name false positives, and several UI error paths. The finding list is in `docs/notes/code_review_findings.md`. Commits `02cf985`, `10aef5e`.
- The five failing tests of the baseline run: the four guideline-collection integration tests were updated with the re-indexed corpus (`d1bcb10`), and the loop-breaker end-to-end test, which depended on a live LLM, was rewritten with deterministic fakes for every LLM call (`02a547f`). A separate opt-in test still exercises the real LLM (`pytest -m live`).
- The generator prompt's JSON example lacked citation markers, so local models dropped them and answers were downgraded. Commit `02cf985`.

## Documentation and references

- 90 verified references, patents and venue research (`report/references/`, `report/paper/VENUES.md`). Commit `4817e42`.
- README rewritten (overview, novelty, Mermaid architecture, features, results generated from files, accurate structure, privacy, limitations, team, license placeholder, disclaimer), this changelog, the Review 4/5 checklist (`docs/review4_5_checklist.md`), the demo and viva script (`docs/DEMO_SCRIPT.md`), and `scripts/render_docs_results.py`, which writes every result number into these documents from the result files (working tree).
- Architecture decision log extended from ADL-026 with the decisions above (`docs/architecture_decision_log.md`, working tree).
- Report build script with chapter sources, result-driven tables and figure scripts, and the updated architecture, turn-sequence and SOFT-ASK state diagrams (`report/build_report.py`, `report/content/`, `report/diagrams/`, `report/figures/`; working tree, in progress).
