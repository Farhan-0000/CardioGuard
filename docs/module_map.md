# MedBridge — Module Map

> **Date:** 2026-10-10
> **Scope:** the 20 approved implementation modules (M-01 – M-20, as defined in `docs/development_backlog.md` and `docs/final_project_audit.md`) and the modules added during the 2026-10 finalization (M-21 – M-27).
> **Columns:** files (repo-relative), status, tests (file and number of collected test cases) and the ADL entries that govern the module (`docs/architecture_decision_log.md`).

## How this map was produced

- **Files** were taken from the working tree on 2026-10-10. Every module listed as "extended" or "rewritten" was changed between the Review-3 baseline commit `fad6fea` and `HEAD` (`git diff --stat fad6fea HEAD`).
- **Test counts** are the number of test cases each file contributes to a collection run. Reproduce them with:

  ```bash
  python -m pytest --collect-only -q        # 2,920 collected; the one `live` test is deselected by default
  ```

  Counts show coverage of intent, not a pass rate. The authoritative pass/fail and line-coverage figures are those produced for the final report's testing chapter (pending at the time of writing).
- **Markers.** Tests marked `postgres`, `qdrant` or `ollama` are skipped automatically when the service is unreachable, and `live` tests run only with `pytest -m live` (repository `conftest.py`, ADL-058). Where a test file depends on a service, this is noted.

**Status vocabulary**

| Status | Meaning |
| :--- | :--- |
| Implemented — extended | Original module kept; new behaviour added in 2026-10 without changing its role |
| Implemented — rewritten | Same role and public interface, internals substantially replaced in 2026-10 |
| Implemented — new | Module did not exist at Review 3 |
| Off by default | Implemented and tested, but disabled in production until evaluated (evaluation variant) |
| Legacy (kept working) | Superseded as the primary interface but maintained and tested |

---

## 1. Approved modules M-01 – M-20

| ID | Module | Files | Status (2026-10) | Tests (collected) | ADL |
| :--- | :--- | :--- | :--- | :--- | :--- |
| M-01 | Configuration | `medbridge/config.py`, `.env.example`, `scripts/setup_env.py` | Implemented — extended (local-first defaults, `LOCAL_ONLY_MODE`, per-role models, privacy settings, V1/V2 flags) | `medbridge/tests/unit/test_config.py` (6); `tests/privacy/test_setup_env.py` (5) | 027, 039, 040, 057 |
| M-02 | Database Layer | `medbridge/db/models.py`, `medbridge/db/connection.py`, `alembic.ini`, `alembic/versions/6a69a1c21f43_initial_schema.py`, `f7737fab1c53_fix_enum_hyphenation.py`, `g1a7c3e9d2b4_privacy_trace_feedback.py`, `h2b8d4f6a1c3_feedback_unique_legacy_marker.py` | Implemented — extended (`audit_logs.trace`, `sessions.erased_at`, `sessions.legacy_plaintext_rows`, `session_keys`, `feedback`; `hide_parameters=True`); append-only rules unchanged | `medbridge/tests/unit/test_db_models.py` (6); `medbridge/tests/integration/test_db_connection.py` (2, postgres); `medbridge/tests/integration/test_migrations.py` (5, postgres) | 012, 031, 032, 035, 036, 038 |
| M-03 | API Schemas & Middleware | `medbridge/api/schemas/enums.py`, `requests.py`, `responses.py`, `medbridge/api/middleware/error_handler.py` | Implemented — extended (`Explanation`, `GateDecision`, `RetrievedEvidence`, `StageTiming`, richer `CitationResponse`, session token fields; validation errors no longer echo input) | `medbridge/tests/unit/test_schemas.py` (33); `test_enums.py` (16); `test_error_handler.py` (11) | 009, 035, 036 |
| M-04 | API Routes | `medbridge/api/routes/__init__.py`, `sessions.py`, `messages.py`, `health.py`, `config.py`, `metrics.py`, `admin.py`, `features.py` | Implemented — extended (SSE stream, regenerate, erasure, feedback, context/BP/summary, public config, metrics, admin login/audit, readiness in `/health`) | `medbridge/tests/integration/test_routes.py` (16); `test_message_routes.py` (28); `test_health_route.py` (14); `test_core_public_config.py` (6); `test_metrics_route.py` (10); `test_admin_routes.py` (6, partly postgres); `test_feature_routes.py` (14, partly postgres) | 032, 033, 034, 054, 055 |
| M-05 | Session Manager & Turn Store | `medbridge/state/session_manager.py`, `medbridge/state/turn_store.py` | Implemented — rewritten (single-flush `persist_turn`, encrypted reads/writes, crypto-shred `erase_session`, row-locked re-projection, regenerate conflict check) | `medbridge/tests/integration/test_session_manager.py` (14, postgres); `test_turn_store.py` (9, postgres); `test_turn_concurrency.py` (9, postgres) | 031, 032, 036, 037, 038 |
| M-06 | State Projector | `medbridge/state/projector.py` | Implemented — extended (dose-change history for repeated MEDICATION_ADDED, order-preserving condition union); still a pure fold | `medbridge/tests/unit/test_state_projector.py` (39); `medbridge/tests/integration/test_projector_db.py` (2, postgres) | 001, 038, 044 |
| M-07 | Emergency Classifier | `medbridge/core/emergency_classifier.py`; development case files `tests/data/emergency_cases.json`, `tests/data/emergency_cases_hinglish.json` | Implemented — rewritten (26 English + 14 Hinglish rule IDs, clause-level negation, informational/historical suppression, typo normalisation, `classify_detailed()`); `classify()` signature unchanged | `medbridge/tests/unit/test_emergency_classifier.py` (270); `test_emergency_nl_patterns.py` (543) | 002, 045 |
| M-08 | Pipeline Orchestrator | `medbridge/core/orchestrator.py`, `medbridge/core/pipeline_types.py`, `medbridge/core/citations.py`, `medbridge/core/smalltalk.py` | Implemented — rewritten (traced stage methods over `TurnContext`, speculative retrieval, citation validation, small talk, effective question, per-session turn lock, regenerate, Gate 3 hook, `AblationConfig`) | `medbridge/tests/unit/test_orchestrator.py` (109); `test_pipeline_types.py` (18); `test_core_citations.py` (23); `test_core_smalltalk.py` (38); `medbridge/tests/integration/test_e2e_pipeline.py` (6, postgres + qdrant; one further `live` test deselected) | 027, 028, 029, 036, 037, 038, 043, 052, 054, 056 |
| M-09 | Response Templates | `medbridge/core/templates.py` | Implemented — extended (configurable emergency numbers, small-talk and SOFT-ASK templates, loop-breaker prefix) | `medbridge/tests/unit/test_templates.py` (52) | 015, 029 |
| M-10 | Resilient LLM Wrapper | `medbridge/ai/llm_wrapper.py`; shared prompt helpers `medbridge/ai/prompt_utils.py` (XML isolation, tag neutralisation, chunk rendering) | Implemented — rewritten (native Ollama `/api/chat`, JSON-schema `format`, `think`/`keep_alive`, per-role models, truncation handling, local-only enforcement, request-cache hook) | `medbridge/tests/unit/test_llm_wrapper.py` (75); `test_prompts_contract.py` (66, covers all four prompts) | 004, 039, 040, 056 |
| M-11 | Context Extractor (LLM 1) | `medbridge/ai/context_extractor.py`, `medbridge/ai/schemas/extractor.py` | Implemented — rewritten (conversation history for coreference, few-shot prompt, deterministic BP/drug post-processing) | `medbridge/tests/unit/test_context_extractor.py` (46) | 023, 041, 044 |
| M-12 | Context Gate (LLM 2, Gate 1) | `medbridge/ai/context_gate.py`, `medbridge/ai/schemas/context_gate.py` | Implemented — rewritten (rule-based prompt with 8 examples, 12-field `missing_fields` vocabulary, intent isolated as untrusted input); labels and temperature 0 / seed 42 unchanged | `medbridge/tests/unit/test_context_gate.py` (36) | 041, 043 |
| M-13 | Evidence Gate (LLM 3, Gate 2) | `medbridge/ai/evidence_gate.py`, `medbridge/ai/schemas/evidence_gate.py` | Implemented — rewritten (ordered rules with 7 examples, indexed chunks, message + query inputs); four labels and temperature 0 / seed 42 unchanged | `medbridge/tests/unit/test_evidence_gate.py` (26) | 024, 041, 043, 051 |
| M-14 | Response Generator (LLM 4) | `medbridge/ai/response_generator.py`, `medbridge/ai/schemas/response_generator.py` | Implemented — rewritten (index-based `[n]` citations, generator modes `grounded` / `vanilla` / `prompt_safety` / `direct` for the ablation, SOFT-ASK field prompts) | `medbridge/tests/unit/test_response_generator.py` (44); `test_prompts_contamination.py` (24, covers all prompts) | 041, 042, 056 |
| M-15 | Embedding Client | `medbridge/retrieval/embedder.py`, `medbridge/retrieval/cache.py` | Implemented — extended (concurrent dense + sparse encoding, LRU query-embedding cache, warm-up) | `medbridge/tests/unit/test_embedder.py` (12) | 022, 047, 049 |
| M-16 | Hybrid Retriever | `medbridge/retrieval/hybrid_retriever.py` | Implemented — rewritten (explicit RRF k = 60, LRU + TTL result cache, `RetrievalError` on infrastructure failure, dense/sparse/hybrid modes for evaluation) | `medbridge/tests/unit/test_hybrid_retriever.py` (20); `medbridge/tests/integration/test_hybrid_retriever_qdrant.py` (9, qdrant); `test_retrieval.py` (9) | 022, 048, 049 |
| M-17 | Cross-Encoder Reranker | `medbridge/retrieval/reranker.py` | Implemented — extended (automatic cuda > mps > cpu device, serialised `predict`, warm-up); model unchanged after ablation | `medbridge/tests/unit/test_reranker.py` (10) | 020, 049, 050 |
| M-18 | Ingestion Pipeline | `medbridge/ingestion/__main__.py`, `parser.py`, `html_parser.py`, `chunker.py`, `tokenizer.py`, `indexer.py`, `manifest.py`, `pipeline.py`, `state.py`, `stats.py`; `data/corpus_manifest.json`; `scripts/download_corpus.py` | Implemented — rewritten (27-document manifest with SHA-256 pins, PDF + HTML parsing, 400-token tokenizer-measured chunks, table chunks, cleaning, incremental hash-based runs, `--prune`, `--stats`) | `medbridge/tests/unit/test_ingestion.py` (23); `test_ingestion_chunker.py` (15); `test_ingestion_html.py` (9); `test_ingestion_manifest.py` (24); `test_ingestion_pdf.py` (18); `test_ingestion_pipeline.py` (13); `medbridge/tests/integration/test_guideline_collection_indexing.py` (9, qdrant) | 003, 046, 047 |
| M-19 | Application Entry Point | `medbridge/main.py`, `medbridge/logging_config.py` | Implemented — extended (lifespan warm-up and readiness, retention job, security middleware registration, web mount, PII log processor) | `medbridge/tests/integration/test_main_app.py` (18); `medbridge/tests/unit/test_logging.py` (3) | 034, 035, 049, 053 |
| M-20 | Streamlit Frontend | `medbridge/frontend/app.py`, `medbridge/frontend/components/chat.py`, `citations.py`, `disclaimer.py`, `sidebar.py`, `medbridge/frontend/services/api_client.py`, `session.py` | Legacy (kept working): bearer session token, token never in the URL, new session on 401/403/404, corpus-accurate texts | `medbridge/tests/unit/test_frontend_app.py` (14); `test_frontend_components.py` (18); `test_frontend_services.py` (24) | 007, 033, 053 |

`medbridge/tests/integration/test_database.py` is an empty placeholder file and contributes no tests.

---

## 2. Modules added in the 2026-10 finalization (M-21 – M-27)

| ID | Module | Files | Status (2026-10) | Tests (collected) | ADL |
| :--- | :--- | :--- | :--- | :--- | :--- |
| M-21 | Privacy & Security Layer | `medbridge/privacy/redaction.py`, `crypto.py`, `tokens.py`, `deps.py`, `middleware.py`, `log_scrub.py`, `retention.py`, `evaluation.py`, `README.md`; `scripts/pii_eval.py`; data `tests/privacy/pii_testset.json` (development), `tests/privacy/pii_testset_blind_v2.json` (blind held-out) | Implemented — new. Redaction evaluated once on the blind set with a frozen detector (`reports/privacy_redaction_metrics.md`) | `tests/privacy/test_redaction.py` (167); `test_redaction_blind.py` (6); `test_crypto.py` (15); `test_encryption_at_rest.py` (2, postgres); `test_tokens_and_deps.py` (13); `test_middleware.py` (12); `test_log_scrub_and_retention.py` (8, partly postgres); `medbridge/tests/unit/test_privacy_erasure_tokens.py` (7) | 030, 031, 032, 033, 034, 035 |
| M-22 | Web SPA & Admin Dashboard | `medbridge/web/__init__.py` (`mount_web`), `index.html`, `dashboard.html`, `css/app.css`, `base.css`, `dashboard.css`, `print.css`, `js/*.js` (12 modules: `actions`, `api`, `app`, `bp`, `dashboard`, `errors`, `format`, `markdown`, `normalize`, `sse`, `store`, `writeguard`), `js/ui/*.js` (14 modules), `dev_mock_server.py`, `screenshots.py`; Node unit tests `medbridge/web/tests/*.test.mjs` | Implemented — new. Development screenshots in `medbridge/web/dev_screenshots/` come from the mock server and are not results | `medbridge/tests/unit/test_web_static.py` (62); `test_web_js.py` (1; runs the Node suite, skipped when Node.js is not installed) | 053, 054, 055 |
| M-23 | Deterministic NLP | `medbridge/nlp/drugs.py`, `query_expansion.py`, `bp_parse.py` | Implemented — new (189 brand → generic mappings incl. 65 Indian brands, lay → clinical query expansion, first-person BP-reading parser) | `medbridge/tests/unit/test_nlp_drugs.py` (146); `test_nlp_query_expansion.py` (25); `test_nlp_bp_parse.py` (61) | 044 |
| M-24 | Complementary Features | `medbridge/features/bp.py` (ACC/AHA categories, BP series), `medbridge/features/summary.py` (clinician summary); HTTP surface in M-04 (`features.py`, `metrics.py`, `admin.py`) and UI in M-22 | Implemented — new | `medbridge/tests/unit/test_features_bp.py` (17); `test_features_summary.py` (5); route tests listed under M-04 | 055 |
| M-25 | Evaluation Package | `evaluation/run.py`, `judge.py`, `judges/common.py`, `judges/llm_judge.py`, `judges/nli_judge.py`, `analyze.py`, `calibrate.py`, `metrics.py`, `outcomes.py`, `degraded.py`, `cache.py`, `evidence.py`, `manifest.py`, `io.py`, `paths.py`, `perf.py`, `plots.py`, `human_annotation.py`, `model_selection.py`, `retrieval_eval.py`, `README.md`; datasets `evaluation/datasets/hallubench_v1.json`, `hallubench_schema.json`, `schema.py`, `HALLUBENCH.md`, `retrieval_eval_v1.json`; drivers `scripts/run_study.sh`, `scripts/study_ollama.sh` | Implemented — new. Results present: retrieval metrics and reranker ablation (`evaluation/results/retrieval/`), model selection round 1 (`evaluation/results/model_selection_round1_prefix/`). Hallucination study (C0–C5, V1, V2), judges, calibration and statistics: **pending** | `tests/evaluation/test_analyze.py` (21); `test_annotation_perf_models.py` (5); `test_cache.py` (10); `test_calibrate.py` (14); `test_degraded.py` (17); `test_evidence_manifest.py` (12); `test_hallubench_dataset.py` (34, partly qdrant); `test_llm_judge.py` (50); `test_metrics.py` (40); `test_nli_judge.py` (14); `test_retrieval_eval.py` (5); `test_run.py` (28); `test_schema.py` (20) | 050, 056, 057 |
| M-26 | Additional Verification Gates (V1, V2) | `medbridge/ai/evidence_calibration.py` (V1 calibrated Evidence Gate), `medbridge/ai/grounding_verifier.py` (V2 Gate 3) | Off by default (`EVIDENCE_GATE_CALIBRATION_ENABLED=false`, `GROUNDING_VERIFIER_ENABLED=false`); thresholds to be calibrated on the dev split; results pending | `medbridge/tests/unit/test_evidence_calibration.py` (36); `test_grounding_verifier.py` (54) | 051, 052 |
| M-27 | Benchmark, Adversarial & Operations Harnesses | `tests/benchmark/benchmark_runner.py`, `evaluator.py`, `live.py`, `citations.py`, `report_generator.py`, `schemas.py`, `tests/benchmark/vignettes/*.json` (275 MedBridge-AQ vignettes, changes in `vignettes/CHANGES.md`); `tests/adversarial/adversarial_runner.py`, `rabbits_runner.py`, `payloads.py`, `stubs.py`; `scripts/smoke_test.py`, `load_test.py`, `latency_probe.py`, `llm_sanity_check.py` | Implemented — rebuilt. `reports/medbridge_aq_benchmark_report.md` and `reports/rabbits_benchmark_report.md` currently hold bannered **mock-mode** output (harness checks, not model behaviour); `reports/benchmark_runs/*_dryrun.*` are small live dry runs. Full live AQ, RABBITS, adversarial and load-test runs: **pending** | `medbridge/tests/unit/test_benchmark_runner.py` (26, partly postgres); `tests/adversarial/test_rabbits_and_runner.py` (19); `test_security_adversarial.py` (73, partly postgres); `tests/scripts/test_smoke_test.py` (6); `test_load_test.py` (9); `test_latency_probe.py` (5) | 045, 056, 058 |

---

## 3. Cross-cutting placement notes

- **Turn persistence** (`medbridge/state/turn_store.py`) is listed under M-05 because it replaced the session manager's write path; the projector (M-06) remains the only code that computes snapshots.
- **Citation validation and small talk** (`medbridge/core/citations.py`, `smalltalk.py`) and the ablation switches (`medbridge/core/pipeline_types.py`) belong to M-08 because only the orchestrator calls them.
- **Prompt helpers** (`medbridge/ai/prompt_utils.py`) are listed under M-10 but are used by M-11 – M-14 and M-26; the XML-isolation contract tests (`test_prompts_contract.py`) and the decontamination tests (`test_prompts_contamination.py`) cover all four prompts.
- **Security dependencies** (`medbridge/privacy/deps.py`) are part of M-21 but are attached to the routes of M-04.
- **Module documentation** lives next to the code: `medbridge/README.md` and the `README.md` files in `ai/`, `core/`, `ingestion/`, `privacy/`, `retrieval/`, `state/` and `frontend/`, plus `evaluation/README.md`.

## 4. Traceability to the Review-4 rubric

Rubric row 2 ("all 20 modules M-01 … M-20 working + new features") maps to section 1 of this file for M-01 – M-20 and to section 2 for the new modules. Each row names its tests, so a reviewer can run the tests of a single module, for example:

```bash
python -m pytest medbridge/tests/unit/test_context_gate.py -q          # M-12
python -m pytest tests/privacy -q                                       # M-21
```

[TO BE FILLED: name of the team member responsible for each module, for the individual contribution statement]
