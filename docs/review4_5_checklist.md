# Review 4, Review 5 and Draft Review: evidence checklist

This page maps every rubric row of the three final reviews (master brief §11) to the evidence a panel member can open: files, report sections, figures and demo steps. Use it to prepare each review and to answer "where is that?" during the viva.

**Status legend**

- **done**: the evidence exists and can be shown now.
- **pending results**: the code and pipeline are done, but the number comes from an experiment run that has not finished. The README and `docs/DEMO_SCRIPT.md` show *pending* until `scripts/render_docs_results.py` finds the result file.
- **team action**: something only the students or the guide can do (fill a placeholder, label the annotation sheet, submit the paper, rehearse).

Before each review:

1. `python scripts/render_docs_results.py` to refresh every number in the README, the demo script and the evidence check at the bottom of this page.
2. `python -m medbridge.cli doctor` on the demo laptop.
3. Rebuild the report with `report/build_report.py` so its tables come from the latest result files.

Demo steps refer to [`docs/DEMO_SCRIPT.md`](DEMO_SCRIPT.md). Report sections follow the institutional structure used by the report sources in `report/content/` (Chapters 1–5, references, appendices).

## Review 4 (25 marks)

| # | Parameter (marks) | Evidence | Status |
|---|---|---|---|
| 1 | Action taken on previous review comments (2) | `CHANGELOG.md` ("Baseline as received" and every area since `fad6fea`); the audit-item table below; report front matter "Action Taken on Previous Review Comments" table (Comment, Action Taken, Evidence); baseline numbers in `docs/finalization_plan.md` §0, `reports/baseline/pytest_baseline.txt`, `reports/performance/before_gemma3-4b.json` | **done** for the engineering actions. **team action**: copy the exact Review-3 and draft-review comments into the report's Action Taken table ([TO BE FILLED: review comments]) |
| 2 | Completion of the approved project scope (3) | Module table below (M-01 to M-20 plus the new modules), each with code and tests; live end-to-end check: <!-- RESULT:smoke -->16/16 checks passed against a live local stack on 2026-10-09 (`reports/smoke_test_latest.json`)<!-- /RESULT -->; report Ch. 4 "Implementation" | **done** |
| 3 | Functionality and integration of modules (3) | `reports/smoke_test_latest.json`; end-to-end tests `medbridge/tests/integration/test_e2e_pipeline.py`; never-throw and fallback tests in `medbridge/tests/unit/test_orchestrator.py`; load test `reports/performance/load_test.json` (load test result: <!-- RESULT:load -->*pending* (result file not generated yet)<!-- /RESULT -->); demo steps 1–8 | **done** (smoke, E2E). **pending results**: live load test |
| 4 | Technical accuracy and quality of implementation (3) | Decisions with rationale and architecture-preservation notes in `docs/architecture_decision_log.md` (ADL-027 onward); invariants in `docs/project_constitution.md` §12; staged orchestrator `medbridge/core/orchestrator.py`; ruff and mypy configuration in `pyproject.toml`; 37 review findings fixed (`docs/notes/code_review_findings.md`); RRF k = 60 bug fix and retrieval design backed by `evaluation/results/retrieval/metrics.json`; report Ch. 3 "Methodology and System Design" | **done** |
| 5 | Testing and validation (3) | Unit and integration tests `medbridge/tests/`; security, benchmark, evaluation, privacy and script tests under `tests/`; UI module tests `medbridge/web/tests/`; emergency case files `tests/data/emergency_cases.json` and `tests/data/emergency_cases_hinglish.json`; PII sets `tests/privacy/pii_testset.json` and `tests/privacy/pii_testset_blind_v2.json` with `reports/privacy_redaction_metrics.json`; HalluBench `evaluation/datasets/hallubench_v1.json`; test summary: <!-- RESULT:tests -->*pending* (result file not generated yet)<!-- /RESULT -->; live AQ: <!-- RESULT:aq -->*pending* (result file not generated yet)<!-- /RESULT -->; live RABBITS: <!-- RESULT:rabbits -->*pending* (result file not generated yet)<!-- /RESULT -->; report appendix test-case table | **done** (suites, PII, emergency sets). **pending results**: coverage report, live AQ, RABBITS and adversarial runs, HalluBench study |
| 6 | Results analysis and interpretation (2) | `evaluation/results/summary.json`, `evaluation/results/tables/`, `evaluation/results/error_analysis.md`; figures `reports/figures/hallucination_rate_by_condition.png`, `reports/figures/unsafe_answer_rate_unanswerable.png`, `reports/figures/ablation_waterfall.png`, `reports/figures/risk_coverage.png`, `reports/figures/confusion_matrix_C5.png`; already available: `reports/figures/retrieval_metrics.png`, `reports/figures/corpus_composition.png`, `reports/figures/model_selection.png`; report Ch. 4 "Results and Observations", "Comparison and Analysis" | **done**: retrieval, corpus, PII. **pending results**: hallucination study, model-selection re-run |
| 7 | Technical refinement, limitations and future scope (2) | Before/after latency: baseline `reports/performance/before_gemma3-4b.json`, comparison `reports/performance/latency_table.csv` (<!-- RESULT:perf -->*pending* (result file not generated yet)<!-- /RESULT -->); optimisations listed in `CHANGELOG.md` (warm-up, speculative retrieval, caches, single flush, RRF fix, structured outputs); README "Limitations"; report Ch. 4 "Limitations" and Ch. 5 "Limitations and Future Enhancements" | **done** (refinements, limitations). **pending results**: "after" latency probe |
| 8 | Completeness and technical quality of the final report (2) | Report build `report/build_report.py` → `report/MedBridge_Final_Report.docx`; sources `report/content/`; diagrams `report/figures/architecture_pipeline.png`, `report/figures/fsm_soft_ask.png`, `report/figures/sequence_turn.png`; references `report/references/references.json` | **pending results** (report build in progress; Ch. 4 numbers wait for the study). **team action**: fill the yellow [TO BE FILLED] placeholders |
| 9 | Research publication / IP submission (2) | Paper draft `report/paper/MedBridge_IEEE_Paper.docx` and LaTeX `report/paper/latex/`; venue research `report/paper/VENUES.md` (proof-of-submission rules in its §6); patent comparison `report/references/patents.json`; optional disclosure draft `report/paper/patent_disclosure_draft.md` | **team action**: choose the venue ([TO BE FILLED: venue]), submit, and attach the submission proof to the report appendix |
| 10 | Demonstration, individual contribution, readiness (3) | `docs/DEMO_SCRIPT.md` (10-minute flow, fallbacks, viva answers); slides `report/MedBridge_Review_Presentation.pptx`; setup `HOW_TO_RUN.md` and `medbridge/cli.py` (`doctor`, `up`); screenshots `report/screenshots/`; contribution statement in the report appendix and README "Team" | **done** (script, launcher). **team action**: rehearse on the Windows laptop, fill [TO BE FILLED: contribution split], record guide feedback |

### Action taken on the audit items (master brief §1.2)

| Audit item | Action taken | Evidence |
|---|---|---|
| Benchmark result was mock (no LLM called) | Mock mode relabelled as a routing simulator; live MedBridge-AQ mode; full live hallucination study with two judges and statistics | `tests/benchmark/benchmark_runner.py`, `evaluation/run.py`, `scripts/run_study.sh`, `evaluation/README.md` |
| Ablation study and 3 baselines never implemented | Presets C0–C5 (plus C1G reference) and variants V1, V2, V12, run through the real orchestrator | `medbridge/core/pipeline_types.py`, `evaluation/analyze.py` |
| PII anonymisation deferred | Redaction before any LLM call or write, encryption at rest, crypto-shred erasure, log scrubbing, retention | `medbridge/privacy/`, `tests/privacy/`, `reports/privacy_redaction_metrics.json` |
| No authentication (session id only) | Signed, expiring session tokens; admin login for the dashboard | `medbridge/privacy/tokens.py`, `medbridge/privacy/deps.py` |
| Dependency on external LLM APIs | Ollama-only by default, `LOCAL_ONLY_MODE` fails closed | `medbridge/ai/llm_wrapper.py`, `medbridge/config.py` |
| Review-3 latencies (ABSTAIN 25.2 s, GENERALIZE 35.8 s, as shown in the Review-3 deck) | Warm-up at start-up, speculative retrieval, caches, single flush, structured outputs; before/after probe | `medbridge/main.py`, `medbridge/retrieval/cache.py`, `reports/performance/` |
| Sequential pipeline, monolithic orchestrator, several flushes | Stage methods, concurrent dense/sparse encoding, one flush per turn | `medbridge/core/orchestrator.py`, `medbridge/state/turn_store.py` |
| Streamlit UI with CSS hacks | New SPA and dashboard; Streamlit kept as legacy | `medbridge/web/`, `medbridge/frontend/` |
| Corpus of 3 PDFs | 27 curated, hash-pinned documents; retrieval evaluation | `data/corpus_manifest.json`, `reports/corpus_stats.md`, `evaluation/datasets/retrieval_eval_v1.json` |
| README with Windows-only links and outdated paths | README rewritten; structure taken from the actual tree | `README.md` |

### Approved modules (M-01 to M-20) and new modules

| Module | Code | Tests |
|---|---|---|
| M-01 Configuration | `medbridge/config.py` | `medbridge/tests/unit/test_config.py` |
| M-02 Database layer | `medbridge/db/models.py`, `medbridge/db/connection.py`, `alembic/versions/` | `medbridge/tests/unit/test_db_models.py`, `medbridge/tests/integration/test_database.py`, `medbridge/tests/integration/test_migrations.py` |
| M-03 API schemas and middleware | `medbridge/api/schemas/`, `medbridge/api/middleware/`, `medbridge/privacy/middleware.py` | `medbridge/tests/unit/test_schemas.py`, `medbridge/tests/unit/test_error_handler.py`, `tests/privacy/test_middleware.py` |
| M-04 API routes | `medbridge/api/routes/` | `medbridge/tests/integration/test_routes.py`, `medbridge/tests/integration/test_message_routes.py`, `medbridge/tests/integration/test_feature_routes.py`, `medbridge/tests/integration/test_admin_routes.py`, `medbridge/tests/integration/test_metrics_route.py`, `medbridge/tests/integration/test_health_route.py` |
| M-05 Session manager and turn store | `medbridge/state/session_manager.py`, `medbridge/state/turn_store.py` | `medbridge/tests/integration/test_session_manager.py`, `medbridge/tests/integration/test_turn_store.py`, `medbridge/tests/integration/test_turn_concurrency.py` |
| M-06 State projector | `medbridge/state/projector.py` | `medbridge/tests/unit/test_state_projector.py`, `medbridge/tests/integration/test_projector_db.py` |
| M-07 Emergency classifier | `medbridge/core/emergency_classifier.py` | `medbridge/tests/unit/test_emergency_classifier.py`, `medbridge/tests/unit/test_emergency_nl_patterns.py` |
| M-08 Pipeline orchestrator | `medbridge/core/orchestrator.py`, `medbridge/core/pipeline_types.py` | `medbridge/tests/unit/test_orchestrator.py`, `medbridge/tests/unit/test_pipeline_types.py`, `medbridge/tests/integration/test_e2e_pipeline.py` |
| M-09 Response templates | `medbridge/core/templates.py` | `medbridge/tests/unit/test_templates.py` |
| M-10 Resilient LLM wrapper | `medbridge/ai/llm_wrapper.py` | `medbridge/tests/unit/test_llm_wrapper.py` |
| M-11 Context extractor | `medbridge/ai/context_extractor.py` | `medbridge/tests/unit/test_context_extractor.py` |
| M-12 Context Gate (Gate 1) | `medbridge/ai/context_gate.py` | `medbridge/tests/unit/test_context_gate.py` |
| M-13 Evidence Gate (Gate 2) | `medbridge/ai/evidence_gate.py`, `medbridge/ai/evidence_calibration.py` | `medbridge/tests/unit/test_evidence_gate.py`, `medbridge/tests/unit/test_evidence_calibration.py` |
| M-14 Response generator | `medbridge/ai/response_generator.py`, `medbridge/core/citations.py` | `medbridge/tests/unit/test_response_generator.py`, `medbridge/tests/unit/test_core_citations.py` |
| M-15 Embedding client | `medbridge/retrieval/embedder.py` | `medbridge/tests/unit/test_embedder.py` |
| M-16 Hybrid retriever | `medbridge/retrieval/hybrid_retriever.py`, `medbridge/retrieval/cache.py` | `medbridge/tests/unit/test_hybrid_retriever.py`, `medbridge/tests/integration/test_hybrid_retriever_qdrant.py`, `medbridge/tests/integration/test_retrieval.py` |
| M-17 Cross-encoder reranker | `medbridge/retrieval/reranker.py` | `medbridge/tests/unit/test_reranker.py` |
| M-18 Ingestion pipeline | `medbridge/ingestion/` | `medbridge/tests/unit/test_ingestion*.py`, `medbridge/tests/integration/test_guideline_collection_indexing.py` |
| M-19 Application entry point | `medbridge/main.py` | `medbridge/tests/integration/test_main_app.py` |
| M-20 Streamlit frontend (legacy) | `medbridge/frontend/` | `medbridge/tests/unit/test_frontend_app.py`, `medbridge/tests/unit/test_frontend_components.py`, `medbridge/tests/unit/test_frontend_services.py` |
| New: privacy layer | `medbridge/privacy/` | `tests/privacy/`, `medbridge/tests/unit/test_privacy_erasure_tokens.py` |
| New: deterministic NLP | `medbridge/nlp/` | `medbridge/tests/unit/test_nlp_*.py` |
| New: Gate 3 grounding verifier | `medbridge/ai/grounding_verifier.py` | `medbridge/tests/unit/test_grounding_verifier.py` |
| New: small talk | `medbridge/core/smalltalk.py` | `medbridge/tests/unit/test_core_smalltalk.py` |
| New: web SPA and dashboard | `medbridge/web/` | `medbridge/tests/unit/test_web_static.py`, `medbridge/tests/unit/test_web_js.py`, `medbridge/web/tests/` |
| New: BP log and clinician summary | `medbridge/features/` | `medbridge/tests/unit/test_features_bp.py`, `medbridge/tests/unit/test_features_summary.py` |
| New: launcher | `medbridge/cli.py` | `medbridge/tests/unit/test_cli.py` |
| New: evaluation package | `evaluation/` | `tests/evaluation/` |
| New: benchmark, RABBITS, adversarial, ops scripts | `tests/benchmark/`, `tests/adversarial/`, `scripts/` | `tests/adversarial/test_security_adversarial.py`, `tests/adversarial/test_rabbits_and_runner.py`, `tests/scripts/` |

## Review 5, final panel (25 marks)

| # | Parameter (marks) | Evidence | Status |
|---|---|---|---|
| 1 | Domain knowledge and problem understanding (4) | Report Ch. 1 "Background and Motivation", "Problem Statement", "Objectives", "Scope of the Project"; Ch. 2 "Clinical Background: Hypertension Guidance and Its Variability"; README "Overview"; corpus scope `data/corpus_manifest.json`, `reports/corpus_stats.md`; viva answers in `docs/DEMO_SCRIPT.md` | **done**. **team action**: each student should be able to explain the five actions and the clinical red flags without notes |
| 2 | Methodology, architecture and technical design (4) | Report Ch. 3 "Methodology and System Design"; `report/figures/architecture_pipeline.png`, `report/figures/fsm_soft_ask.png`, `report/figures/sequence_turn.png`; README "Architecture" (Mermaid); contracts in `docs/finalization_plan.md` §3; invariants in `docs/project_constitution.md` §12; ADL entries in `docs/architecture_decision_log.md`; demo step 2 ("Why this answer?" panel shows both gates) | **done** |
| 3 | Completion, functionality and integration (4) | As Review 4 rows 2–3; live demo steps 1–8; dashboard `/dashboard` (demo step 7); `reports/smoke_test_latest.json` | **done**. **pending results**: live load test |
| 4 | Testing, validation and analysis of results (4) | As Review 4 rows 5–6; hallucination study headline: C5 <!-- RESULT:rate:C5:llm.hallucination -->*pending* (result file not generated yet)<!-- /RESULT --> vs C1 <!-- RESULT:rate:C1:llm.hallucination -->*pending* (result file not generated yet)<!-- /RESULT -->; judge agreement <!-- RESULT:kappa:3class -->*pending* (result file not generated yet)<!-- /RESULT -->; retrieval `evaluation/results/retrieval/retrieval_metrics.md`; PII blind set <!-- RESULT:pii:blind_v2:strict -->precision 0.944, recall 0.744, F1 0.832 (strict span match, 203 gold spans)<!-- /RESULT --> | **pending results** (study, judges). **team action**: label the 50-item human sheet (`python -m evaluation.human_annotation --export`, then `--score`) |
| 5 | Innovation, technical merit and practical relevance (3) | Dual Answerability Gate and optional Gate 3 (README "What is new"); comparison with existing approaches and patents in report Ch. 2 "Comparison of Existing Approaches", "Patent Review", "Identified Research Gap"; local-only privacy design (README "Privacy and security"); Indian context: Hinglish red flags and 65 Indian brand names (`medbridge/core/emergency_classifier.py`, `medbridge/nlp/drugs.py`); ethics and DPDP/HIPAA discussion in the report; feasibility on a 6 GB GPU (model selection: <!-- RESULT:model_selection -->qwen3.5:4b: lenient routing 78.0%, macro-F1 0.71, turn p50 10.3 s; gemma3:4b: lenient routing 50.8%, macro-F1 0.47, turn p50 9.3 s; qwen3:8b: lenient routing 84.7%, macro-F1 0.74, turn p50 16.0 s<!-- /RESULT -->) | **done** (design, privacy, ethics). **pending results**: whether the gates reduce hallucination |
| 6 | Quality of final report and supporting evidence (3) | `report/MedBridge_Final_Report.docx` with every number generated from result files; 90 verified references `report/references/references.json`; consistency between docs and results through `scripts/render_docs_results.py`; individual contribution appendix | **pending results** (report build). **team action**: contribution statement, similarity check |
| 7 | Presentation, demo, viva and teamwork (3) | `report/MedBridge_Review_Presentation.pptx`; `docs/DEMO_SCRIPT.md` (timed flow, fallbacks, 20 viva questions); demo roles split between the two students ([TO BE FILLED: who presents which step]) | **team action**: rehearse twice on the demo laptop, once with the "LLM is slow" fallback |

## Draft / report review (5 marks)

| # | Parameter (marks) | Evidence | Status |
|---|---|---|---|
| 1 | Completeness of report chapters (1) | Preliminary pages `report/content/00_title.md` to `report/content/05_abbreviations.md`; chapters `report/content/ch1_introduction.md`, `report/content/ch2_literature.md`, `report/content/ch3_methodology.md`; Ch. 4 and Ch. 5 written by the report stream once results exist | **pending results** (Ch. 4–5). **team action**: title page and certificate placeholders |
| 2 | Technical consistency between report and implementation (1) | Report values are read from result files at build time (`report/lib/results_context.py`); README and demo numbers come from `scripts/render_docs_results.py`; changes and their rationale in `CHANGELOG.md` and `docs/architecture_decision_log.md`; condition definitions in one place (`medbridge/core/pipeline_types.py`) | **done** (mechanism). Rebuild after the study finishes |
| 3 | Quality of analysis, results and supporting evidence (1) | `evaluation/results/summary.json`, `evaluation/results/error_analysis.md`, `reports/figures/`, `reports/privacy_redaction_metrics.md`, `evaluation/results/retrieval/retrieval_metrics.md`, `reports/corpus_stats.md` | **done**: retrieval, PII, corpus. **pending results**: study |
| 4 | Citations, referencing and similarity compliance (1) | 90 verified references with DOI/URL in `report/references/references.json`, rendered in IEEE style by `report/lib/references.py`; patents in `report/references/patents.json`; original prose (guideline text paraphrased, quotes kept short) | **done** (references). **team action**: run the institutional similarity check ([TO BE FILLED: similarity report and percentage]) |
| 5 | Individual contribution and incorporation of guide feedback (1) | Report appendix "Individual Contribution Statement" and front-matter "Action Taken on Previous Review Comments"; `CHANGELOG.md` | **team action**: [TO BE FILLED: contribution split] and [TO BE FILLED: guide feedback and the action taken] |

## Evidence check

<!-- EVIDENCE-CHECK:START -->
Computed by `scripts/render_docs_results.py`: 156 of 167 evidence paths named in this checklist exist in the repository.

Not present yet (produced by a pending step or another work-stream):

- `reports/performance/load_test.json`
- `evaluation/results/summary.json`
- `evaluation/results/tables/`
- `evaluation/results/error_analysis.md`
- `reports/figures/hallucination_rate_by_condition.png`
- `reports/figures/unsafe_answer_rate_unanswerable.png`
- `reports/figures/ablation_waterfall.png`
- `reports/figures/risk_coverage.png`
- `reports/figures/confusion_matrix_C5.png`
- `reports/performance/latency_table.csv`
- `report/screenshots/`
<!-- EVIDENCE-CHECK:END -->
