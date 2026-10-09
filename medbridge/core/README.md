# Core orchestrator (`medbridge.core`)

The deterministic runtime that turns one patient message into one of five
actions (`SOFT-ASK`, `ANSWER`, `GENERALIZE`, `ABSTAIN`, `ESCALATE`) and
persists the turn. Contract: `docs/finalization_plan.md` §3.1–§3.4.

## Modules

| Module | Responsibility |
|---|---|
| `orchestrator.py` | `PipelineOrchestrator.process_message(session_id, message, db=None, *, on_event, ablation, regenerate)` and `warmup()`. Each stage is a method on a per-turn `TurnContext`; stage timings, gate decisions and evidence are collected in an `Explanation`. Never raises. `get_orchestrator()` returns the shared instance (FastAPI dependency). |
| `pipeline_types.py` | `StageName`, patient-facing `STAGE_LABELS` (streamed over SSE), `PipelineEvent`/`EventCallback`, `AblationConfig` (evaluation-only presets C0–C5, V1, V2; production is `AblationConfig.full()`), `GeneratorMode`. |
| `emergency_classifier.py` | Deterministic, LLM-free red-flag classifier on the raw text: colloquial phrasing, typos, BP crisis readings, negation/informational/historical suppression. `classify()` and `classify_detailed()` (rule ids, negated, suppressed, decisive BP). Owned by the emergency work-stream. |
| `citations.py` | `validate_citations(text, chunks)`: strips `[n]` markers that do not map to a chunk, renumbers the rest by first appearance and rebuilds every citation from chunk metadata (`source = f"{org} {year}"`, falling back to the title, then the guideline id). HTML sources (`source_format == "html"`) get `page=None` and a section such as `"Treatment (section 4)"`. |
| `smalltalk.py` | `is_small_talk(text)`: conservative whole-message matcher for greetings/thanks/goodbyes/"who are you"; returns a template key or `None`. |
| `templates.py` | Pre-vetted texts: ESCALATE (configurable emergency numbers + 911), ABSTAIN, loop-breaker prefix, GENERALIZE fallback, pipeline-error fallback, SOFT-ASK fallback question, small-talk replies. |

## Stages of `process_message`

| # | Stage | What happens | Failure behaviour |
|---|---|---|---|
| 0 | privacy | `privacy.redaction.redact()`; the redacted text is used by every later stage. | Fails closed: safe error response, nothing sent to an LLM or stored. |
| 1 | emergency | `classify_detailed(raw message)` before any LLM → ESCALATE template. `explanation.emergency` records `action`, `matched_rules`, `negated`, `suppressed`, `bp` — never the evidence spans (patient-text fragments). | Classifier crash → continue (Gate 2 can still ESCALATE). |
| 2 | load_state | Snapshot, soft-ask counter, last `EXTRACTOR_HISTORY_TURNS` turns. | Empty state. |
| 2b | small_talk | Deterministic GENERALIZE template, no clinical claims. | — |
| 3 | extraction | `ContextExtractor.extract()` → delta events, search query, intent; pure in-memory `StateProjector.apply_events`. | Spec fallback: no events, raw intent = message, retrieval query = `nlp.expand_query(redacted message)`. |
| 4 | context_gate | Gate 1 (T = 0, seed 42). SOFT-ASK below the limit → generator asks for the missing facts, counter + 1. SOFT-ASK at `SOFT_ASK_MAX_COUNT` (2) → loop-breaker: forced GENERALIZE, Gate 2 skipped, counter reset. PROCEED resets the counter. With `SPECULATIVE_RETRIEVAL`, retrieval starts alongside Gate 1 and is cancelled and recorded as `discarded` on SOFT-ASK. | PROCEED. |
| 5 | retrieval + rerank | `HybridRetriever.hybrid_search` (top-20, RRF k = 60) → `CrossEncoderReranker.rerank_chunks` (top-5, thread pool). | Empty evidence / RRF order kept. |
| 6 | evidence_gate | Gate 2 (T = 0, seed 42) with `message=` (patient words, preferred) and `query=` (search query); V1 `calibrated_evidence_decision` when enabled. `explanation.gate2.features` always holds the production `evidence_features(...)` (max/mean-top-3 rerank, entity coverage, chunk count) so V1 thresholds can be calibrated from C5 traces. | GENERALIZE. |
| 7 | generation | ABSTAIN/ESCALATE → templates (generator never called). ANSWER/GENERALIZE/SOFT-ASK → `ResponseGenerator.generate(message=, query=)`. Citation validation; an ANSWER with zero valid citations is downgraded to GENERALIZE. | Pre-vetted GENERALIZE template (SOFT-ASK: fallback question). |
| 7b | verification | Optional Gate 3 (`GROUNDING_VERIFIER_ENABLED` or V2): unsupported sentences removed when the unsupported fraction ≤ τ; otherwise ANSWER is regenerated as GENERALIZE once, then ABSTAIN. `explanation.gate3` holds only numbers and status (`status`, `available`, `n_claims`, `n_unsupported`, `unsupported_fraction`, `threshold`, `attempts`, `action_change`, `removed_sentences_count`). | Draft kept, `status: failed`. |
| 8 | persistence | Exactly one `turn_store.persist_turn(TurnRecord)` (single flush). | The patient still gets the answer. |

`regenerate=True` re-runs the last stored user message: extraction still runs
for the search query, but no new clinical events and no new user message are
stored.

`warmup()` loads the embedders, Qdrant connection and reranker concurrently
with the LLM models and, when `GROUNDING_VERIFIER_ENABLED`, awaits
`GroundingVerifier.warmup()`; its boolean result becomes
`readiness["nli_verifier"]` (shown as `models_loaded.nli_verifier` in `/health`).

## Tests

`medbridge/tests/unit/test_orchestrator.py` (every route, fallback, ablation,
speculative retrieval, V1/V2, regenerate, trace privacy, warm-up),
`test_core_citations.py`, `test_core_smalltalk.py`, `test_templates.py`.
