# Implementation and Results

This chapter covers the implementation, interface, testing, component evaluations, code review, hallucination study and limitations. Every number is read from a results file; missing values are marked as pending.

## Implementation Details

**Application and API.** At start-up (`medbridge/main.py`) a background task loads the encoders and reranker, probes Qdrant and asks Ollama to keep the models resident, so the first message does not pay the model-loading cost that accounted for much of the first-turn latency of the received prototype (Section 4.8); a second task runs the retention job. One `Settings` class (`medbridge/config.py`) defaults to fully local operation: local-only mode, gate temperature 0.0, generator temperature 0.3, seed 42, loop-breaker at 2, and speculative retrieval, redaction, encryption and authentication on, with V1 and V2 off. Each response carries the text, action, citations, counter, redaction count and `Explanation`. The streaming endpoint runs the same orchestrator call in a background task and emits `stage` events, then `meta`, `token` events that replay the validated text and `done` (or `error` and a safe `done`); a disconnect does not stop the turn, and an erased session gets HTTP 410 before any pipeline work.

**Orchestrator and language-model layer.** The original `_execute_pipeline`, a method of about 390 lines in a 539-line module, became one short method per stage with injected collaborators, so unit tests replace any component with a fake and check every route and fallback without a model. All model calls go through `ResilientLLMWrapper`, which sends the JSON schema of the expected Pydantic model as Ollama's `format` parameter; decoding is therefore grammar-constrained and small local models return valid JSON. Calls set temperature, seed, context size and a token limit, disable "thinking" on hybrid models and share one HTTP client; failures are retried with backoff, a truncated reply is retried once with a doubled budget and never repaired or cached, and when every attempt fails the orchestrator applies the fallback. Prompts have a system part (role, ordered rules, examples on topics absent from the benchmark) and a user part containing only data blocks. The extractor sees the last four turns to resolve references such as "that dose". The generator has the production `grounded` mode and the evaluation-only `direct`, `vanilla` and `prompt_safety` modes of C0 to C2.

**Retrieval, ingestion and persistence.** The retriever sends one Qdrant query with dense and sparse prefetch branches of 20 candidates and RRF with *k* = 60, caches non-empty results and raises `RetrievalError` on infrastructure failure, so an outage is never mistaken for "nothing relevant found"; the reranker selects CUDA, MPS or CPU. Ingestion verifies each file against its manifest checksum and stores each chunk with document, organisation, year, section, page range, URL, topics and checksum. The turn store encrypts every text field and flushes once; four Alembic migrations define the schema.

**Features and evaluation package.** A BP log classifies readings into the ACC/AHA categories with a deterministic function, and manual entries become BP_READING events through the same projection, so the chart and the gates see the same data. A clinician summary exports the snapshot, BP log and each answer with its action and citations under a disclaimer, and an administrator dashboard shows actions, gate decisions, fallbacks, latency, feedback and redacted audit entries. The evaluation package implements Section 3.6 with resumable runs and a circuit breaker after consecutive degraded items, and the AQ, RABBITS, adversarial, smoke, load and latency harnesses run live; their mock modes remain only as labelled self-tests.

## User Interface

The Streamlit interface was replaced by a single-page application served by FastAPI (`medbridge/web/`), written in plain HTML, CSS and JavaScript with no build step or third-party scripts, so it runs offline under the strict Content Security Policy; Markdown is rendered by an escape-first renderer. It offers a searchable conversation sidebar, a 2,000-character composer, light and dark themes, a mobile layout and keyboard and screen-reader support. A live stepper shows the streamed stages, each answer carries a colour-coded action badge, and citation markers open a source drawer with guideline, organisation, year, section, page and excerpt (Figure 4.1). The "Why this answer?" panel (Figure 4.2) shows the decision path, each gate's decision, source and rationale, the reranked passages and per-stage latency without recomputing anything, and an ESCALATE response is an emergency card with the configured numbers (112 and 108 for India; Figure 4.3).

[[figure:report/screenshots/answer_citations.png|ANSWER with citation chips and the source drawer|width=4.3]]

[[figure:report/screenshots/why_panel.png|"Why this answer?" panel with gate decisions and evidence|width=4.3]]

[[figure:report/screenshots/escalate.png|ESCALATE response rendered as an emergency card|width=4.3]]

A patient-context panel shows the snapshot, a BP chart, a manual entry form and a "clear my data" button; messages can be regenerated, rated and exported, and an unconfirmed erasure is never reported as successful. Screenshots are captured with `python -m medbridge.web.screenshots` against the real backend (more in Appendix B).

## Testing and Validation

Unit tests replace the model, database and vector store with fakes; integration tests run against a PostgreSQL test database and the Qdrant collection and are skipped when a service is unreachable; tests needing a running model are excluded by default (Tables 4.1 and 4.2). Static analysis uses `ruff` and `mypy`.

[[table:test_summary]]

Table: Test suites and what they establish
| Suite | Location | What it establishes |
|---|---|---|
| Unit | `medbridge/tests/unit/` | Five routes, loop-breaker, template bypass, emergency-before-LLM order, every fallback, speculative retrieval, citation downgrade, ablation presets, Gate 3, one persistence call per turn; structured outputs and truncation; prompt isolation and decontamination |
| Integration | `medbridge/tests/integration/` | Migrations, append-only rules, encrypted turn store, concurrent turns, live retriever, every API route, end-to-end pipeline with PostgreSQL and Qdrant |
| Privacy | `tests/privacy/` | Redaction thresholds, blind-set reproduction, no clinical-value corruption, encryption, crypto-shredding, tokens, headers, limits, log scrubbing, retention |
| Emergency | `medbridge/tests/unit/test_emergency_*.py` | Every English and Hinglish case, determinism, the 10 ms budget |
| Adversarial | `tests/adversarial/` | Isolation under injected, unterminated and invisible-character tags; fast-path precedence; SQL injection; authentication |
| Evaluation, scripts, web | `tests/evaluation/`, `tests/scripts/`, `medbridge/web/tests/` | Statistics against reference implementations, degraded-turn and calibration rules, judges; operational scripts; Markdown rendering with cross-site-scripting payloads |

`scripts/smoke_test.py` exercises a running server with one request per behaviour, from health and session creation to erasure and access after erasure (Appendix C). The latest recorded run, from {{smoke.generated_at|date}}, passed {{smoke.counts.PASS|int}} checks with {{smoke.counts.FAIL|int}} failures (`reports/smoke_test_latest.json`); it predates the final review-fix commit and must be repeated on the final build.

## Retrieval Evaluation

Table 4.3 reports each metric on the 70 labelled queries (Section 3.6) with a 95 per cent bootstrap interval.

[[table:retrieval_metrics|Retrieval quality per configuration (mean, 95% bootstrap CI)]]

Hybrid fusion ranks best: MRR {{retrieval.modes.hybrid.mrr|cif}} and nDCG@5 {{retrieval.modes.hybrid.ndcg@5|cif}}, against {{retrieval.modes.dense.mrr|cif}} and {{retrieval.modes.dense.ndcg@5|cif}} for dense and {{retrieval.modes.sparse.mrr|cif}} and {{retrieval.modes.sparse.ndcg@5|cif}} for sparse retrieval. Dense retrieval already finds most relevant passages (Recall@5 {{retrieval.modes.dense.recall@5|cif}} against {{retrieval.modes.hybrid.recall@5|cif}} for hybrid), so fusion mainly moves them higher, which matters because only five reach the gates. The dense and hybrid intervals overlap, so the gain over dense retrieval is suggestive rather than established. Hybrid Recall@20, the ceiling for reranking, is {{retrieval.modes.hybrid.recall@20|cif}}. Reranking with the default cross-encoder did not help (MRR {{retrieval.modes.hybrid+rerank.mrr|cif}}, nDCG@5 {{retrieval.modes.hybrid+rerank.ndcg@5|cif}}), and no alternative cross-encoder was significantly better (Table 4.4).

[[table:reranker_ablation|Reranker ablation (hybrid top-20 to top-5)]]

Possible reasons are that the query set is small and that page-level regular-expression labels do not reward promoting a relevant passage from an unlabelled page; the medical `ncbi/MedCPT-Cross-Encoder` (MRR {{reranker.ncbi_MedCPT-Cross-Encoder.modes.hybrid+rerank.mrr|cif}}) was not significantly better than the web-trained MS MARCO models and did not improve on fusion alone, so training on web queries does not explain the result. The default reranker was kept because reranking is part of the specified architecture, it is the fastest candidate, and its score is the evidence signal of V1 and the ranking variable of the risk-coverage analysis, which rank-based fusion scores cannot provide. Using a medical cross-encoder such as MedCPT as the V1 evidence signal is left for future work.

## Model Selection

The application model performs extraction, both gate decisions and generation, and should fit a 6 GB GPU at 4-bit quantisation. `gemma3:4b` [@gemmateam2025gemma3] (the Review-3 model), `qwen3:8b` [@yang2025qwen3] and the later `qwen3.5:4b` of the same family were compared by running C5 on the development split only, recording routing accuracy with Wilson intervals, per-class recall, macro-F1, first-attempt JSON validity, latency and throughput (routing measures the gates, not hallucination). `qwen3:8b` (5.2 GB of weights) is the larger model and would spill to the CPU on the 6 GB GPU once its KV cache is added. A first round on {{model_selection_r1.generated_at|date}} used the prompts from before the code review (Table 4.5).

[[table:model_selection_round1|Model selection, round 1 (C5, development split)]]

In round 1 grammar-constrained outputs made every role of every model valid on the first attempt (for example {{model_selection_r1.results.gemma3:4b.roles.context_gate.valid_first_attempt|ci}} for the Context Gate of `gemma3:4b`). Strict routing accuracy was modest: {{model_selection_r1.results.gemma3:4b.routing.accuracy_strict|ci}} for `gemma3:4b`, {{model_selection_r1.results.qwen3.5:4b.routing.accuracy_strict|ci}} for `qwen3.5:4b` and {{model_selection_r1.results.qwen3:8b.routing.accuracy_strict|ci}} for `qwen3:8b` on 59 turns. The Context Gate over-asked (`gemma3:4b` chose SOFT-ASK {{model_selection_r1.results.gemma3:4b.routing.per_class.SOFT-ASK.predicted|int}} times for {{model_selection_r1.results.gemma3:4b.routing.per_class.SOFT-ASK.support|int}} gold SOFT-ASK turns), `qwen3:8b` over-used GENERALIZE, and `qwen3:8b` was about twice as slow (median {{model_selection_r1.results.qwen3:8b.latency_turn_p50_ms|s}} s against {{model_selection_r1.results.gemma3:4b.latency_turn_p50_ms|s}} s) without being significantly more accurate (its strict accuracy was below that of `qwen3.5:4b`, and all intervals overlap). The routing rules were therefore rewritten (Section 3.3): Gate 1 asks only when the question concerns the patient's own cardiovascular care, a specific personal fact would change the safe answer, and that fact is missing, and Gate 2's "absent from the evidence" rule was restricted to named products. These rule changes were based on the development split only, and the few-shot examples were decontaminated against every HalluBench turn at the same time. The final round repeats the protocol with the revised prompts (Table 4.6, Figure 4.4).

[[table:model_selection|Model selection, final round (C5, development split)]]

[[figure:reports/figures/model_selection.png|Model-selection comparison (C5, development split)|width=5.4]]

The rule gives priority to lenient accuracy and macro-F1, with attention to SOFT-ASK and ABSTAIN recall, subject to the 6 GB limit and acceptable latency, preferring the faster model where intervals overlap. [TO BE FILLED: final application model chosen from the final-round table, with a one-sentence justification based on its numbers]. The LLM judge must come from a different family than the chosen model.

## PII Redaction Evaluation

The redactor was evaluated at span level with `scripts/pii_eval.py` (`reports/privacy_redaction_metrics.json`). The development set (240 messages, 183 spans, 90 negatives) was written with the detector, so its numbers are in-sample. The blind set (237 messages, 203 spans, 89 negatives) was written afterwards by an author without access to the detector or its metrics, covering Indian, US and UK contexts in formal, casual, Hinglish and text-message styles; the detector was frozen, its SHA-256 hash recorded and the set evaluated once. Strict matching requires exact offsets and category, overlap matching an overlapping span of the same category (Table 4.7).

[[table:pii_metrics|Redaction quality on the development and blind sets]]

On the development set strict precision, recall and F1 were {{pii.datasets.development.results.all.overall_strict.precision|f3}}, {{pii.datasets.development.results.all.overall_strict.recall|f3}} and {{pii.datasets.development.results.all.overall_strict.f1|f3}}; on the blind set, the honest estimate, they were {{pii.datasets.blind_v2.results.all.overall_strict.precision|f3}}, {{pii.datasets.blind_v2.results.all.overall_strict.recall|f3}} and {{pii.datasets.blind_v2.results.all.overall_strict.f1|f3}} (overlap F1 {{pii.datasets.blind_v2.results.all.overall_overlap.f1|f3}}). Table 4.8 explains the gap.

[[table:pii_per_category|Blind-set redaction recall per identifier category]]

Structured identifiers transfer well, but only {{pii.datasets.blind_v2.results.all.per_category_strict.NAME.tp|int}} of {{pii.datasets.blind_v2.results.all.per_category_strict.NAME.support|int}} blind-set names were found (recall {{pii.datasets.blind_v2.results.all.per_category_strict.NAME.recall|f3}}), although almost every predicted name was correct: the blind author used cues the detector did not know ("X here", "From: X") and names without a cue. A cue-free detector would risk redacting brand names such as Telma or Concor. Address boundaries were the second weakness. No protected clinical value was corrupted ({{pii.datasets.blind_v2.results.all.protected_values_corrupted|len}} on the blind set), and {{pii.datasets.blind_v2.results.all.negatives_altered|len}} of 89 blind negatives was altered ("Dr. Google" treated as a person). Fixing the address-boundary bug and this false positive after the blind run gives a strict F1 of {{pii.datasets.blind_v2.post_hoc.results.all.overall_strict.f1|f3}}, an optimistic post-hoc figure rather than an independent estimate. Detection took {{pii.datasets.blind_v2.results.all.mean_latency_ms|f3}} ms per message. Reliable name removal would need a named-entity model or gazetteer.

## Emergency Classifier Evaluation

The classifier was developed against 288 English messages (156 must escalate, 132 must not) and 141 Hinglish and Indian-English messages (90 and 51), with deliberately hard negatives: negations, educational and historical mentions, exertional breathlessness, targets, thresholds, dates, doses, laboratory values and near-miss words. The unit tests `test_precision_recall_summary` and `test_hinglish_precision_recall_summary` require zero errors on the English and Hinglish files, and the classifier passes (Table 4.9; cases in Appendix C).

Table: Emergency classifier on the labelled case files
| Case file | Messages | Must escalate | Must not escalate | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|
| `emergency_cases.json` (English) | 288 | 156 | 132 | 156 | 0 | 0 | 132 | 1.000 | 1.000 |
| `emergency_cases_hinglish.json` | 141 | 90 | 51 | 90 | 0 | 0 | 51 | 1.000 | 1.000 |

These files were written alongside the rules, so they show coverage of the intended phrasings, not real-world accuracy; the HalluBench emergency items give the independent estimate. Side-effect questions that name a red flag are escalated by design and, since Gate 2 cannot overrule the fast-path, are the main known false positives. BP values in words, readings in cmHg and "SOB" alone are not supported, and a number pair without BP wording must exceed both thresholds.

## Performance Optimisation

Review 3 reported 25.2 s for an ABSTAIN and 35.8 s for a GENERALIZE turn. A baseline re-measured on the received code with `gemma3:4b` (Table 4.10) showed the first out-of-domain ABSTAIN turn at {{perf.before_gemma3-4b.cold.1.latency_s|f1}} s cold and {{perf.before_gemma3-4b.warm_summary.S2-t1:ABSTAIN.p50|f1}} s warm (model loading dominated), a single-turn ANSWER at {{perf.before_gemma3-4b.cold.5.latency_s|f1}} s cold and {{perf.before_gemma3-4b.warm_summary.S5-t1:ANSWER.p50|f1}} s warm, and the fast-path at {{perf.before_gemma3-4b.warm_summary.S1-t1:ESCALATE.p50|f3}} s.

[[table:perf_baseline|Baseline latency of the received prototype]]

Optimisations that preserve the routing semantics were then added: warm-up with keep-alive, structured outputs without repair loops, "thinking" disabled and bounded outputs, a persistent HTTP client, concurrent encoding with caches, speculative retrieval, GPU reranking, one database flush per turn, templates for small talk and chunks that fit the encoder windows (Table 4.11).

[[table:perf_before_after|Latency before and after optimisation (warm runs)]]

[TO BE FILLED: two or three sentences interpreting the before/after table once the "after" latency probe has been run on the final configuration, including the share of turn time spent in language-model calls, which the per-stage traces show to be the dominant cost.]

## Code Review and Quality Assurance

Before the final experiments an adversarial review of seven areas (invariants, ablation validity, statistics and judges, privacy, persistence, prompts, web interface) checked every proposed defect three times independently. It confirmed 37 findings (5 high, 17 medium, 15 low severity), rejected 3, and all 37 were fixed in commit `10aef5e` with regression tests (`docs/notes/code_review_findings.md`). The high-severity findings would have biased the study or the patient state: degraded turns counted as valid outcomes; a vector-store outage logged as an empty retrieval, favouring gated conditions; a filter meant for non-clinical claims that also removed clinical ones; concurrent writers dropping snapshot events; and target questions recorded as BP readings. Medium fixes included a vanilla baseline that had used the production safety prompt, gate examples containing a benchmark item's fabricated drug, intent outside the isolation block, truncated output, the counter on regeneration and erasure after token expiry. All prompts changed, so cached outputs were invalidated and the study and final model-selection round were run afterwards.

[[include:ch4_study_results.md]]

## Limitations

- **Knowledge base.** Several guidelines could not be downloaded by script because publishers serve a browser check that was not bypassed: the 2023 ESH, the 2022 AHA/ACC/HFSA and 2021 ESC heart-failure, the full 2023 chronic coronary disease (only a review is included), the 2024 ESC chronic coronary syndromes and atrial fibrillation, the 2019 ESC/EAS dyslipidaemia guidelines and ISSHP pregnancy guidance [@mancia2023esh; @heidenreich2022hf; @virani2023ccd]. Heart failure and coronary disease are thinly covered. The ESC 2024 guideline reserves text-and-data-mining rights and the ISH 2020 guideline came from an institutional repository; both are flagged for the team. Headings are detected heuristically, and HTML sources are cited by section.
- **Benchmark and judges.** HalluBench was drafted by one author with an AI assistant, without clinician review or inter-annotator agreement, with mostly casual English and one Hinglish item. The LLM judge is a model whose labels can depend on the other claims in a batch; the NLI thresholds are calibrated against it, so the judges are not fully independent; and the human sheet must still be completed. Memoisation makes raw latencies of later conditions meaningless, so study latencies are reported cache-imputed.
- **In-sample component metrics.** The emergency files and the PII development set were written with the rules; the blind PII set and HalluBench emergency items give the independent estimates.
- **Privacy.** Names without a known cue are missed (blind recall {{pii.datasets.blind_v2.results.all.per_category_strict.NAME.recall|f3}}) and reach the local model, though encrypted at rest. Master-key rotation and token revocation are not implemented, pre-encryption rows remain in plaintext but hidden, and rate limiting is per process.
- **System.** One Uvicorn worker is assumed (across workers the soft-ask counter would become last-write-wins). Streaming is simulated: stages are live, but text is replayed after the turn. Turns take several to tens of seconds; the 4-billion-parameter application model routed only {{model_selection.results.qwen3.5:4b.routing.accuracy_strict|ci}} of development turns exactly right (Table 4.6) and can produce advice not in the evidence, since citation validation checks only that markers point to real passages. The Windows target laptop, CUDA paths, other browsers and screen readers were not tested by the authors, and the brand lexicon was not checked against a drug database.
- **Scope.** MedBridge is a research prototype, not clinically validated and not a medical device, evaluated only on synthetic data; it must not be used for diagnosis or treatment.
