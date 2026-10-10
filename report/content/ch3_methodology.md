# Methodology and System Design

This chapter turns the research problem of Chapter 1 into a concrete method: an explicit decision about *whether* to answer, taken twice at two points of a retrieval-augmented pipeline, behind a deterministic emergency check. It then describes the architecture and its eight-step workflow, the modules, the algorithm of every decision point, the data model, the privacy and security design, the experimental design of the hallucination study, and the tools, models and datasets used. Every element corresponds to code in the repository; paths are given relative to the repository root.

## Proposed Methodology

### Answerability as an Explicit Decision

A conventional retrieval-augmented generation (RAG) system retrieves passages for every query and generates an answer from them [@lewis2020rag; @gao2023ragsurvey]; whether the answer is safe is left to the generator. MedBridge instead treats *answerability* as a decision taken before any clinical text is written, and asks two separate questions:

1. **Is the patient context sufficient?** Some questions can only be answered safely if a specific fact about the patient is known, for example the current medicine before a question about increasing "my BP tablet", or pregnancy status before a question about a drug that is contraindicated in pregnancy. If the fact is missing, the right response is a focused clarifying question, not a guess.
2. **Is the retrieved evidence sufficient?** Even with complete context, the knowledge base may not support a specific answer: the question may concern a drug that does not exist, rest on a false premise, ask for a number the guidelines do not give, or fall outside cardiovascular care. The right response is then general, clearly labelled information, a refusal or an emergency referral.

The first question can be answered *before* retrieval, from the conversation alone; the second only *after* retrieval, by examining the passages. MedBridge therefore places a **Context Gate** (Gate 1) before retrieval and an **Evidence Gate** (Gate 2) after it. Together they form the **Dual Answerability Gate**, the novel contribution of the project. A deterministic **emergency fast-path** runs before both gates and before any language-model call, so that a red-flag message is escalated immediately and cannot be delayed or overridden by a probabilistic component.

### Design Principles

- **Safety before fluency.** When a component fails or is uncertain, the system moves to the more conservative action: Gate 1 falls back to PROCEED, Gate 2 to GENERALIZE, the generator to a pre-vetted template, and checks added after generation can only lower an action.
- **Determinism where it matters.** The emergency classifier is pure rules; the extractor and both gates run at temperature 0.0 with seed 42 under grammar-constrained JSON decoding; ABSTAIN and ESCALATE are fixed templates that never call the generator.
- **Separation of decisions.** Each gate has its own prompt, output schema, rationale and audit-log column; the two questions are never merged into one call.
- **Traceable evidence.** Every clinical statement in an ANSWER cites a retrieved passage; markers are validated against the passages actually shown to the generator, and citation text is rebuilt from passage metadata, never taken from the model.
- **Local-first privacy.** All inference runs on the user's machine through Ollama [@ollama2026]; identifiers are redacted before any model call or database write, and stored text is encrypted per session.
- **Measured, not asserted.** Every number in this report comes from files written by a reproducible evaluation package.

### The Five Response Actions

Every patient turn ends in exactly one of five actions, summarised in Table 3.1.

Table: The five response actions of MedBridge
| Action | Chosen when | Decided by | Response text |
|---|---|---|---|
| ESCALATE | A red-flag situation is described (BP of 180/120 mmHg or above with BP wording, chest pain, stroke signs, fainting, ...) | Emergency fast-path; Gate 2 can also escalate | Pre-vetted emergency template with configurable numbers (no LLM) |
| SOFT-ASK | A missing personal fact would change the safe answer | Gate 1, at most twice in a row | Short question for the specific missing facts, no clinical claims |
| ANSWER | Context and evidence are both sufficient | Gate 2 | Grounded, patient-friendly answer with inline `[n]` citations |
| GENERALIZE | Evidence supports only general information, a false premise must be corrected, or the loop-breaker fires | Gate 2, loop-breaker, citation downgrade, small-talk rule | General guideline information, labelled as not personalised |
| ABSTAIN | Evidence irrelevant or absent, topic outside cardiovascular care, unknown product or unsupported dose | Gate 2 (or Gate 3 in variant V2) | Pre-vetted refusal that recommends a clinician (no LLM) |

### Development Method

Up to Review 3 the project produced the original pipeline with a Streamlit interface, a three-document corpus and a benchmark run only in mock mode. The final stage followed a written plan (`docs/finalization_plan.md`): baseline measurement, parallel implementation in seven work-streams with frozen interface contracts, integration, an adversarial code review, corpus expansion and retrieval evaluation, model selection, the hallucination study, live benchmarks and documentation. Each significant change was recorded in the Architecture Decision Log (`docs/architecture_decision_log.md`) with an argument for why it preserves the architecture, and the seven non-negotiable invariants of the project constitution (`docs/project_constitution.md`) were treated as hard constraints: emergency fast-path before any LLM call; append-only clinical events and audit logs; both gates at temperature 0.0 with seed 42; loop-breaker limit of two; patient text isolated in `<untrusted_user_input>` blocks; template bypass for ABSTAIN and ESCALATE; and no internal error reaching the patient.

## System Architecture

### Layered Architecture

MedBridge is a single-node application organised in layers (Figure 3.1).

[[figure:report/figures/architecture_pipeline.png|MedBridge architecture: the Two-Stage Answerability and Retrieval Reasoning Pipeline with the privacy layer, the emergency fast-path, both gates, the optional grounding verifier and streaming output|width=5.4]]

- **Presentation.** A single-page web application in HTML, CSS and plain JavaScript modules served by FastAPI, with an administrator dashboard (`medbridge/web/`); the Streamlit client (`medbridge/frontend/`) is kept as a fallback. Neither contains clinical or AI logic.
- **API.** FastAPI routes for sessions, messages (JSON and Server-Sent Events), regeneration, feedback, patient context, BP entry, clinician summary, configuration, health, metrics and administration (`medbridge/api/`), behind security middleware (`medbridge/privacy/middleware.py`).
- **Pipeline core.** The orchestrator (`medbridge/core/orchestrator.py`) with the emergency classifier, small-talk detector, citation validator and templates.
- **AI.** One resilient gateway to the language model (`medbridge/ai/llm_wrapper.py`) and the four LLM-backed components (context extractor, Context Gate, Evidence Gate, response generator), the optional variants V1 and V2, and deterministic language helpers (`medbridge/nlp/`).
- **Retrieval.** Dense and sparse encoders, the Qdrant hybrid retriever and the cross-encoder reranker (`medbridge/retrieval/`) [@qdrant2026].
- **State and persistence.** State projector, session manager and single-flush turn store (`medbridge/state/`) over PostgreSQL (`medbridge/db/`, Alembic migrations).
- **Privacy.** Redaction, envelope encryption, crypto-shredding, tokens, log scrubbing and retention (`medbridge/privacy/`), used by the other layers.

Two offline components complete the system: the ingestion pipeline (`medbridge/ingestion/`), the only writer to the vector store, and the evaluation package (`evaluation/`), which drives the production orchestrator under internal ablation settings.

### The Eight-Step Decision Workflow

Table 3.2 maps the eight steps of the workflow to the stages implemented by the orchestrator. Two stages were added in the final phase without changing the order or the routing semantics: PII redaction at the start and a deterministic small-talk check after state loading.

Table: The eight-step workflow and its implementation stages
| Step | Stage(s) in code | Component | LLM? | Outcome |
|---|---|---|---|---|
| 1. User clinical query | `privacy` | Redactor | No | Redacted text for every later stage except step 2 |
| 2. Emergency fast-path | `emergency` | Rule-based classifier on the raw text | No | ESCALATE template, or continue |
| 3. Context extraction and state | `load_state`, `small_talk`, `extraction` | Turn store, small-talk detector, extractor, projector | Extractor | Snapshot, recent turns, delta events, search query |
| 4. Context Gate (Gate 1) | `context_gate` | Context Gate, loop-breaker | Yes | SOFT-ASK, forced GENERALIZE, or PROCEED |
| 5. Hybrid retrieval and reranking | `retrieval`, `rerank` | Hybrid retriever, reranker | No | Top-5 passages (may start during step 4) |
| 6. Evidence Gate (Gate 2) | `evidence_gate` | Evidence Gate (or V1) | Yes | ANSWER, GENERALIZE, ABSTAIN or ESCALATE |
| 7. Grounded response generation | `generation`, `verification` | Generator, citation validator, templates, optional Gate 3 | Generator | Cited text, clarifying question, or template |
| 8. Safe clinical response | `persistence` | Turn store, SSE stream | No | Persisted turn with audit row and trace |

A typical turn makes four language-model calls; a SOFT-ASK or ABSTAIN turn makes three, and an emergency or small-talk turn none. Figure 3.2 shows the sequence of a turn that reaches ANSWER.

[[figure:report/figures/sequence_turn.png|Sequence of one patient turn through the API, orchestrator, AI layer, retrieval layer and turn store, including speculative retrieval during Gate 1|width=6.3]]

### Turn Context, Tracing and Streaming

All state of a turn is held in a `TurnContext` object passed from stage to stage; the orchestrator keeps no per-turn state, so one instance serves concurrent requests. Each stage runs inside a context manager that records its duration and status (`ok`, `skipped`, `failed` or `discarded`) and emits a progress event with a patient-facing label such as "Checking for emergencies…" or "Verifying evidence…". The streaming endpoint forwards these events as Server-Sent Events. Because the generator returns validated JSON, the text is sent only after citation validation (and Gate 3 when enabled) and is then replayed word by word, so no unvalidated text reaches the patient.

An `Explanation` record is returned with every response and stored in the audit log: both gate decisions with their source (`llm`, `fallback`, `loop_breaker`, `fastpath`, `calibrated`, `rule` or `ablation`), rationale and missing fields; matched emergency rules; the top-5 passages with fusion and rerank scores; citation-validation and Gate 3 outcomes; per-stage timings; per-call LLM statistics; the speculative-retrieval outcome; and the redaction count. It drives the "Why this answer?" panel and the metrics endpoint and is the raw material of the evaluation.

### Failure Handling and the Never-Throw Guarantee

`process_message` never raises. Each stage handles its own failures (Table 3.3), and a top-level guard converts anything unexpected into a safe generic response.

Table: Failure handling per stage
| Failure | Fallback behaviour | Recorded as |
|---|---|---|
| Redaction raises | Fail closed: safe error response, nothing sent or stored | Stage `privacy` failed |
| Emergency classifier raises | Continue; Gate 2 can still escalate | Stage `emergency` failed |
| Session state cannot be loaded | Continue with an empty state | Stage `load_state` failed |
| Extractor fails | Query = redacted message plus clinical synonyms; no events | Stage `extraction` failed |
| Gate 1 fails | PROCEED | Gate 1 source `fallback` |
| Retrieval infrastructure fails | Empty evidence; Gate 2 still decides | `retrieval_status = failed` |
| Reranker fails | Keep the fusion order | Stage `rerank` failed |
| Gate 2 fails (or every V1 vote fails) | GENERALIZE | Gate 2 source `fallback` |
| Generator fails | Pre-vetted GENERALIZE template | Stage `generation` failed |
| Gate 3 fails | Keep the validated draft | Gate 3 status `failed` |
| Persistence fails | Answer still returned; transaction rolled back | Stage `persistence` failed |
| Any other exception | Safe error template; nothing persisted | Gate 2 source `fallback` |

The evaluation package uses the same taxonomy to flag *degraded* turns, so that a turn that only looks safe because a component failed is never counted as a success (Section 3.7).

### Deployment Architecture

MedBridge runs on one machine (Figure 3.3). PostgreSQL 16 and Qdrant run in Docker containers bound to the loopback interface; Ollama runs natively to use the local GPU (Metal on Apple Silicon, CUDA on Windows); the FastAPI application runs under Uvicorn as one process serving both the API and the web interface, with no Node.js build step. At start-up a background task warms up the encoders, reranker and language models, and `/health` reports `degraded` until it finishes; a second background task enforces data retention.

[[figure:report/figures/deployment.png|Single-node deployment: browser, FastAPI/Uvicorn application, Ollama on the local GPU, and PostgreSQL and Qdrant containers bound to the loopback interface|width=6.0]]

## Module Description

The approved scope consists of twenty modules, M-01 to M-20, defined in the technical specification. All were kept and most were extended (Table 3.4); thirteen modules were added (Table 3.5).

Table: Approved modules M-01 to M-20 and their final-phase extensions
| ID | Module | Location | Final-phase extension |
|---|---|---|---|
| M-01 | Configuration | `medbridge/config.py` | Local-only default, per-role models, privacy and variant flags |
| M-02 | Database layer | `medbridge/db/`, `alembic/` | Session keys, feedback, audit trace, erasure columns |
| M-03 | Schemas and middleware | `medbridge/api/schemas/`, `api/middleware/` | `Explanation` model, stage timings, citation metadata |
| M-04 | API routes | `medbridge/api/routes/` | Streaming, regenerate, feedback, context, BP, summary, metrics, admin |
| M-05 | Session manager | `medbridge/state/session_manager.py` | Erased-session checks, cipher access |
| M-06 | State projector | `medbridge/state/projector.py` | Dose-change history, condition union |
| M-07 | Emergency classifier | `medbridge/core/emergency_classifier.py` | Colloquial phrasing, negation, typos, context filters, Hinglish |
| M-08 | Pipeline orchestrator | `medbridge/core/orchestrator.py` | Stage methods, tracing, speculative retrieval, session locks |
| M-09 | Response templates | `medbridge/core/templates.py` | Configurable emergency numbers, small-talk texts |
| M-10 | Resilient LLM wrapper | `medbridge/ai/llm_wrapper.py` | Native structured outputs, local-only enforcement, truncation handling |
| M-11 | Context extractor | `medbridge/ai/context_extractor.py` | History window, BP reconciliation, drug normalisation |
| M-12 | Context Gate | `medbridge/ai/context_gate.py` | Cardiology scope, 12-field vocabulary, routing rules |
| M-13 | Evidence Gate | `medbridge/ai/evidence_gate.py` | Ordered rules for unknown products, doses, false premises |
| M-14 | Response generator | `medbridge/ai/response_generator.py` | Index citations, patient-friendly style, evaluation modes |
| M-15 | Embedding client | `medbridge/retrieval/embedder.py` | Concurrent encoding, cache, warm-up |
| M-16 | Hybrid retriever | `medbridge/retrieval/hybrid_retriever.py` | Explicit RRF k = 60, result cache, explicit errors |
| M-17 | Cross-encoder reranker | `medbridge/retrieval/reranker.py` | Automatic GPU selection, warm-up |
| M-18 | Ingestion pipeline | `medbridge/ingestion/` | HTML sources, manifest, token-based chunking, tables |
| M-19 | Application entry point | `medbridge/main.py` | Background warm-up, readiness, retention, web mount |
| M-20 | Streamlit frontend | `medbridge/frontend/` | Session tokens; kept as fallback |

Table: Modules added in the final phase
| ID | Module | Location | Responsibility |
|---|---|---|---|
| N-01 | Privacy layer | `medbridge/privacy/` | Redaction, encryption, crypto-shredding, tokens, middleware, log scrubbing, retention |
| N-02 | Language helpers | `medbridge/nlp/` | Drug normalisation, query expansion, BP parsing |
| N-03 | Citation validator | `medbridge/core/citations.py` | Validate and renumber `[n]` markers; rebuild citations |
| N-04 | Small-talk detector | `medbridge/core/smalltalk.py` | Claim-free handling of greetings and thanks |
| N-05 | Pipeline types | `medbridge/core/pipeline_types.py` | Stage labels, progress events, ablation presets |
| N-06 | Calibrated Evidence Gate (V1) | `medbridge/ai/evidence_calibration.py` | Permutation votes and evidence-sufficiency thresholds |
| N-07 | Grounding verifier (V2) | `medbridge/ai/grounding_verifier.py` | Sentence-level NLI check of generated claims |
| N-08 | Turn store | `medbridge/state/turn_store.py` | Single-flush persistence, encrypted reads, row locking |
| N-09 | Complementary features | `medbridge/features/` | BP log with categories, clinician summary, feedback |
| N-10 | Operations endpoints | `medbridge/api/routes/metrics.py`, `admin.py` | Metrics, admin login and audit feed |
| N-11 | Web application | `medbridge/web/` | Chat interface, explanation panel, dashboard |
| N-12 | Evaluation package | `evaluation/` | HalluBench, runner, judges, calibration, statistics |
| N-13 | Benchmark harnesses | `tests/benchmark/`, `tests/adversarial/`, `scripts/` | Live AQ, RABBITS, adversarial, smoke and load tests |

## Algorithms

This section specifies the decision procedures as pseudocode that summarises the implementation faithfully; each subsection names the source file.

### Emergency Fast-Path

The emergency classifier (`medbridge/core/emergency_classifier.py`) is a deterministic rules engine of regular expressions compiled at import time and dictionary look-ups, with no model, network or randomness. It runs on the raw message, because redaction placeholders could hide cues, and its input is never logged (Algorithm 3.1).

```algorithm
Deterministic emergency fast-path: classify_detailed
Input: raw message m
Output: ESCALATE or NONE, with matched, negated and suppressed rule ids
1  t <- NORMALISE(m)        # NFKC, lower case, collapse letter runs, misspelling map,
                            # one-edit typo matcher for red-flag words, Hinglish variants
2  R <- EXTRACT_BP_READINGS(t)   # "190/125", "190 over 125", "top number 200", "in the 190s"
3  for each reading (sys, dia) in R that is not a date, dose, lab value, target or threshold:
4      if BP wording present and (sys >= 180 or dia >= 120) then fire bp_crisis
5      if no BP wording and sys >= 180 and dia >= 120 then fire bp_crisis
6      if sys < 80 and dizziness/faintness/confusion affirmed then fire bp_low_symptomatic
7  for each red-flag rule r (chest pain, stroke signs, syncope, breathlessness, severe
       headache, seizure, haemoptysis, angioedema, self-harm, pre-eclampsia, Hinglish rules):
8      for each match of r in t, within its clause (split at . ; ? ! "but" "however" ...):
9          if NEGATED(match, clause) then mark r negated     # "no", "denies", "nahi"
10         else if INFORMATIONAL or HISTORICAL or HEAVY_EXERTION then mark r suppressed
11         else fire r
12 return ESCALATE if any rule fired else NONE
   # a negated or suppressed symptom never cancels a BP crisis or another affirmed symptom
```

The classifier has 40 rule identifiers: 26 for English (two BP rules and 24 symptom rules) and 14 for Roman-script Hindi and Indian-English phrasing. Its context filters encode deliberate trade-offs: educational questions are suppressed only when the message has no first-person, present-tense marker; historical mentions need an explicit time anchor and no recency word in the same clause; and side-effect questions that name a red flag are escalated, because the fast-path prefers recall when a message is ambiguous. The test suite enforces a 10 ms budget per message.

### Context Gate and Soft-Ask Loop-Breaker

Gate 1 (`medbridge/ai/context_gate.py`) receives the snapshot after this turn's projection, the patient message and the extractor's intent, and returns SOFT-ASK or PROCEED with a rationale and missing fields from a fixed vocabulary of twelve facts (current medications, BP reading, age, sex, pregnancy status, kidney function, heart rate, symptom details, diagnosis, medication dose, comorbidities, laboratory results). It asks only when all three conditions hold: the question concerns the patient's own cardiovascular care and is answerable from guidelines; a specific personal fact would change the safe answer; and that fact is missing. General, out-of-domain, fabricated-drug and false-premise questions always PROCEED so that Gate 2 can handle them. Because asking indefinitely would itself be unsafe, the loop-breaker limits consecutive SOFT-ASK turns to two (Figure 3.4, Algorithm 3.2).

[[figure:report/figures/fsm_soft_ask.png|Finite-state machine of the Context Gate and the soft-ask loop-breaker (threshold 2)|width=5.5]]

```algorithm
Context Gate with loop-breaker (threshold T = 2)
Input: turn context with stored counter c = soft_ask_count before this turn
1  if speculative retrieval is enabled then start RETRIEVE as a background task
2  d <- GATE1_LLM(snapshot, patient message, intent)     # temperature 0.0, seed 42
3  if d is None then d <- PROCEED (source = fallback)
4  if d = PROCEED then
5      c <- 0; continue to Gate 2 with the retrieval result
6  else if c < T then                                    # SOFT-ASK
7      action <- SOFT-ASK; c <- c + 1; discard the speculative retrieval
8      generate a question for d.missing_fields (no evidence, no clinical claims)
9  else                                                  # third consecutive request
10     action <- GENERALIZE (source = loop_breaker); c <- 0
11     skip Gate 2; generate labelled general guidance from the retrieved evidence
12 persist c with the turn; record its value before the turn in the trace
```

After a SOFT-ASK, the next message is usually an answer ("I had a stroke two years ago") rather than a question. Gate 1, Gate 2 and the generator therefore receive an *effective patient message*: the original question followed by each follow-up, labelled `Original question:` and `Follow-up details from the patient:`. The emergency classifier and extractor still see only the current message, and the trace records counts, never text. A regenerated turn restarts from the counter recorded before the original turn, so regeneration can neither trip nor undo the loop-breaker.

### Speculative Retrieval

Retrieval depends only on the extractor's search query, not on Gate 1's decision. The orchestrator therefore starts it as a background task right after extraction, concurrently with Gate 1 (Algorithm 3.3), behind the `SPECULATIVE_RETRIEVAL` flag (on by default). Gate 1 still fully controls whether the result is used: on SOFT-ASK the task is cancelled and its passages never reach Gate 2 or the generator.

```algorithm
Speculative retrieval
1  task <- start RETRIEVE(search_query)              # concurrently with Gate 1
2  [g_start, g_end] <- time window of GATE1_LLM
3  if Gate 1 = SOFT-ASK and counter below threshold then
4      cancel task; record retrieval as "discarded"; evidence is never used
5  else                                              # PROCEED or loop-breaker
6      outcome <- await task
7      saved_ms <- overlap([outcome.start, outcome.end], [g_start, g_end])
8      use outcome.passages exactly as if retrieval had run after Gate 1
```

Since the passages are identical whichever order is used, routing is unchanged; unit tests check that SOFT-ASK discards the result and that output is the same with the flag on and off.

### Hybrid Retrieval with Reciprocal Rank Fusion and Reranking

Clinical questions mix semantic intent ("tablets that make me pee more") with exact terms such as drug names. MedBridge combines dense retrieval, which captures meaning [@karpukhin2020dpr], with BM25 sparse retrieval, which captures exact terms [@robertson2009bm25], a known complementary pairing [@ma2022dprhybrid]. Dense vectors come from `BAAI/bge-small-en-v1.5` (384 dimensions) [@xiao2024cpack] and sparse vectors from `Qdrant/bm25`. The two lists are fused in Qdrant with reciprocal rank fusion (RRF) [@cormack2009rrf], which scores a passage *d* as the sum over lists of 1/(*k* + rank(*d*)) with *k* = 60, and the top 20 are rescored by the cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` [@nogueira2019passage; @wang2020minilm] (Algorithm 3.4).

```algorithm
Hybrid retrieval with RRF (k = 60) and cross-encoder reranking
Input: search query q (extractor output plus clinical synonyms and generic drug names)
1  if (q, 20, "hybrid") is in the result cache and not expired then return it
2  (v_dense, v_sparse) <- encode q with both encoders concurrently (cached)
3  L_dense <- top-20 by cosine similarity; L_sparse <- top-20 by BM25 score
4  for each passage d in L_dense or L_sparse:
5      score_rrf(d) <- sum over lists L containing d of 1 / (60 + rank_L(d))
6  C <- 20 passages with the highest score_rrf           # fused inside Qdrant
7  score_rerank(d) <- CrossEncoder(q, text(d)) for d in C # worker thread, GPU if available
8  return the 5 passages with the highest score_rerank, with metadata and both scores
   # non-empty results are cached; an infrastructure failure raises RetrievalError
```

During the final phase it was found that Qdrant's unparameterised fusion uses a much smaller rank constant than 60; the constant is now passed explicitly, so the implementation matches the design. The design is evaluated in Section 4.4.

### Evidence Gate Routing

Gate 2 (`medbridge/ai/evidence_gate.py`) receives the top-5 passages in a `<clinical_evidence>` block, the snapshot in `<patient_context>` and the patient's own words in `<untrusted_user_input>`, and returns one of four actions (Algorithm 3.5). The patient's words are preferred to the search query because false premises and unknown drug names must be judged as written. The prompt applies the *first* rule that fits, and the schema puts the rationale before the action so that the model states the deciding rule before committing.

```algorithm
Evidence Gate (Gate 2) routing
Input: evidence E (top-5), snapshot S, patient message p
1  skip if the turn already ended in SOFT-ASK, the loop-breaker, an emergency or small talk
2  out <- GATE2_LLM(E, S, p)          # temperature 0.0, seed 42; first rule that fits:
   #  1. possible emergency happening now                                    -> ESCALATE
   #  2. ABSTAIN if: a. outside cardiovascular care; b. evidence irrelevant or too thin;
   #     c. a named drug, brand, supplement, device or treatment absent from E;
   #     d. a dose, regimen or number E does not state;
   #     e. dosing for children or pregnancy/breastfeeding decisions E does not cover
   #  3. E covers the topic only generally, or the question rests on a false belief
   #     that E contradicts                                                  -> GENERALIZE
   #  4. E directly and specifically answers this question for this patient -> ANSWER
   #  if torn between ANSWER and GENERALIZE, choose GENERALIZE
3  if out is None or invalid then out <- GENERALIZE (source = fallback)
4  record evidence features (max and mean rerank score, entity coverage)
5  return out.action, out.rationale
```

For a follow-up in the effective-message format, the gate judges the original question and never abstains only because the latest message has no question. The few-shot examples of both gates use topics absent from the benchmark (for example a drug name invented for the prompt), and an automated test compares every prompt example with every benchmark turn to prevent contamination.

### Grounded Generation and Citation Validation

For ANSWER and GENERALIZE, the generator (`medbridge/ai/response_generator.py`, temperature 0.3) receives the passages as `<chunk index="1">` to `<chunk index="5">` and is instructed to use only their facts, to end every clinical sentence with a marker such as `[2]`, to correct false premises, never to state numbers or drugs that are not in the evidence, and never to tell the patient to change a medicine. Single-digit markers were chosen because small local models copy long identifiers unreliably, while a digit is easy to produce and to check [@wu2025sourcecheckup; @wang2025medcite]. The validator then enforces that an answer cites only what it was shown (Algorithm 3.6).

```algorithm
Citation validation and ANSWER-to-GENERALIZE downgrade
Input: draft x, passages E = [e1..e5] in prompt order, action a
1  for each marker group in x ("[3]", "[1, 2]", "[1-3]"; at most two digits per number):
2      keep each n with 1 <= n <= |E|; drop the others (counted as stripped)
3  renumber kept markers [1], [2], ... by first appearance
4  rebuild citations from passage metadata: "ORG YEAR", section, page (none for HTML),
       URL, first ~350 characters                          # never written by the LLM
5  if a = ANSWER and no valid citation remains and the Evidence Gate is active then
6      a <- GENERALIZE               # Gate 2's ANSWER remains visible in the audit log
7  return cleaned text, citations, a
```

### V1: Calibrated Evidence Gate

The study also evaluates two variants that stay close to the design, as foreseen by the project's decision rule: if the base design did not improve significantly, the closest variants would be evaluated, and evaluating them in any case strengthens the analysis. V1 (`medbridge/ai/evidence_calibration.py`) has exactly the position and outputs of Gate 2 and adds two mechanisms inspired by self-consistency and selective prediction [@kadavath2022know; @geifman2017selective; @wen2025abstention]. Because gates must stay at temperature 0.0, diversity comes not from sampling but from deterministic permutations of the evidence order, exploiting the position sensitivity of language models; a quantitative evidence signal then acts as a veto that can only make the decision more conservative (Algorithm 3.7).

```algorithm
V1 calibrated Evidence Gate
Input: evidence E, snapshot S, message p, votes V = 3, thresholds r_answer, r_any, c_min
1  f.max_rerank <- max cross-encoder score over E (-99 if E is empty)
2  f.entity_coverage <- share of salient entities of p (drugs, numbers with units,
       clinical keywords; redaction placeholders ignored) found in E
3  votes <- GATE2_LLM on up to V orderings of E (original, reversed, rotations), concurrently
4  if every vote failed then return GENERALIZE (source = fallback)
5  a <- majority vote; ties -> more conservative (ESCALATE > ABSTAIN > GENERALIZE > ANSWER)
6  if a in {ANSWER, GENERALIZE} and f.max_rerank < r_any then a <- ABSTAIN
7  else if a = ANSWER and (f.max_rerank < r_answer or f.entity_coverage < c_min)
8      then a <- GENERALIZE
9  return a                        # ESCALATE never overridden, ABSTAIN never upgraded
```

The thresholds are calibrated on the development split only, by replaying the same pure rule on logged Gate 2 traces (Section 3.7).

### V2: Post-Generation Grounding Verifier (Gate 3)

V2 (`medbridge/ai/grounding_verifier.py`) adds a third gate after generation and before the patient sees anything, following the idea of verifying and revising generated text against evidence [@gao2023rarr; @dhuliawala2024cove; @laban2022summac]. It skips non-claims (short headings, greetings, referral and disclaimer clauses, emergency-care advice) and checks each claim with a local natural-language inference (NLI) cross-encoder, `cross-encoder/nli-MiniLM2-L6-H768`, against the passages it cites. This model is deliberately different from the NLI judge of the evaluation, so the verifier is not graded by itself (Algorithm 3.8).

```algorithm
V2 grounding verifier (Gate 3)
Input: draft x with action a in {ANSWER, GENERALIZE}, passages E, thresholds theta, tau
1  for each clinical claim s in x:
2      P <- 3-sentence windows of the passages s cites (all passages if none)
3      supported(s) <- max over w in P of NLI_entailment(w, claim(s)) >= theta
4  u <- unsupported claims / claims
5  if u <= tau then remove unsupported sentences, re-validate citations, keep a
6  else if a = ANSWER then
7      x' <- regenerate as GENERALIZE and verify again
8      a <- GENERALIZE with cleaned x' if u(x') <= tau, otherwise ABSTAIN (template)
9  else a <- ABSTAIN (template)
10 if the NLI model is unavailable, keep x and record the failure
```

V2 can only remove content or lower the action; its thresholds are calibrated on the development split against the LLM judge's sentence verdicts (Section 3.7).

### Natural-Language Robustness Helpers

Four deterministic helpers bridge patient language and guideline language without extra model calls.

- **Drug normalisation** (`medbridge/nlp/drugs.py`) maps 189 cardiovascular brand names (124 international and 65 Indian, such as Telma, Amlong, Concor and Ecosprin) to generic ingredients and drug classes, ignoring strengths, salts and release suffixes, with fuzzy matching at a cutoff of 0.88. It is applied to extracted medication events and to the search query, addressing the known fragility of language models to drug names [@gallifant2024rabbits].
- **Query expansion** (`medbridge/nlp/query_expansion.py`) appends clinical synonyms to lay terms ("water pill" to thiazide diuretic, "blood thinner" to anticoagulant) and the generics of mentioned brands, keeping the original words.
- **BP parsing** (`medbridge/nlp/bp_parse.py`) stores a value as a reading only when the text ties it to the patient ("my bp", "this morning"); targets, hypothetical values and general questions never become BP_READING events.
- **Small talk** (`medbridge/core/smalltalk.py`) answers whole-message greetings, thanks and acknowledgements of at most eight words without digits or clinical vocabulary with a claim-free GENERALIZE template; "yes" and "no" are excluded because they often answer a clarifying question, and the check runs after the emergency fast-path.

## Data Model and Event Sourcing

### Event-Sourced Patient State

Patient context is managed with event sourcing [@fowler2005eventsourcing]. The extractor never writes the patient state; it emits *delta events* of six types (BP_READING, MEDICATION_ADDED, MEDICATION_STOPPED, SYMPTOM_REPORTED, DEMOGRAPHIC, LAB_RESULT), and the event type is restricted to these values in the JSON schema, so decoding cannot invent a new type. The deterministic projector (`medbridge/state/projector.py`) folds events into a snapshot: demographics are merged, a MEDICATION_ADDED event for a listed medicine is a dose change with the previous dose kept in a history, MEDICATION_STOPPED moves a medicine to the discontinued list, and only recent readings and laboratory values are kept in the snapshot while the full series stays in the event log. The snapshot is thus always a reproducible function of the events.

### Database Schema

Figure 3.5 shows the schema and Table 3.6 the role of each table; Appendix E lists all columns.

[[figure:report/figures/er_diagram.png|Entity-relationship diagram of the MedBridge PostgreSQL schema|width=6.0]]

Table: Database tables
| Table | Purpose | Mutability | Encrypted content |
|---|---|---|---|
| `sessions` | Conversation, activity and erasure times | Never deleted | None |
| `clinical_events` | Typed delta events | Append-only (database rules) | Payload |
| `context_snapshots` | Projected snapshot and soft-ask counter | Updated under a row lock | Snapshot |
| `audit_logs` | One row per turn: request, gate actions and rationales, final action, evidence ids, response, latency, trace | Append-only (database rules) | Request, rationales, response, private trace |
| `message_history` | User and assistant messages | Deleted on erasure | Text |
| `session_keys` | Wrapped per-session data key | Deleted on erasure (crypto-shredding) | Key is wrapped |
| `feedback` | One rating per assistant message | Deleted on erasure | Comment |

Four PostgreSQL rules created in the initial migration reject UPDATE and DELETE on `clinical_events` and `audit_logs`, so immutability holds even if application code is wrong. The audit row is the immutable record of every gate decision and rationale; its `trace` keeps the identifier-free part of the `Explanation` (actions, sources, scores, timings, counts) in plaintext for metrics and encrypts any field that could contain patient text.

### Single-Flush Persistence and Concurrency

The original code flushed the database five or more times per turn. The turn store (`medbridge/state/turn_store.py`) now performs all reads first, then adds the events, snapshot and counter, both messages, one audit row and the activity time, and flushes exactly once. Within a process, turns of a session are serialised by an asynchronous lock held from state loading to persistence; across processes, the store locks the snapshot row (`SELECT … FOR UPDATE`) and re-applies only its own events, so the stored snapshot always equals the fold of the stored events. A regenerated reply is stored only if the message it replaces is still the latest. Figure 3.6 shows the data flow of a turn and the offline build of the knowledge base.

[[figure:report/figures/dataflow.png|Data-flow diagram: run-time flow of one patient turn between the processes and data stores (top) and the offline knowledge-base build (bottom)|width=6.2]]

## Privacy and Security Design

Health conversations contain direct identifiers and sensitive clinical details. MedBridge protects both with layered controls (Figure 3.7) that follow the data-minimisation, purpose-limitation, storage-limitation and security-safeguard principles of India's Digital Personal Data Protection Act, 2023 [@dpdp2023india] and the HIPAA Privacy Rule [@hhs_hipaa_privacy_rule]. The design aims to align with these principles; it is not a claim of legal compliance.

[[figure:report/figures/privacy_flow.png|Privacy controls along the request path: redaction, local-only inference, encryption at rest, signed tokens, log scrubbing, retention and crypto-shredding|width=6.2]]

### PII and PHI Redaction

The first stage of every turn replaces identifiers with typed placeholders before any model call, retrieval, log line or database write (`medbridge/privacy/redaction.py`). The detector is rule-based: patterns and contextual cues for names (after "my name is", greetings, sign-offs, titles), phone numbers, e-mail addresses, postal addresses and PIN or ZIP codes, Aadhaar, PAN and Social Security numbers, record and insurance numbers, URLs and IP addresses. An exact date of birth becomes an age placeholder such as `[AGE: 54]`, keeping the clinically relevant information in the spirit of de-identification guidance [@hhs2012deidentification]. BP readings, laboratory values, drug names and doses are never altered. A rule-based detector was preferred to a statistical one such as Presidio [@presidio2026] because it needs no model, runs in a fraction of a millisecond and can be audited; its lower recall on names without a cue is measured in Section 4.6. Redaction fails closed, and only the emergency classifier sees the raw text.

### Local-Only Inference

With `LOCAL_ONLY_MODE=true` (the default), the LLM wrapper builds only the Ollama provider, never contacts the optional remote providers, and accepts an Ollama endpoint only on a loopback address (`localhost`, `127.0.0.0/8`, `::1`) serving a local, non-cloud model. Any other configuration blocks every call and the specification fallbacks apply, so no patient text leaves the machine.

### Encryption at Rest and Crypto-Shredding Erasure

Stored content uses envelope encryption (`medbridge/privacy/crypto.py`): a master key from the environment, never stored in the database, wraps one random data key per session, kept in `session_keys`, following the separation of key-encrypting and data-encrypting keys in key-management guidance [@barker2020sp80057pt1r5]. The data key encrypts message and audit text, rationales, event payloads, the snapshot, feedback comments and the private trace with Fernet authenticated encryption (AES-128-CBC with HMAC-SHA256); readers decrypt transparently.

The right to erasure seems to conflict with the append-only invariant, since audit and event rows can never be changed or deleted. Crypto-shredding resolves this: destroying the key makes every ciphertext of the session permanently unreadable, the cryptographic-erase technique of media-sanitisation guidance [@nist2025sp80088r2], while the append-only rows stay byte-for-byte unchanged (Algorithm 3.9).

```algorithm
Right to erasure by crypto-shredding: DELETE /api/sessions/{id}
Input: session id s, bearer token for s (accepted up to the retention window after expiry)
1  verify token signature and session binding; otherwise 401/403
2  in one transaction:
3      count append-only rows of s still in legacy plaintext
4      delete the row of s in session_keys              # the data key is destroyed
5      delete rows of s in message_history, context_snapshots, feedback
6      set sessions.erased_at                           # events and audit rows untouched
7  invalidate the in-process key cache for s
8  return 204; later requests for s return 410 before any pipeline work
```

The retention job, which runs about 30 seconds after start-up and then hourly, applies the same procedure to sessions inactive for longer than `DATA_RETENTION_DAYS` (30 by default). Rows written before encryption existed cannot be rewritten; they are counted, flagged and hidden by every reader.

### Authentication, Access Control and Transport Security

Review 1 listed "no authentication (session ID only)" as a limitation. Every session route now requires a signed, expiring bearer token `v1.<payload>.<signature>`, an HMAC-SHA256 over a payload bound to one session (`medbridge/privacy/tokens.py`), compared in constant time; a missing token yields HTTP 401 and an invalid or expired one HTTP 403. The dashboard is disabled unless an administrator password is set; login is rate-limited, issues an eight-hour token, and the audit feed shows only redacted previews or `[erased]`.

Outside the pipeline, the middleware adds a strict Content Security Policy (scripts only from the application's origin, no framing), standard security headers, `Cache-Control: no-store` on API responses, CORS restricted to configured origins, per-IP sliding-window rate limits on messages (configurable), session creation (10 per minute) and admin login (5 per minute) with HTTP 429 and `Retry-After`, and a request-size limit (HTTP 413), all as pure ASGI middleware so streams are not buffered. A logging processor scrubs identifiers from every log value (`medbridge/privacy/log_scrub.py`).

### Prompt-Injection Defence

Prompt injection is the leading risk for LLM applications [@owasp2025llmtop10], including indirect injection [@greshake2023notwhat; @liu2024formalizing]. All patient-derived text, including earlier turns and the extractor's intent, is placed in `<untrusted_user_input>` blocks that the prompts declare to be data, never instructions, and tag-like sequences are neutralised before insertion, including unterminated tags, look-alike angle brackets and invisible characters (`medbridge/ai/prompt_utils.py`). Architecturally, the most safety-critical decisions do not depend on the generator: the emergency fast-path is deterministic, ABSTAIN and ESCALATE are templates, and grammar-constrained decoding confines each gate to its labels.

### Privacy Principles and User-Facing Controls

Table 3.7 maps the principles to mechanisms. In the interface, a first-visit consent and privacy notice explains what is stored and for how long, a compact non-dismissible disclaimer states that the assistant does not diagnose or prescribe, a "PII redacted (n)" indicator marks messages from which identifiers were removed, and a "clear my data" control calls the erasure endpoint.

Table: Privacy principles and their implementation in MedBridge
| Principle (DPDP Act 2023 / HIPAA) | MedBridge mechanism |
|---|---|
| Data minimisation | Identifiers redacted before use and storage; date of birth reduced to age; trace stores counts |
| Purpose limitation | Data used only for the conversation; evaluation uses synthetic data |
| Storage limitation | Inactive sessions erased after a configurable period (default 30 days) |
| Security safeguards | Per-session encryption, signed tokens, security headers, rate limits, loopback-only services |
| Right to erasure | Crypto-shredding with the append-only audit trail preserved |
| Notice and consent | First-visit consent notice; persistent disclaimer |
| No disclosure to third parties | Local-only inference by default |
| Accountability | Immutable audit log of every gate decision; dashboard without direct identifiers |

## Experimental Design

The central research question (RQ1 in Chapter 1) is whether the Dual Answerability Gate reduces hallucinated and unsafe answers. Figure 3.8 summarises the study design.

[[figure:report/figures/evaluation_design.png|Design of the hallucination study: HalluBench splits, conditions C0 to C5 and V1/V2, the two judges and the human subset, calibration on the development split and statistical analysis on the test split|width=6.3]]

### Benchmark: MedBridge-HalluBench

MedBridge-HalluBench v1 (`evaluation/datasets/hallubench_v1.json`, data card `evaluation/datasets/HALLUBENCH.md`) is a fully synthetic benchmark of 247 items with 298 user turns in natural, often casual phrasing. Each item has a category, subcategory, style, one to three turns with a gold action and a list of acceptable actions, gold evidence references and notes naming every fabricated entity and false premise. The six categories mirror the routing semantics, so that SOFT-ASK first turns and loop-breaker sequences test Gate 1, the ANSWER, GENERALIZE and ABSTAIN distinctions test Gate 2, and emergency items test the fast-path. The composition, computed from the dataset file, is shown below.

[[table:hallubench_composition]]

Unanswerable items include fabricated drugs and treatments, false premises, dose questions no guideline answers, requests for specific numbers, and paediatric and pregnancy edge cases; out-of-domain items are split evenly between non-cardiac medical and non-medical questions. The styles are casual English (155 items), Indian English (35), typo-laden (33) and clinical (24), and 43 items are multi-turn. Gold evidence was labelled by regular-expression search over the parsed corpus, never with the MedBridge retriever, and cites 120 distinct pages across all 27 documents.

A stratified split with seed 42 gives a development split of 49 items (59 turns) and a test split of 198 items (239 turns). The development split is reserved for development decisions (prompt refinement, model selection and threshold calibration); the test split is never used for tuning, and all reported results use it. Fourteen items close to examples in the original gate prompts are tagged `prompt_overlap` for a sensitivity analysis. Acceptable actions are lenient only where neighbouring actions are equally safe: answerable items accept ANSWER or GENERALIZE, most unanswerable items ABSTAIN or GENERALIZE, paediatric, pregnancy-dosing and out-of-domain items only ABSTAIN, and emergency turns only ESCALATE.

### Conditions

Conditions are internal ablation presets (`AblationConfig` in `medbridge/core/pipeline_types.py`) passed to the production orchestrator; they are never exposed through the API, and production always runs the full configuration. All conditions share the application model, the retrieval and, where applicable, the prompts, so differences come from pipeline structure alone. The definitions are given below.

Table: Experimental conditions of the hallucination study
| Condition | Retrieval | Gate 1 | Gate 2 | Generator prompt mode | Fast-path |
|---|---|---|---|---|---|
| C0 Direct LLM | No | No | No | `direct`: model knowledge, no evidence | Off |
| C1 Vanilla RAG | Yes | No | No | `vanilla`: answer from passages with `[n]`, no safety rules | On |
| C2 RAG + prompt safety | Yes | No | No | `prompt_safety`: vanilla plus a prose safety block | On |
| C3 Context Gate only | Yes | Yes | No (always ANSWER) | `grounded` (production) | On |
| C4 Evidence Gate only | Yes | No (always PROCEED) | Yes | `grounded` | On |
| C5 MedBridge (dual gate) | Yes | Yes | Yes | `grounded` | On |
| V1 Calibrated Gate 2 | Yes | Yes | V1 | `grounded` | On |
| V2 Triple gate | Yes | Yes | Yes, plus Gate 3 | `grounded` | On |
| V12 (optional) | Yes | Yes | V1, plus Gate 3 | `grounded` | On |

The C2 safety block asks, in prose, for what the gates do: ask for missing personal facts, say when the evidence does not cover the question, decline non-cardiovascular topics, correct false premises, never invent drugs or doses, never advise changing medication, and send red flags to emergency care. C2 versus C1 thus measures prompt-only safety and C5 versus C2 what the gates add beyond instructions. An optional reference condition, C1G (retrieval plus the production prompt, no gates), lets the gate effects be measured from a baseline that differs only by the gates. The fast-path is constant across C1 to C5 and the variants, so emergency items favour no gated condition; C0 is a pure language-model baseline without it. The research questions map as follows: C5 against C0, C1 and C2 for RQ1; C3 and C4 against C5 for RQ2; V1, V2 and V12 against C5 for RQ3; latency per condition for RQ4. Each item runs in a fresh session in a dedicated evaluation database, with the turns of a multi-turn item sent in order; the unit of analysis is the *item-turn*.

### Hallucination Judges

1. **LLM judge.** A larger local model from a different family than the application model, `gemma3:12b` [@gemmateam2025gemma3], avoids the self-preference of models judging their own output [@zheng2023judging; @panickssery2024llm]; the tool warns if the families coincide. At temperature 0.0, seed 42 and with structured output, it (i) decomposes each response into atomic clinical claims, in the manner of FActScore [@min2023factscore], excluding only pure boilerplate that names no clinical entity (excluded items are still labelled for a sensitivity analysis); (ii) labels each claim SUPPORTED, UNSUPPORTED or CONTRADICTED against the condition's own passages plus the top-3 passages it retrieves independently for each claim, so that C0, which has no retrieval, is judged on equal terms, with a strict rubric under which only explicit statements or direct paraphrases count as support; and (iii) classifies the behaviour (answered specifically, general information, clarification, abstained, escalated) and flags accepted false premises, endorsed fabricated entities and unsupported doses using the gold notes. The judge never sees the condition or category; template responses get zero claims without a call; claims are labelled in batches of five; and output truncated by the token limit is retried once with a larger budget and otherwise recorded as an error.
2. **NLI judge.** The cross-encoder `cross-encoder/nli-deberta-v3-base` [@he2023debertav3], trained for natural-language inference [@williams2018mnli], scores the same claims against the same passages over three-sentence windows. A claim is SUPPORTED if entailment reaches θ, CONTRADICTED if contradiction reaches θ_c and exceeds entailment, and UNSUPPORTED otherwise.
3. **Human subset.** A blinded sheet of 50 test items, stratified by category with seed 42, contains the C5 and C1 claims with their evidence for two annotators (Appendix D).

Agreement is measured with Cohen's κ [@cohen1960coefficient] at claim level and interpreted with conventional bands [@landis1977measurement].

### Metrics

The metrics are defined below exactly as implemented in `evaluation/analyze.py`.

Table: Metrics of the hallucination study
| Metric | Definition |
|---|---|
| Hallucination rate | Responses with at least one UNSUPPORTED or CONTRADICTED claim / all responses (per judge); templates have zero claims |
| Unsupported-claim ratio | (UNSUPPORTED + CONTRADICTED) / all claims, pooled |
| Unsafe-answer rate | Unanswerable items only: behaviour "answered specifically", any behaviour flag, or at least one UNSUPPORTED/CONTRADICTED claim |
| Appropriate abstention | Unanswerable items: behaviour in the acceptable actions and not unsafe |
| Coverage / over-abstention | Answerable turns with gold ANSWER or GENERALIZE: share answered / share abstained |
| Risk-coverage | Coverage = answered / judged; risk = hallucination among answered; ranked by maximum rerank score [@geifman2017selective] |
| Routing | Strict and lenient accuracy, per-class precision, recall, F1, macro-F1, confusion matrix |
| Emergency | Recall on gold-ESCALATE turns; false-escalation rate |
| Citation quality | Marker validity = valid / (valid + stripped) markers; supported citation precision = markers whose passage supports a claim of the sentence (LLM judge, and NLI) |
| Latency | Raw and cache-imputed p50 and p95 per condition |
| Degraded turns | Count, rate and reasons per condition; excluded from headline metrics |

### Statistical Analysis

Proportions carry Wilson 95 per cent intervals [@wilson1927probable]; the unsupported-claim ratio has a cluster-bootstrap interval with 10,000 resamples and seed 42 [@efron1993bootstrap]. Conditions are compared on the same item-turns with the exact two-sided McNemar test [@mcnemar1947note]: C5 against each of C0 to C4, and V1, V2 and V12 against C5, for the hallucination and unsafe-answer outcomes under each judge, with Holm adjustment within each family, outcome and judge [@holm1979simple]. Cochran's Q tests whether C0 to C5 differ at all [@cochran1950comparison]. Effect sizes are the paired risk difference (Newcombe interval), the conditional odds ratio (exact interval, Haldane correction for zero counts) and the relative risk reduction; the formulas are documented in `evaluation/metrics.py` and checked against standard implementations in the tests. Sensitivity analyses include degraded turns, exclude `prompt_overlap` items, disable the boilerplate filter and work at item level, and every headline metric is stratified by category.

The decision rule is fixed in advance: if C5 does not reduce the hallucination or unsafe-answer rate significantly (Holm-adjusted *p* < 0.05) against C1 and C2, the result is reported as it is, with an error analysis of failing categories, false PROCEED and false ANSWER decisions and judge disagreements, and V1 and V2 are evaluated under the same protocol. The variants are evaluated in either case, without overstating their effect.

### Calibration Protocol

Calibration uses the development split only, in an order the code enforces: C5 is run and judged on the development split, `evaluation.calibrate` writes `evaluation/results/calibration.json`, and only then are all conditions run on the test split. The runner refuses V1, V2 and V12 on the test split without the corresponding calibration block, injects the values into the run's settings and records them with the file checksum in every record.

- **NLI judge.** θ maximises binary κ with the LLM judge over a grid from 0.05 to 0.95; θ_c then maximises three-class κ.
- **V1.** A grid search replays the V1 rule on logged C5 traces, minimising false ANSWERs on unanswerable turns subject to coverage of at least C5's coverage minus 0.05.
- **V2.** C5 responses are scored sentence by sentence with the production verifier; θ maximises κ with the LLM judge's sentence verdicts, and τ minimises the hallucination rate while keeping at least 85 per cent of C5's answered coverage.

### Degraded Turns, Memoisation and Reproducibility

A turn that took a fallback path may look safe without the gates having worked. `evaluation/degraded.py` therefore flags any gate fallback, failed V1 vote, fallback template, failed stage, retrieval failure, unavailable verifier, failed LLM call or runner error, and every later turn of an affected item. Degraded turns are re-run (bypassing the cache only for the failed calls), excluded from headline metrics and included in a sensitivity analysis; the analysis refuses headline results if more than 5 per cent of any condition's turns are degraded unless explicitly overridden.

Identical language-model requests are memoised in a cache keyed by a hash of the full request (model, messages, schema, temperature, seed, context size), so a request issued identically by two conditions is computed once and paired conditions differ only in pipeline structure. The cache is namespaced by the Ollama digests of the models, stores only complete outputs, and latency is reported both raw and imputed (each cached call charged the median uncached latency). Each run writes a manifest with the git commit, dataset checksum, model digests and quantisation, hardware, seeds, settings and calibration status; raw records keep the full explanation and evidence, and every figure and table is generated from these files (`evaluation/README.md`).

### Supporting Experiments

The following component experiments are reported in Chapter 4.

- **Retrieval evaluation** (`evaluation/retrieval_eval.py`): 70 labelled queries (26 lay, 44 clinical, 13 topics, 114 relevant spans); Recall@5, Recall@20, MRR and nDCG@5 for dense, sparse, hybrid and hybrid-plus-rerank retrieval, plus an ablation of four rerankers. A passage is relevant if it comes from the labelled document and overlaps a labelled page.
- **Model selection** (`evaluation/model_selection.py`): C5 on the development split for each candidate model; routing accuracy, macro-F1, first-attempt JSON validity, latency and throughput.
- **PII redaction** (`scripts/pii_eval.py`): span-level precision, recall and F1 on a development set and a blind held-out set, plus clinical-value protection.
- **Emergency classifier**: labelled English and Hinglish case files checked by the unit tests.
- **MedBridge-AQ** (`tests/benchmark/`): 275 vignettes (93 ANSWER, 50 SOFT-ASK, 43 GENERALIZE, 42 ABSTAIN, 47 ESCALATE) run live with routing accuracy, citation precision and latency.
- **RABBITS** (`tests/adversarial/rabbits_runner.py`): 189 brands in two templates differing only in the drug name; action consistency, normalisation agreement and citation overlap [@gallifant2024rabbits].
- **Adversarial security** (`tests/adversarial/`): 19 injection and jailbreak payloads scored deterministically, plus hermetic isolation, SQL-injection and authentication tests.
- **Performance and load** (`scripts/latency_probe.py`, `scripts/load_test.py`): cold and warm latency per action before and after optimisation, and latency, throughput and errors under 1, 5 and 10 concurrent users.

## Tools, Technologies and Datasets

### Software Stack and Models

The main technologies are listed below with the versions pinned in `pyproject.toml`.

Table: Software stack
| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.11 |
| Web framework | FastAPI, Uvicorn | 0.143.0, 0.54.0 |
| Validation and settings | Pydantic, pydantic-settings | 2.14.0, 2.15.0 |
| Database | PostgreSQL (`postgres:16-alpine` image); SQLAlchemy async, asyncpg, Alembic | 16; 2.1.4, 0.32.0, 1.20.0 |
| Vector store | Qdrant server (Docker), qdrant-client | client 1.19.1 |
| Embeddings | fastembed | 0.9.0 |
| Cross-encoders | sentence-transformers (PyTorch) | 6.1.0 |
| LLM runtime | Ollama | 0.35.0 |
| Security and logging | cryptography (Fernet), structlog | 50.0.2, 26.1.0 |
| Document parsing | PyMuPDF | 1.28.2 |
| Interfaces | HTML, CSS, JavaScript modules; Streamlit (legacy) | —; 1.65.0 |
| Analysis and reporting | NumPy, SciPy, Matplotlib, python-docx | 2.4.6, 1.17.1, 3.11.2, 1.2.0 |
| Quality and infrastructure | pytest, ruff, mypy; Docker Compose | — |

All language models are open-weight models served by Ollama at 4-bit quantisation (Q4_K_M), so the application model fits the 6 GB GPU of the target laptop.

Table: Models used by MedBridge and its evaluation
| Role | Model | Notes |
|---|---|---|
| Application LLM (extractor, gates, generator) | [TO BE FILLED: final application model chosen by the model-selection run, Section 4.5] | Candidates `qwen3:8b` (repository default), `qwen3.5:4b`, `gemma3:4b` [@yang2025qwen3; @gemmateam2025gemma3] |
| Dense embedder | `BAAI/bge-small-en-v1.5` | 384 dimensions [@xiao2024cpack] |
| Sparse encoder | `Qdrant/bm25` | BM25 weights [@robertson2009bm25] |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Also the V1 evidence signal [@wang2020minilm] |
| Grounding verifier (V2) | `cross-encoder/nli-MiniLM2-L6-H768` | Different from the NLI judge |
| LLM judge (evaluation) | `gemma3:12b` | Different family from the application model |
| NLI judge (evaluation) | `cross-encoder/nli-deberta-v3-base` | Thresholds calibrated on development split [@he2023debertav3] |

### Knowledge-Base Corpus

The knowledge base grew from the 3 PDF guidelines of Review 3 to {{corpus.totals.documents|int}} curated, publicly readable documents from {{corpus.totals.organisations|int}} organisations (table below): guidelines of AHA/ACC, ESC, ISH, WHO, KDIGO and NICE [@jones2025aha; @mcevoy2024esc; @nice2019ng136; @who2021pharm; @unger2020ish; @kdigo2021bp; @joglar2024af; @arnett2019primary], scientific statements and position papers, an openly licensed review of the 2023 chronic coronary disease guideline, and MedlinePlus and NHLBI patient pages for lay language. To avoid corrupting the knowledge base with unvetted text, nothing is scraped: a document is indexed only if `data/corpus_manifest.json` lists its organisation, year, URL, licence note and SHA-256 checksum, and every chunk carries the document checksum. One source was excluded for its licence, and several major guidelines could not be obtained automatically (see the limitations section of Chapter 4).

[[table:corpus]]

Documents are parsed with PyMuPDF or an HTML parser, cleaned of headers, reference lists and tables of contents, and split into chunks of at most 400 tokens with a 60-token overlap, measured with the embedder's tokenizer so that nothing exceeds the 512-token window of the embedder and reranker; tables become separate Markdown chunks. The index holds {{corpus.totals.chunks|comma}} chunks ({{corpus.totals.table_chunks|comma}} tables), with a median of {{corpus.totals.chunk_tokens.median|int}} tokens and {{corpus.totals.chunk_tokens.over_512|int}} chunks over 512 tokens. Ingestion is incremental, skipping unchanged documents by checksum.

### Evaluation Datasets and Hardware

The datasets used for testing and evaluation are summarised below; all are synthetic or derived from public documents.

Table: Evaluation and test datasets
| Dataset | Location | Size | Used for |
|---|---|---|---|
| MedBridge-HalluBench v1 | `evaluation/datasets/hallubench_v1.json` | 247 items, 298 turns (dev 49/59, test 198/239) | Hallucination study, calibration, model selection |
| Retrieval evaluation set | `evaluation/datasets/retrieval_eval_v1.json` | 70 queries, 114 spans | Retrieval metrics, reranker ablation |
| PII development set | `tests/privacy/pii_testset.json` | 240 messages, 183 spans | Redaction development |
| PII blind held-out set | `tests/privacy/pii_testset_blind_v2.json` | 237 messages, 203 spans | Single blind evaluation |
| Emergency case files | `tests/data/emergency_cases*.json` | 288 English, 141 Hinglish | Emergency classifier tests |
| MedBridge-AQ vignettes | `tests/benchmark/vignettes/` | 275 vignettes | Live routing benchmark |
| RABBITS drug pairs | from `medbridge/nlp/drugs.py` | 189 brands × 2 templates | Brand/generic robustness |
| Adversarial payloads | `tests/adversarial/payloads.py` | 19 payloads | Injection and jailbreak tests |

All experiments in this report were run on an Apple MacBook with an M5 chip and 24 GB of unified memory (macOS, Python 3.11, Ollama 0.35.0, Metal for Ollama and MPS for the cross-encoders). The target demonstration machine is a Windows 11 laptop with an NVIDIA RTX 4050 Laptop GPU (6 GB); the software is cross-platform with shell and PowerShell scripts, but its performance on that machine was not measured by the authors of this report.

## Chapter Summary

This chapter defined the method: a deterministic emergency fast-path followed by two separate answerability gates, one for patient context before retrieval and one for evidence after it, with validated grounded generation, template bypass for refusals and emergencies, and a never-throw orchestrator. It described the architecture, workflow and modules, the algorithm of every decision point including the V1 and V2 variants, the event-sourced data model, and a privacy design in which crypto-shredding reconciles erasure with immutability. It specified the hallucination study, with a synthetic benchmark, nine conditions, two judges, enforced calibration discipline and paired statistics, together with the supporting experiments, tools, models and datasets. Chapter 4 reports the implementation and results.
