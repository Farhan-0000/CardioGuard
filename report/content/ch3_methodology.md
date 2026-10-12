# Methodology and System Design

This chapter presents the method and the system; paths are relative to the repository root.

## Proposed Methodology

MedBridge takes the decision to answer twice, for the two reasons given in Section 1.2. A **Context Gate** (Gate 1) before retrieval asks whether a missing fact about the patient, such as the current medicine or pregnancy status, would change the safe answer, and if so asks for it. An **Evidence Gate** (Gate 2) after retrieval asks whether the passages support a specific answer, or whether the question names an unknown drug, rests on a false premise, asks for a number the guidelines do not give or lies outside cardiovascular care. Together they form the **Dual Answerability Gate**, the contribution of the project, behind a deterministic **emergency fast-path** that runs before any language-model call. The design principles are *safety before fluency* (failures and later checks can only move to a more conservative action), *determinism where it matters* (rule-based emergency handling, gates at temperature 0.0 with seed 42 under grammar-constrained decoding, fixed templates), *separate, audited decisions*, *traceable evidence*, *local-first privacy* (local inference through Ollama [@ollama2026], redaction, encryption) and *measured, not asserted*. Every turn ends in one of the actions of Table 3.1.

Table: The five response actions of MedBridge {size=9}
| Action | Chosen when | Decided by | Response text |
|---|---|---|---|
| ESCALATE | A red flag is described (systolic ≥ 180 or diastolic ≥ 120 mmHg with BP wording, chest pain, stroke signs, fainting, ...) | Emergency fast-path; Gate 2 can also escalate | Pre-vetted emergency template with configurable numbers (no LLM) |
| SOFT-ASK | A missing personal fact would change the safe answer | Gate 1, at most twice in a row | Short question for the missing facts, no clinical claims |
| ANSWER | Context and evidence are both sufficient | Gate 2 | Grounded, patient-friendly answer with inline `[n]` citations |
| GENERALIZE | Evidence supports only general information, a false premise must be corrected, or the loop-breaker fires | Gate 2, loop-breaker, citation downgrade, Gate 3 (an ANSWER failing verification), small-talk rule | General guideline information, labelled as not personalised |
| ABSTAIN | Evidence irrelevant or absent, topic out of scope, unknown product or unsupported dose | Gate 2 or Gate 3 (on by default since `7a3bc67`) | Pre-vetted refusal recommending a clinician (no LLM) |

Changes followed a written plan (`docs/finalization_plan.md`) and were recorded in the Architecture Decision Log under seven hard invariants: emergency fast-path before any LLM call; append-only events and audit logs; gates at temperature 0.0 with seed 42; loop-breaker limit of two; patient text in `<untrusted_user_input>` blocks; template bypass for ABSTAIN and ESCALATE; and no internal error reaching the patient.

## System Architecture

MedBridge is a single-node application organised in layers (Figure 3.1, Table 3.2). The twenty approved modules M-01 to M-20 were kept, most extended, and M-21 to M-27 added; `docs/module_map.md` lists the files, tests and decision-log entries of each.

Table: Layers of MedBridge and the modules they contain {size=9}
| Layer | Location | Modules |
|---|---|---|
| Presentation | `medbridge/web/`, `frontend/` | Chat application and dashboard (M-22); Streamlit client kept as fallback (M-20) |
| API | `medbridge/api/` | Schemas and error handler (M-03); routes for sessions, messages and streaming, regeneration, feedback, context, BP, summary, metrics, administration (M-04) |
| Pipeline core | `medbridge/core/` | Orchestrator with pipeline types, citation validator and small-talk detector (M-08); emergency classifier (M-07); templates (M-09) |
| AI and language | `medbridge/ai/`, `nlp/` | LLM wrapper (M-10); context extractor (M-11); Context Gate (M-12); Evidence Gate (M-13); generator (M-14); V1 and V2 (M-26); drug, query and BP helpers (M-23) |
| Retrieval and ingestion | `medbridge/retrieval/`, `ingestion/` | Embedding client (M-15); hybrid retriever (M-16); reranker (M-17); ingestion, the only writer to the vector store (M-18) |
| State and persistence | `medbridge/state/`, `db/` | Session manager and single-flush turn store (M-05); projector (M-06); database layer and migrations (M-02) |
| Privacy and features | `medbridge/privacy/`, `features/` | Redaction, encryption, tokens, middleware, log scrubbing, retention (M-21); BP log and clinician summary (M-24) |
| Application and tooling | `medbridge/`, `evaluation/`, `tests/`, `scripts/` | Configuration (M-01); entry point (M-19); evaluation package (M-25); benchmark and operations harnesses (M-27) |

[[figure:report/figures/architecture_pipeline.png|MedBridge architecture and request pipeline|width=4.5]]

**Eight-step decision workflow.** Table 3.3 maps the approved workflow to the orchestrator's stages; redaction and a small-talk check were added without changing order or routing. A typical turn makes four language-model calls, a SOFT-ASK or ABSTAIN turn three, and an emergency or small-talk turn none (Figure 3.2).

Table: The eight-step workflow and its implementation stages {size=9}
| Step | Stage(s) in code | LLM? | Outcome |
|---|---|---|---|
| 1. User clinical query | `privacy` (redactor) | No | Redacted text for every later stage except step 2 |
| 2. Emergency fast-path | `emergency` (rules on the raw text) | No | ESCALATE template, or continue |
| 3. Context extraction and state | `load_state`, `small_talk`, `extraction` | Extractor | Snapshot, recent turns, delta events, search query |
| 4. Context Gate (Gate 1) | `context_gate` with loop-breaker | Yes | SOFT-ASK, forced GENERALIZE, or PROCEED |
| 5. Hybrid retrieval and reranking | `retrieval`, `rerank` | No | Top-5 passages (may start during step 4) |
| 6. Evidence Gate (Gate 2) | `evidence_gate` (or V1) | Yes | ANSWER, GENERALIZE, ABSTAIN or ESCALATE |
| 7. Grounded response generation | `generation`, `verification` | Generator | Cited text, clarifying question or template |
| 8. Safe clinical response | `persistence`, SSE stream | No | Persisted turn with audit row and trace |

[[figure:report/figures/sequence_turn.png|Sequence of one patient turn with speculative retrieval|width=4.6]]

**Tracing, failure handling and deployment.** A turn's state lives in a `TurnContext` passed between traced stages. An `Explanation` record, returned and audited with every response, holds both gate decisions with their source (`llm`, `fallback`, `loop_breaker`, `fastpath`, `calibrated`, `rule` or `ablation`) and rationale, matched emergency rules, the top-5 passages, citation and Gate 3 outcomes and timings; it drives the "Why this answer?" panel and the evaluation. `process_message` never raises: redaction fails closed, Gate 1 falls back to PROCEED, Gate 2 to GENERALIZE, a failed retrieval leaves Gate 2 to judge empty evidence, a failed reranker keeps the fusion order, a failed generator yields a GENERALIZE template, and a top-level guard returns a safe response; the evaluation flags such *degraded* turns. MedBridge runs on one machine (Figure 3.3): PostgreSQL 16 and Qdrant in loopback-bound Docker containers, Ollama natively on the GPU, and one Uvicorn process serving the API and web interface with background warm-up and retention tasks.

[[figure:report/figures/deployment.png|Single-node deployment on the loopback interface|width=3.3]]

## Algorithms

The decision procedures are given as pseudocode that summarises the implementation.

### Emergency Fast-Path

The classifier (`medbridge/core/emergency_classifier.py`) is a deterministic engine of regular expressions and look-ups with no model or network. It runs on the raw message, because redaction placeholders could hide cues, and never logs it (Algorithm 3.1). Its 40 rules are 26 for English (two BP, 24 symptom) and 14 for Hinglish and Indian-English phrasing. It prefers recall: educational questions are suppressed only without a first-person, present-tense marker, historical mentions only with an explicit time anchor and no recency word, and side-effect questions naming a red flag escalate. Tests enforce an average below 5 ms and a median below 10 ms at the 2,000-character maximum.

```algorithm
Deterministic emergency fast-path: classify_detailed
Input: raw message m.  Output: ESCALATE or NONE, with matched, negated and suppressed rule ids
1  t <- NORMALISE(m)   # NFKC, lower case, letter runs, misspellings, one-edit typos, Hinglish
2  for each BP reading (sys, dia) in t that is not a date, dose, lab value, target or threshold:
3      fire bp_crisis if (BP wording and (sys >= 180 or dia >= 120)) or (sys >= 180 and dia >= 120)
4      fire bp_low_symptomatic if BP wording and sys < 80 and dizziness, faintness or confusion is affirmed
5  for each match of a red-flag rule (chest pain, stroke signs, syncope, breathlessness,
       severe headache, seizure, haemoptysis, angioedema, self-harm, pre-eclampsia, Hinglish)
       within its clause (split at . ; ? ! "but" "however" ...):
6      if NEGATED (no, denies, nahi) then mark negated
7      else if INFORMATIONAL, HISTORICAL or HEAVY_EXERTION then mark suppressed else fire
8  return ESCALATE if any rule fired else NONE   # negation never cancels another affirmed rule
```

### Context Gate and Soft-Ask Loop-Breaker

Gate 1 (`medbridge/ai/context_gate.py`) returns SOFT-ASK or PROCEED with a rationale and missing fields from twelve facts (current medications, BP reading, age, sex, pregnancy status, kidney function, heart rate, symptom details, diagnosis, dose, comorbidities, laboratory results). It asks only when a specific personal fact that would change the safe answer is missing, and lets general, out-of-domain, fabricated-drug and false-premise questions PROCEED to Gate 2 (Section 4.10 reports where it erred). The loop-breaker limits consecutive SOFT-ASK turns to two (Figure 3.4, Algorithm 3.2). After a SOFT-ASK, the gates and generator receive an *effective patient message* (the question followed by each follow-up), and a regenerated turn restarts from the counter recorded before the original turn.

[[figure:report/figures/fsm_soft_ask.png|State machine of the Context Gate and loop-breaker|width=3.6]]

```algorithm
Context Gate with loop-breaker (threshold T = 2)
Input: turn context with stored counter c = soft_ask_count before this turn
1  start speculative RETRIEVE in the background (Algorithm 3.3)
2  d <- GATE1_LLM(snapshot, patient message, intent); if d is None then d <- PROCEED (fallback)
3  if d = PROCEED then c <- 0; continue to Gate 2 with the retrieval result
4  else if c < T then action <- SOFT-ASK; c <- c + 1; discard the speculative retrieval;
       generate a question for d.missing_fields (no evidence, no clinical claims)
5  else action <- GENERALIZE (source = loop_breaker); c <- 0; skip Gate 2;
       generate labelled general guidance from the retrieved evidence
6  persist c with the turn and record its value before the turn in the trace
```

### Speculative Retrieval

Retrieval depends only on the extractor's query, so it starts concurrently with Gate 1 (flag `SPECULATIVE_RETRIEVAL`, on by default) while Gate 1 keeps full control over its use (Algorithm 3.3); the passages are identical in either order, so routing is unchanged, as unit tests check.

```algorithm
Speculative retrieval
1  task <- start RETRIEVE(search_query); [g_start, g_end] <- time window of GATE1_LLM
2  if Gate 1 = SOFT-ASK and the counter is below the threshold then
3      cancel task; record retrieval as "discarded"     # evidence is never used
4  else outcome <- await task; saved_ms <- overlap([outcome.start, outcome.end], [g_start, g_end])
5      use outcome.passages exactly as if retrieval had run after Gate 1
```

### Hybrid Retrieval with Reciprocal Rank Fusion and Reranking

Dense retrieval captures meaning ("tablets that make me pee more") [@karpukhin2020dpr] and BM25 exact drug names [@robertson2009bm25]. Dense vectors come from `BAAI/bge-small-en-v1.5` (384 dimensions) [@xiao2024cpack] and sparse vectors from `Qdrant/bm25`; Qdrant fuses the lists with RRF, *k* = 60 [@cormack2009rrf], and `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores the top 20 [@nogueira2019passage; @wang2020minilm] (Algorithm 3.4).

```algorithm
Hybrid retrieval with RRF (k = 60) and cross-encoder reranking
Input: search query q (extractor output plus clinical synonyms and generic drug names)
1  if (q, 20, "hybrid") is in the result cache and not expired then C <- the cached top-20 and go to step 5
2  (v_dense, v_sparse) <- encode q with both encoders concurrently (cached)
3  L_dense <- top-20 by cosine similarity; L_sparse <- top-20 by BM25 score
4  score_rrf(d) <- sum over lists L containing d of 1 / (60 + rank_L(d)); C <- top-20 by score_rrf
5  score_rerank(d) <- CrossEncoder(q, text(d)) for d in C   # worker thread, GPU if available
6  return the 5 best by score_rerank with metadata and both scores
   # non-empty results are cached; an infrastructure failure raises RetrievalError
```

### Evidence Gate Routing

Gate 2 (`medbridge/ai/evidence_gate.py`) receives the passages in `<clinical_evidence>`, the snapshot in `<patient_context>`, lexicon notes on named medicines in `<drug_reference>` and the patient's own words in `<untrusted_user_input>`, so that false premises and unknown names are judged as written. It applies the *first* rule that fits, with the rationale before the action in the schema (Algorithm 3.5). The few-shot examples of both gates use topics absent from the benchmark, which a test checks against every benchmark turn.

```algorithm
Evidence Gate (Gate 2) routing
Input: evidence E (top-5), snapshot S, patient message p
1  skip if the turn already ended in SOFT-ASK, the loop-breaker, an emergency or small talk
2  out <- GATE2_LLM(E, S, p)        # temperature 0.0, seed 42; first rule that fits:
   #  1. possible emergency happening now                                       -> ESCALATE
   #  2. outside cardiovascular care; evidence irrelevant or too thin; a named drug, brand,
   #     device or treatment that is neither listed in <drug_reference> nor present in E
   #     (a listed medicine is judged by its generic name and class); a dose or number E
   #     does not state; paediatric dosing or pregnancy decisions E does not cover -> ABSTAIN
   #  3. E covers the topic only generally, or contradicts a false belief       -> GENERALIZE
   #  4. E directly and specifically answers this question for this patient    -> ANSWER
   #  if torn between ANSWER and GENERALIZE, choose GENERALIZE
3  if out is None or invalid then out <- GENERALIZE (source = fallback)
4  record evidence features (max and mean rerank score, entity coverage); return out
```

### Grounded Generation and Citation Validation

The generator (`medbridge/ai/response_generator.py`, temperature 0.3) receives the passages as `<chunk index="1">` to `<chunk index="5">` and must use only their facts, end every clinical sentence with a marker such as `[2]`, correct false premises, never state numbers or drugs absent from the evidence and never advise changing a medicine. Single-digit markers were adopted because local models copied long identifiers unreliably (ADL-042), and server-side checking is needed because cited sources often fail to support their statements [@wu2025sourcecheckup; @wang2025medcite]. Algorithm 3.6 ensures that an answer cites only what it was shown.

```algorithm
Citation validation and ANSWER-to-GENERALIZE downgrade
Input: draft x, passages E = [e1..e5] in prompt order, action a
1  for each marker group in x ("[3]", "[1, 2]", "[1-3]"): keep n with 1 <= n <= |E|, strip others
2  renumber kept markers [1], [2], ... by first appearance
3  rebuild citations from passage metadata (ORG YEAR, section, page, URL, excerpt)  # never LLM text
4  if a = ANSWER and no valid citation remains and the Evidence Gate is active then
5      a <- GENERALIZE              # Gate 2's ANSWER remains visible in the audit log
6  return cleaned text, citations, a
```

### V1: Calibrated Evidence Gate

V1 (`medbridge/ai/evidence_calibration.py`) keeps the position and outputs of Gate 2 and adds permutation voting and a selective-prediction veto based on retrieval-evidence scores rather than the model's own confidence [@kadavath2022know; @geifman2017selective; @wen2025abstention]. At temperature 0.0, diversity comes from permuting the evidence order, and the evidence signal can only make the decision more conservative (Algorithm 3.7); thresholds are calibrated on the development split by replaying logged Gate 2 traces.

```algorithm
V1 calibrated Evidence Gate
Input: evidence E, snapshot S, message p, votes V = 3, thresholds r_answer, r_any, c_min
1  f.max_rerank <- max cross-encoder score over E (-99 if E is empty)
2  f.entity_coverage <- share of salient entities of p (drugs, numbers with units, clinical
       keywords; redaction placeholders ignored) found in E
3  votes <- GATE2_LLM on up to V orderings of E (original, reversed, rotations), concurrently
4  if every vote failed then return GENERALIZE (source = fallback)
5  a <- majority vote; ties -> more conservative (ESCALATE > ABSTAIN > GENERALIZE > ANSWER)
6  if a in {ANSWER, GENERALIZE} and f.max_rerank < r_any then a <- ABSTAIN
7  else if a = ANSWER and (f.max_rerank < r_answer or f.entity_coverage < c_min) then
8      a <- GENERALIZE                  # ESCALATE never overridden, ABSTAIN never upgraded
```

### V2: Post-Generation Grounding Verifier (Gate 3)

V2 (`medbridge/ai/grounding_verifier.py`) checks the draft against the evidence before the patient sees it [@gao2023rarr; @dhuliawala2024cove; @laban2022summac]. It skips non-claims (headings, greetings, referral and disclaimer clauses) and scores each claim with a local NLI cross-encoder, `cross-encoder/nli-MiniLM2-L6-H768`, not the evaluation's NLI judge (Algorithm 3.8). V2 can only remove content or lower the action; its thresholds are calibrated on the development split against the LLM judge's sentence verdicts. Since commit `16e594b` a draft whose every claim is unsupported (u = 1) is withdrawn as in step 5.

```algorithm
V2 grounding verifier (Gate 3)
Input: draft x with action a in {ANSWER, GENERALIZE}, passages E, thresholds theta, tau
1  for each clinical claim s in x: P <- 3-sentence windows of the passages s cites (all if none)
2      supported(s) <- max over w in P of NLI_entailment(w, claim(s)) >= theta
3  u <- unsupported claims / claims
4  if u <= tau and u < 1 then remove unsupported sentences, re-validate citations, keep a
5  else if a = ANSWER then regenerate as GENERALIZE and verify again;
       a <- GENERALIZE with the cleaned text if its u <= tau and u < 1, otherwise ABSTAIN (template)
6  else a <- ABSTAIN (template)
7  if the NLI model is unavailable, keep x and record the failure
```

### Natural-Language Robustness Helpers

Deterministic helpers bridge patient and guideline language: drug normalisation (`medbridge/nlp/drugs.py`) maps 189 brands (124 international, 65 Indian, such as Telma and Ecosprin) to generics with fuzzy matching at 0.88 [@gallifant2024rabbits] and, since the brand-name fix, supplies the `<drug_reference>` notes and a current-medication backstop (Section 4.11); query expansion adds clinical synonyms ("water pill" to thiazide diuretic); BP parsing records a reading only when the text ties it to the patient; and greetings get a claim-free template.

## Data Model and Persistence

Patient context is event-sourced [@fowler2005eventsourcing]. The extractor emits typed *delta events* (BP_READING, MEDICATION_ADDED, MEDICATION_STOPPED, SYMPTOM_REPORTED, DEMOGRAPHIC, LAB_RESULT) and never writes the state; the deterministic projector folds them into a snapshot (a repeated MEDICATION_ADDED is a dose change with history), so the snapshot is always a reproducible function of the events (Figure 3.5, Table 3.4).

[[figure:report/figures/er_diagram.png|Entity-relationship diagram of the PostgreSQL schema|width=3.8]]

Table: Database tables, key columns and protection {size=9}
| Table | Purpose and key columns | Mutability | Encrypted content |
|---|---|---|---|
| `sessions` | Conversation: `created_at`, `last_active_at`, `erased_at`, `legacy_plaintext_rows` | Never deleted | None |
| `clinical_events` | Typed delta events: `event_type` (six values), `payload` | Append-only (rules) | Payload |
| `context_snapshots` | Projected `snapshot` and `soft_ask_count` | Updated under a row lock | Snapshot |
| `audit_logs` | One row per turn: request, `gate_1_action` and `gate_2_action` with rationales, `final_action`, evidence ids, response, latency, `trace` | Append-only (rules) | Request, rationales, response, private trace |
| `message_history` | User and assistant messages with action and citations | Deleted on erasure | Text |
| `session_keys` | `wrapped_key`: wrapped per-session data key | Deleted on erasure | Key is wrapped |
| `feedback` | One rating and comment per assistant message | Deleted on erasure | Comment |

PostgreSQL rules (`DO INSTEAD NOTHING`) turn every UPDATE and DELETE on `clinical_events` and `audit_logs` into a no-op, so immutability holds even if application code is wrong. The turn store (`medbridge/state/turn_store.py`) writes a whole turn with one flush instead of five or more, and session and row locks (`SELECT … FOR UPDATE`) keep the stored snapshot equal to the fold of the stored events (Figure 3.6).

[[figure:report/figures/dataflow.png|Data flow of a patient turn (top) and the knowledge-base build (bottom)|width=4.0]]

## Privacy and Security Design

MedBridge applies layered controls (Figure 3.7) that follow the minimisation, purpose, storage-limitation and security principles of the DPDP Act [@dpdp2023india] and the HIPAA Privacy Rule [@hhs_hipaa_privacy_rule], without claiming legal compliance.

[[figure:report/figures/privacy_flow.png|Privacy controls along the request path|width=4.0]]

**Redaction and local-only inference.** The first stage replaces identifiers with typed placeholders before any model call, retrieval, log line or write (`medbridge/privacy/redaction.py`): names after cues ("my name is", greetings, sign-offs, titles), phone numbers, e-mail and postal addresses, PIN and ZIP codes, Aadhaar, PAN and Social Security numbers, record and insurance numbers, URLs and IP addresses. A date of birth becomes an age placeholder such as `[AGE: 54]` [@hhs2012deidentification]; BP readings, laboratory values, drug names and doses are never altered. A rule-based detector was preferred because it needs no model, is fast and can be audited (recall in Section 4.6), and it fails closed. With `LOCAL_ONLY_MODE=true` (the default) the LLM wrapper accepts only a loopback Ollama endpoint serving a local model, so no patient text leaves the machine.

**Encryption and crypto-shredding erasure.** Envelope encryption (`medbridge/privacy/crypto.py`) uses a master key from the environment to wrap one random data key per session [@barker2020sp80057pt1r5], which encrypts all stored text, payloads, snapshots, comments and the private trace with Fernet (AES-128-CBC with HMAC-SHA256). Destroying the key makes every ciphertext of the session unreadable, the cryptographic erase of media-sanitisation guidance [@nist2025sp80088r2], while the append-only rows stay unchanged (Algorithm 3.9). A retention job applies the same procedure hourly to sessions inactive for longer than `DATA_RETENTION_DAYS` (30 by default); rows written before encryption existed are counted, flagged and hidden.

```algorithm
Right to erasure by crypto-shredding: DELETE /api/sessions/{id}
Input: session id s, bearer token for s (accepted up to max(90 days, DATA_RETENTION_DAYS) after expiry)
1  verify token signature and session binding; otherwise 401/403
2  in one transaction: count append-only rows of s still in legacy plaintext;
       delete the row of s in session_keys                 # the data key is destroyed
       delete rows of s in message_history, context_snapshots, feedback; set sessions.erased_at
3  invalidate the in-process key cache for s               # events and audit rows untouched
4  return 204; later requests for s return 410 before any pipeline work
```

**Access control and injection defence.** Every session route requires a signed, expiring bearer token (HMAC-SHA256 over a payload bound to one session, compared in constant time; `medbridge/privacy/tokens.py`), answering 401 when it is missing and 403 when it is invalid. The dashboard needs an administrator login; middleware adds a strict Content Security Policy, security headers, restricted CORS, per-IP rate limits and a request-size limit, and logs are scrubbed. Against prompt injection [@owasp2025llmtop10; @greshake2023notwhat; @liu2024formalizing], patient-derived text goes into `<untrusted_user_input>` blocks declared to be data, with tag-like sequences, look-alike brackets and invisible characters neutralised (`medbridge/ai/prompt_utils.py`), and the most safety-critical decisions do not depend on the generator.

## Experimental Design

The hallucination study (Figure 3.8) tests RQ1 to RQ3 on a synthetic benchmark with paired conditions.

[[figure:report/figures/evaluation_design.png|Design of the hallucination study|width=3.2]]

### Benchmark: MedBridge-HalluBench

MedBridge-HalluBench v1 (`evaluation/datasets/hallubench_v1.json`) has 247 synthetic items with 298 user turns in natural, often casual phrasing. Each item has a category, subcategory, style, one to three turns with gold and acceptable actions, gold evidence and notes naming every fabricated entity and false premise. The six categories mirror the routing (Table 3.5): SOFT-ASK and loop-breaker items test Gate 1, ANSWER, GENERALIZE and ABSTAIN distinctions Gate 2, and emergency items the fast-path.

[[table:hallubench_composition]]

The styles are casual English (155 items), Indian English (35), typo-laden (33) and clinical (24); 43 items are multi-turn. Gold evidence was labelled by regular-expression search, not by the retriever. A stratified split (seed 42) gives a development split of 49 items (59 turns) for model selection and calibration and a test split of 198 items (239 turns) for all reported results; 14 items close to the original prompt examples are tagged `prompt_overlap`. Acceptable actions are lenient only where neighbouring actions are equally safe.

### Conditions

Conditions are internal presets (`AblationConfig` in `medbridge/core/pipeline_types.py`) of the production orchestrator that share the model, the retrieval index and the gate prompts; C3 to C5 and the variants also share the production generator prompt, so they differ in pipeline structure alone, while C0 to C2 differ in the generator prompt by design (Table 3.6).

Table: Experimental conditions of the hallucination study {size=8.5}
| Condition | Retrieval | Gate 1 | Gate 2 | Generator prompt mode | Fast-path |
|---|---|---|---|---|---|
| C0 Direct LLM | No | No | No | `direct`: model knowledge, no evidence | Off |
| C1 Vanilla RAG | Yes | No | No | `vanilla`: answer from passages with `[n]`, no safety rules | On |
| C2 RAG + prompt safety | Yes | No | No | `prompt_safety`: vanilla plus a prose safety block | On |
| C1G RAG + production prompt (reference) | Yes | No | No | `grounded` | On |
| C3 Context Gate only | Yes | Yes | No (always ANSWER) | `grounded` (production) | On |
| C4 Evidence Gate only | Yes | No (always PROCEED) | Yes | `grounded` | On |
| C5 MedBridge (dual gate) | Yes | Yes | Yes | `grounded` | On |
| V1 Calibrated Gate 2 | Yes | Yes | V1 | `grounded` | On |
| V2 Triple gate | Yes | Yes | Yes, plus Gate 3 | `grounded` | On |
| V12 (optional) | Yes | Yes | V1, plus Gate 3 | `grounded` | On |

The C2 block asks in prose for what the gates do, so C2 against C1 measures prompt-only safety and C5 against C2 what the gates add; C1G isolates the gates from the prompt. C5 against C0 to C2 answers RQ1, C3 and C4 against C5 RQ2, and the variants against C5 RQ3. Each item runs in a fresh session; the unit of analysis is the *item-turn*.

### Judges, Metrics and Statistics

The **LLM judge**, `gemma3:12b` [@gemmateam2025gemma3], comes from a different family than the application model to avoid self-preference [@zheng2023judging; @panickssery2024llm]. At temperature 0.0 it splits each response into atomic clinical claims [@min2023factscore], labels each SUPPORTED, UNSUPPORTED or CONTRADICTED under a strict rubric against the three passages a cross-encoder ranks highest among the condition's passages and three retrieved per claim (so C0 is still judged against evidence), and classifies the behaviour, flagging accepted false premises, fabricated entities and unsupported doses. It never sees the condition; templates get zero claims. The **NLI judge**, `cross-encoder/nli-deberta-v3-base` [@he2023debertav3; @williams2018mnli], scores the same claims over three-sentence windows against thresholds θ and θ_c. A blinded **annotation sheet** of 50 stratified test items holds the C5 and C1 claims for two human annotators and was labelled by an AI annotator (Appendix 4, Section 4.11); agreement uses Cohen's κ [@cohen1960coefficient; @landis1977measurement]. Table 3.7 defines the metrics as implemented in `evaluation/analyze.py`.

Table: Metrics of the hallucination study {size=9}
| Metric | Definition |
|---|---|
| Hallucination rate | Responses with at least one UNSUPPORTED or CONTRADICTED claim / all responses, per judge |
| Unsupported-claim ratio | (UNSUPPORTED + CONTRADICTED) / all claims, pooled |
| Unsafe-answer rate | Unanswerable items: answered specifically, any behaviour flag, or an unsupported claim |
| Appropriate abstention | Unanswerable items: behaviour acceptable and not unsafe |
| Coverage / over-abstention | Answerable turns: share answered / share abstained |
| Risk-coverage | Hallucination among answered against coverage, ranked by maximum rerank score [@geifman2017selective] |
| Routing and emergency | Strict and lenient accuracy, per-class F1, macro-F1; emergency recall and false escalation |
| Citation quality | Marker validity; supported citation precision (both judges) |
| Latency; degraded turns | Raw and cache-imputed p50 and p95; degraded count and reasons |

Proportions carry Wilson 95 per cent intervals [@wilson1927probable] and the claim ratio a cluster bootstrap with 10,000 resamples [@efron1993bootstrap]. Conditions are compared on the same item-turns with exact two-sided McNemar tests [@mcnemar1947note] (C5 against C0 to C4; V1, V2 and V12 against C5) with Holm adjustment within each family, outcome and judge [@holm1979simple], and Cochran's Q across C0 to C5 [@cochran1950comparison]; effect sizes are the paired risk difference, the conditional odds ratio and the relative risk reduction. Sensitivity analyses vary degraded turns, `prompt_overlap` items, the boilerplate filter and the unit of analysis. The decision rule is fixed in advance: if C5 does not significantly reduce (Holm-adjusted *p* < 0.05) the hallucination or unsafe-answer rate against C1 and C2, the result is reported as it is with an error analysis, and V1 and V2 are evaluated under the same protocol in any case.

### Calibration, Degraded Turns and Reproducibility

Calibration uses the development split only, in an enforced order: C5 is run and judged on it, `evaluation.calibrate` writes `calibration.json`, and V1, V2 and V12 are refused on the test split without it. θ maximises κ between the NLI and LLM judges (θ_c then three-class κ); V1 thresholds minimise false ANSWERs on unanswerable turns with coverage at least C5's minus 0.05; and V2's θ maximises sentence-level κ with the LLM judge. τ was to minimise hallucination while keeping 85 per cent of C5's development coverage, but no τ met that floor ({{calibration.v2.dev_result.coverage|pct}} against {{calibration.v2.coverage_floor|pct}} even at τ = 1), so the calibration fell back to τ = 1, which strips unsupported sentences but, in the studied code, never withdraws an answer. A turn with any fallback, failed stage, failed call or runner error is *degraded* (`evaluation/degraded.py`): it is re-run, excluded from headline metrics and kept for a sensitivity analysis, and headline results are refused above 5 per cent degraded turns per condition. Identical model requests are memoised by a hash of the full request, so paired conditions differ only in structure, and each run writes a manifest of its commit, dataset checksum, model digests, seeds and settings.

### Supporting Experiments and Datasets

Table 3.8 lists the component experiments of Chapter 4; all datasets are synthetic or derived from public documents.

Table: Supporting experiments and datasets {size=9}
| Experiment | Dataset (location, size) | Measures |
|---|---|---|
| Retrieval (`evaluation/retrieval_eval.py`) | `retrieval_eval_v1.json`: 70 queries (26 lay, 44 clinical, 13 topics), 114 relevant spans | Recall@5, Recall@20, MRR, nDCG@5 per mode; ablation of four rerankers |
| Model selection (`evaluation/model_selection.py`) | HalluBench development split (49 items, 59 turns), condition C5 | Routing accuracy, macro-F1, JSON validity, latency, throughput |
| PII redaction (`scripts/pii_eval.py`) | `tests/privacy/`: development set 240 messages, 183 spans; blind set 237 messages, 203 spans | Span-level precision, recall and F1; clinical-value protection |
| Emergency classifier | `tests/data/emergency_cases*.json`: 288 English, 141 Hinglish messages | Confusion counts |
| MedBridge-AQ (`tests/benchmark/`) | 275 vignettes (93 ANSWER, 50 SOFT-ASK, 43 GENERALIZE, 42 ABSTAIN, 47 ESCALATE) | Live routing accuracy, citation precision, latency |
| RABBITS (`tests/adversarial/`) | 60 of the 189 brand–generic pairs (30 Indian) × 2 templates [@gallifant2024rabbits] | Action consistency, stored-generic agreement, before and after the fix |
| Adversarial security (`tests/adversarial/`) | 19 injection and jailbreak payloads; hermetic tests | Deterministic pass/fail |
| Performance, load, smoke (`scripts/`) | 5 latency scenarios (6 turns); 1, 5 and 10 concurrent users; 16 API checks | Cold and warm latency before/after; throughput, errors; pass/fail |

## Tools, Models and Knowledge Base

MedBridge runs on Python 3.11 with FastAPI 0.143.0, Pydantic 2.14.0, SQLAlchemy 2.1.4, Qdrant (client 1.19.1), fastembed 0.9.0, sentence-transformers 6.1.0, cryptography 50.0.2 and PyMuPDF 1.28.2, Ollama 0.35.0 and PostgreSQL 16 (`postgres:16-alpine`). All language models are open-weight and served by Ollama at 4-bit quantisation (Q4_K_M); the application model should fit a 6 GB GPU (not measured), while the 8.2 GB judge `gemma3:12b` runs only on the evaluation machine (Table 3.9). Experiments ran on an Apple MacBook (M5, 24 GB); the target Windows 11 laptop with an RTX 4050 (6 GB) was not measured.

Table: Models used by MedBridge and its evaluation {size=9}
| Role | Model | Notes |
|---|---|---|
| Application LLM (extractor, gates, generator) | `qwen3.5:4b` (4.2B parameters, Q4_K_M, 3.3 GB), chosen by the model-selection run (Section 4.5) | Chosen over `qwen3:8b` [@yang2025qwen3] and `gemma3:4b` [@gemmateam2025gemma3] |
| Dense embedder; sparse encoder | `BAAI/bge-small-en-v1.5`; `Qdrant/bm25` | 384 dimensions [@xiao2024cpack]; BM25 weights [@robertson2009bm25] |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Also the V1 evidence signal [@wang2020minilm] |
| Grounding verifier (V2) | `cross-encoder/nli-MiniLM2-L6-H768` | Different from the NLI judge |
| LLM judge; NLI judge | `gemma3:12b`; `cross-encoder/nli-deberta-v3-base` | Different family from the application model; NLI thresholds calibrated on the development split |

The knowledge base grew from 3 PDF guidelines (905 chunks) at Review 3 to {{corpus.totals.documents|int}} curated, publicly readable documents from {{corpus_orgs.split|int}} issuing organisations (joint labels split; Table 3.10): guidelines of AHA/ACC, ESC, ISH, WHO, KDIGO and NICE [@jones2025aha; @mcevoy2024esc; @nice2019ng136; @who2021pharm; @unger2020ish; @kdigo2021bp; @joglar2024af; @arnett2019primary], scientific statements, a review of the 2023 chronic coronary disease guideline, and MedlinePlus and NHLBI patient pages. A document is indexed only if `data/corpus_manifest.json` records its organisation, year, URL, licence note and SHA-256 checksum; several major guidelines could not be obtained (see the limitations in Chapter 4). The WHO, KDIGO, MedlinePlus and NHLBI documents and the review carry open or public-domain licences; the others are free to read but not redistributable, so they are indexed locally for non-commercial academic use and never committed. Two licences need attention before any wider use: the 2024 ESC guideline [@mcevoy2024esc] reserves text-and-data-mining and AI-training rights to ESC and Oxford University Press (EU Directive 2019/790, Art. 4(3)) and forbids storage in a retrieval system without permission (it was kept from the original corpus and can be pruned from the index), and the 2020 ISH guideline [@unger2020ish] (© American Heart Association, not redistributable) was taken from a co-author's institutional repository, with its indexing terms unconfirmed.

[[table:corpus]]

Documents are cleaned and split into chunks of at most 400 tokens with a 60-token overlap, measured with the embedder's tokenizer so that nothing exceeds the 512-token encoder window; tables become separate chunks. The index holds {{corpus.totals.chunks|comma}} chunks ({{corpus.totals.table_chunks|comma}} tables) from {{corpus.totals.pdf_documents|int}} PDF and {{corpus.totals.html_documents|int}} HTML documents, with a median of {{corpus.totals.chunk_tokens.median|int}} tokens and {{corpus.totals.chunk_tokens.over_512|int}} chunks over 512. Heart failure ({{corpus.topics.heart_failure.documents|int}} documents, {{corpus.topics.heart_failure.chunks|int}} chunks) and coronary disease ({{corpus.topics.coronary_disease.documents|int}}, {{corpus.topics.coronary_disease.chunks|int}}) are the thinnest topics. Ingestion is incremental by checksum (full rebuild {{corpus.last_full_build.seconds|int}} s).
