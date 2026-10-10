# The MedBridge Method {#sec:method}

MedBridge answers patient questions about hypertension and related cardiovascular topics (heart failure, atrial fibrillation, coronary disease, lipids, kidney- and diabetes-related blood pressure, pregnancy, lifestyle). It gives guideline-based information and never diagnoses or prescribes. Each turn passes through the pipeline of [@fig:pipeline] ([@alg:routing]) and ends in one of five actions: ANSWER (a specific, cited answer), SOFT-ASK (a clarifying question), GENERALIZE (general guideline information, marked as such), ABSTAIN (a pre-vetted refusal) or ESCALATE (a pre-vetted emergency message).

[[figure:report/paper/figures/gate_pipeline.png|The MedBridge routing pipeline. Gates 1 and 2 are separate LLM decisions at temperature 0; emergency handling, refusals and escalations involve no generated text. Gate 3 is an optional extension (V2). {#fig:pipeline}|width=3.3]]

**Design principles.** (i) Each answerability decision and its rationale go to an append-only audit log. (ii) Emergency triage, refusal and escalation texts, and the patient state are deterministic. (iii) LLM decisions are narrow classifications with JSON-schema outputs at temperature 0 and seed 42. (iv) Failures degrade safely: a failed extraction leaves the state unchanged, a failed Gate 1 defaults to PROCEED, a failed Gate 2 to GENERALIZE, and a failed generator to a pre-vetted template.

## Emergency Fast-Path and Patient State

Before any model call, a rule-based classifier reads the raw message. It parses blood-pressure readings in many written forms and escalates at a systolic pressure of at least 180 mmHg or a diastolic pressure of at least 120 mmHg. The 2025 AHA/ACC guideline manages such severe hypertension as an outpatient matter when there is no acute organ damage [@jones2025aha]; a patient-facing tool cannot rule that out, so MedBridge deliberately escalates in this range. Further rules cover red-flag symptoms (chest pain, stroke signs, fainting, severe breathlessness), with clause-level filters for negation, informational questions and dated history, and normalisation of misspellings and Hinglish. Since nothing later can overrule it, this path favours recall over precision. For other messages, an extractor LLM converts the redacted text into typed delta events (reading, medication added or stopped, symptom, demographic, laboratory result), a search query and an intent summary. The LLM never writes the patient state: the snapshot is a deterministic fold over stored events, so one faulty extraction cannot overwrite earlier facts. Deterministic helpers map drug brands to generic names and expand lay terms for retrieval.

## Gate 1: Context Gate

Gate 1 sees the snapshot and the message and returns PROCEED or SOFT-ASK with missing fields from a fixed vocabulary (current medicines, a recent reading, age, pregnancy status, ...). It may ask only when the question concerns the patient's own cardiovascular care, a specific personal fact would change the safe answer, and that fact is unknown. General, out-of-scope, fabricated, false-premise and otherwise unanswerable questions always proceed, because more context cannot make them answerable; Gate 2 handles them with the evidence. On SOFT-ASK the generator phrases a short question. A loop-breaker bounds the dialogue: if Gate 1 asks again after two consecutive clarifying questions, the turn is forced to GENERALIZE and Gate 2 is skipped.

## Hybrid Retrieval

The knowledge base holds {{corpus.totals.documents|int}} open-access documents from {{corpus.totals.organisations|int}} organisations, mainly clinical guidelines such as [@jones2025aha], in {{corpus.totals.chunks|comma}} chunks of at most {{corpus.chunking.chunk_size|int}} tokens. Each query is encoded by a dense model (bge-small-en-v1.5 [@xiao2024cpack]) and by BM25; the two ranked lists are fused by reciprocal rank fusion with k = 60 [@cormack2009rrf], and the top 20 candidates are reranked by a MiniLM cross-encoder [@nogueira2019passage] to the top 5. Retrieval depends only on the extractor's query, so it starts concurrently with Gate 1 and is discarded on SOFT-ASK, which leaves routing unchanged. On {{retrieval.modes.hybrid.n}} labelled queries, fusion gave the highest mean reciprocal rank ({{retrieval.modes.hybrid.mrr|cif}} against {{retrieval.modes.dense.mrr|cif}} for dense retrieval, 95% bootstrap intervals), while reranking brought no measurable gain (nDCG@5 {{retrieval.modes.hybrid+rerank.ndcg@5|cif}} with, {{retrieval.modes.hybrid.ndcg@5|cif}} without). We keep it because it is cheap and feeds V1.

## Gate 2: Evidence Gate

Gate 2 sees the five chunks, the snapshot and the patient's own words and applies the first matching rule. ESCALATE: an emergency happening now that the fast-path missed. ABSTAIN: a question outside cardiovascular care; irrelevant or thin evidence; a named drug or product absent from the evidence (it may not exist); a dose or number the evidence does not state; paediatric or pregnancy medicine decisions the evidence does not cover. GENERALIZE: evidence that covers the topic only in general terms, or a false premise the evidence contradicts (the answer corrects it). ANSWER: evidence that addresses this question directly for this patient; when torn between ANSWER and GENERALIZE, the gate prefers GENERALIZE. A unit test checks that its few-shot examples share no topic or invented drug name with the benchmark.

```algorithm
Routing of one patient turn in MedBridge {#alg:routing}
input: raw message m; state S; consecutive SOFT-ASK count c
 1  r <- Redact(m)
 2  if EmergencyRules(m) fires: return Template(ESCALATE)
 3  (ev, q, intent) <- Extract(r, S);  S' <- Fold(S, ev)
 4  R <- start Retrieve(q)            # concurrent with Gate 1
 5  g <- ContextGate(S', r, intent)   # T = 0, seed 42
 6  if g = SOFT-ASK and c < 2:
 7      discard R; c <- c + 1; return Ask(g.missing)
 8  E <- top-5 Rerank(await R)        # dense+BM25, RRF k = 60
 9  if g = SOFT-ASK: a <- GENERALIZE; c <- 0   # loop-breaker
10  else: c <- 0; a <- EvidenceGate(E, S', r)  # T = 0
11  if a in {ABSTAIN, ESCALATE}: return Template(a)
12  d <- Generate(a, E, S', r)        # T = 0.3, [n] citations
13  if V2 enabled: (a, d) <- GroundingVerifier(a, d, E)
14  d <- ValidateCitations(d, E)
15  if a = ANSWER and d has no valid citation: a <- GENERALIZE
16  Persist(ev, turn, gate rationales) atomically; return (a, d)
```

## Generation and Output Control

For ANSWER and GENERALIZE a grounded generator (temperature 0.3) writes from the evidence only; every sentence stating a clinical fact must end with the index of its supporting chunk, and GENERALIZE answers must say that they give general information. A server-side check removes markers that point to no retrieved chunk, and an ANSWER left without a valid citation is downgraded to GENERALIZE. ABSTAIN and ESCALATE never reach the generator. In every prompt the patient's text sits in an `<untrusted_user_input>` element with tag-like sequences escaped, and the evidence in `<clinical_evidence>`, as a defence against prompt injection.

## Optional Extensions V1 and V2

Both variants keep the routing and can only make it more conservative. **V1 (calibrated Evidence Gate)** runs Gate 2 on three deterministic permutations of the evidence order, still at temperature 0, and takes the majority action, breaking ties towards the more conservative one. Thresholds on the top reranker score and on the share of the question's salient entities (drugs, numbers, clinical terms) found in the evidence can then lower ANSWER to GENERALIZE, or either to ABSTAIN, never the reverse. **V2 ("Gate 3")** checks each claim sentence of a draft with an NLI cross-encoder against the chunks it cites. Unsupported sentences are removed while their share stays within a threshold τ; above it, an ANSWER is regenerated as GENERALIZE and checked again, otherwise the ABSTAIN template is returned. Its NLI model differs from the evaluation's NLI judge. Thresholds of both variants are calibrated on the development split only.

**Implementation.** A FastAPI service keeps events and audit records in append-only PostgreSQL tables and retrieves from Qdrant; all models run locally through Ollama. Identifiers are redacted before any model call or write, and session data are encrypted with per-session keys whose destruction erases a session.
