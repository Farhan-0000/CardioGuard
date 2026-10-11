# The MedBridge Method {#sec:method}

MedBridge answers patient questions on hypertension and cardiovascular care from guidelines and never diagnoses or prescribes. Each turn passes through the pipeline of [@fig:pipeline] and ends in one of five actions: ANSWER (a specific, cited answer), SOFT-ASK (a clarifying question), GENERALIZE (general guideline information, marked as such), ABSTAIN (a pre-vetted refusal) or ESCALATE (an emergency template).

[[figure:report/paper/figures/gate_pipeline.png|The MedBridge routing pipeline. Gates 1 and 2 are LLM decisions at temperature 0; emergency, refusal and escalation outputs contain no generated text. Gate 3 (V2 in the study) has been on by default since the study. {#fig:pipeline}|width=3.0]]

**Design principles.** Every answerability decision and its rationale go to an append-only audit log; emergency triage, refusal and escalation texts and the patient state are deterministic; LLM decisions are narrow JSON-schema classifications at temperature 0 (seed 42); and failures degrade safely (Gate 1 to PROCEED, Gate 2 to GENERALIZE, the generator to a template).

## Emergency Fast-Path and Patient State

Before any model call, a rule-based classifier parses blood-pressure readings in many written forms and escalates at a systolic pressure of at least 180 or a diastolic pressure of at least 120 mmHg (stricter than the 2025 AHA/ACC guideline [@jones2025aha], as a patient-facing tool cannot rule out organ damage) and on red-flag symptoms (chest pain, stroke signs, fainting, severe breathlessness), with filters for negation, informational questions and dated history; since nothing later can overrule it, this path favours recall. Otherwise an extractor LLM turns the redacted text into typed delta events (reading, medication, symptom, demographic, laboratory result) and a search query; the snapshot is a deterministic fold over stored events, so one faulty extraction cannot overwrite earlier facts.

## Gate 1: Context Gate

Gate 1 sees the snapshot and the message and returns PROCEED or SOFT-ASK with missing fields from a fixed vocabulary (current medicines, a recent reading, age, ...). It may ask only when the question concerns the patient's own cardiovascular care and an unknown personal fact would change the safe answer; general, out-of-scope, fabricated and false-premise questions proceed, as more context cannot make them answerable. On SOFT-ASK the generator phrases a short question; a third consecutive request forces GENERALIZE and skips Gate 2.

## Hybrid Retrieval

The knowledge base holds {{corpus.totals.documents|int}} freely accessible documents from 10 organisations, mainly clinical guidelines such as [@jones2025aha], in {{corpus.totals.chunks|comma}} chunks of at most {{corpus.chunking.chunk_size|int}} tokens. Dense (bge-small-en-v1.5 [@xiao2024cpack]) and BM25 rankings are fused by reciprocal rank fusion with k = 60 [@cormack2009rrf], and a MiniLM cross-encoder reranks the top 20 candidates to the top 5; retrieval runs concurrently with Gate 1. On {{retrieval.modes.hybrid.n}} labelled queries, fusion had the highest mean reciprocal rank ({{retrieval.modes.hybrid.mrr|cif}} against {{retrieval.modes.dense.mrr|cif}} for dense retrieval; overlapping intervals, not tested), while reranking did not help and slightly lowered both point estimates (nDCG@5 {{retrieval.modes.hybrid+rerank.ndcg@5|cif}} with, {{retrieval.modes.hybrid.ndcg@5|cif}} without); we keep it because it is cheap (about 0.2–0.3 s per warm turn) and feeds V1.
<!-- "10 organisations" is typed by hand: reports/corpus_stats.json totals.organisations (11)
     counts distinct org strings, and "AHA/ACC" and "ACC/AHA" name the same joint issuers
     (the deck computes 10 by splitting joint labels: report/presentation/deck_data.py).
     Use {{corpus.totals.organisations|int}} again once medbridge/ingestion/stats.py
     normalises the names and the stats file is regenerated. -->

## Gate 2: Evidence Gate

Gate 2 sees the five chunks, the snapshot and the patient's own words and applies the first matching rule. ESCALATE: an emergency happening now that the fast-path missed. ABSTAIN: a question outside cardiovascular care; irrelevant or thin evidence; a named drug or product absent from the evidence; a dose or number the evidence does not state; paediatric or pregnancy medicine decisions the evidence does not cover. GENERALIZE: evidence covering the topic only in general terms, or a false premise the evidence contradicts (the answer corrects it). ANSWER: evidence that addresses this question directly for this patient; when torn, the gate prefers GENERALIZE. A unit test keeps benchmark names and turns out of the few-shot examples.

## Generation and Output Control

For ANSWER and GENERALIZE a grounded generator (temperature 0.3) writes from the evidence only and ends every clinical-fact sentence with the index of its supporting chunk. A server-side check removes markers that point to no retrieved chunk and downgrades an ANSWER without a valid citation to GENERALIZE. Against prompt injection, the patient's text sits in an `<untrusted_user_input>` element with tag-like sequences escaped, and the evidence in `<clinical_evidence>`.

## Extensions V1 and V2

**V1 (calibrated Evidence Gate)** runs Gate 2 on three deterministic permutations of the evidence order and takes the majority action, breaking ties towards the more conservative one; thresholds on the top reranker score and on the share of the question's salient entities found in the evidence can then lower ANSWER to GENERALIZE, or either to ABSTAIN.

**V2 ("Gate 3")** checks each claim sentence of a draft with an NLI cross-encoder against the chunks it cites (all retrieved chunks when it cites none) and can only make an output more conservative: unsupported sentences are removed while their share stays within a threshold τ; above it, or (since commit 16e594b, after the study) when every claim is unsupported, an ANSWER is regenerated as GENERALIZE and checked again, and otherwise the ABSTAIN template is returned; a failed verifier delivers the unchecked draft. Its NLI model differs from the NLI judge's, but its thresholds are tuned against the LLM judge's verdicts. Development-split calibration left only V1's reranker threshold for ANSWER active and set τ = {{calibration.v2.thresholds.GROUNDING_MAX_UNSUPPORTED_FRACTION|f1}} for V2 (a fallback: no τ kept 85% of C5's coverage), so in the study V2 only deleted unsupported sentences, and a draft that deletion would have left empty was delivered unchanged.
