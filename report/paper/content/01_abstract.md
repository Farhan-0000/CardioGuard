<!--
Abstract and index terms. IEEE asks for a single paragraph of at most about 250 words,
with no citations. The last sentence is filled in once evaluation/results/summary.json
exists (python -m evaluation.analyze). Do not type any study number by hand: use
{{study...}} values (see report/README.md, "Inline values") or copy from summary.json.
-->

#* Abstract

Large language models explain clinical guidelines fluently, but in patient-facing cardiovascular care a fluent answer to a question the evidence does not cover is itself a safety risk, and a standard retrieval-augmented generation (RAG) pipeline answers every question it receives. We present MedBridge, a locally deployable assistant for hypertension and related cardiovascular care that makes answerability an explicit, auditable decision. A deterministic emergency fast-path screens each message before any model call. A Context Gate decides before retrieval whether missing personal facts call for a clarifying question, and an Evidence Gate decides after hybrid retrieval whether to answer, give general information, abstain or escalate. Abstentions and escalations are pre-vetted templates, so no generated text reaches the user on those paths. Two optional extensions can only make the system more conservative: a calibrated Evidence Gate with permutation voting, and a post-generation grounding verifier based on natural language inference. We introduce MedBridge-HalluBench, a synthetic benchmark of {{hallubench.n_items}} conversations ({{hallubench.n_turns}} turns) in six answerability categories, and compare the full system with direct-LLM, vanilla-RAG, prompt-only-safety and single-gate ablations, using a 4B-parameter open-weight model, an LLM judge and an NLI judge, and paired exact tests. [TO BE FILLED after the study: one or two sentences with the headline hallucination and unsafe-answer results, their confidence intervals and Holm-adjusted p-values, taken from evaluation/results/summary.json.]

#* Index Terms

Retrieval-augmented generation, hallucination, abstention, clinical decision support, hypertension, large language models, patient safety
