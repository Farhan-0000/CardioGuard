#* Abstract

Hypertension affects roughly one in three adults worldwide, and many patients now turn to conversational AI for help with readings, medicines and lifestyle between clinic visits. Large language models answer fluently, but they can invent clinical facts, apply general advice to a patient whose missing details would change it, and miss emergencies. Retrieval-augmented generation reduces fabrication, yet it does not decide *whether* it is safe to answer at all.

This report presents **MedBridge** (presented as *CardioGuard*), a safety-first, evidence-grounded consultation assistant for cardiovascular care centred on hypertension. Its core contribution is a **Dual Answerability Gate** that makes two separate decisions before any answer is generated. A *Context Gate* before retrieval asks whether enough is known about the patient, either requesting the missing facts (SOFT-ASK, at most twice in a row) or proceeding. An *Evidence Gate* after retrieval asks whether the retrieved guideline passages suffice and routes the query to ANSWER, GENERALIZE, ABSTAIN or ESCALATE. A deterministic emergency fast path runs before any model call, ABSTAIN and ESCALATE use pre-vetted templates, and inline citations are validated against the retrieved passages.

MedBridge combines hybrid dense and BM25 retrieval, reciprocal rank fusion and cross-encoder reranking over {{corpus.totals.documents|int}} curated documents ({{corpus.totals.chunks|int}} indexed passages), with all inference on locally hosted open-weight models. Patient text is redacted before any model call, encrypted per session with erasure by key destruction, and recorded in an event-sourced, append-only audit trail; a browser interface streams progress and explains each routing decision.

The gates are evaluated on MedBridge-HalluBench, {{hallubench.n_items|int}} synthetic single- and multi-turn queries in six categories, through a paired ablation of six conditions (direct LLM, vanilla RAG, prompt-only safety, each gate alone and the full system) and two extensions (a calibrated evidence gate and a post-generation grounding verifier), with hallucination judged independently by an LLM judge from a different model family and an entailment (NLI) model, and conditions compared by exact McNemar tests with Holm correction.

[[include:abstract_results.md]]

**Keywords:** retrieval-augmented generation; hallucination; answerability; abstention; clinical decision support; hypertension; local large language models; patient privacy
