#* Abstract

Hypertension affects roughly one in three adults worldwide, and many patients now turn to conversational AI between clinic visits. Large language models answer fluently, but they can invent clinical facts, apply general advice to a patient whose missing details would change it, and miss emergencies. Retrieval-augmented generation reduces fabrication, yet it does not decide *whether* it is safe to answer at all.

This report presents **MedBridge** (presented as *CardioGuard*), a safety-first, evidence-grounded consultation assistant for cardiovascular care centred on hypertension. Its core contribution is a **Dual Answerability Gate**: a *Context Gate* before retrieval asks whether enough is known about the patient and either requests the missing facts (SOFT-ASK, at most twice in a row) or proceeds, and an *Evidence Gate* after retrieval routes the query to ANSWER, GENERALIZE, ABSTAIN or ESCALATE according to whether the retrieved guideline passages suffice. A deterministic emergency fast-path runs before any model call, refusals and referrals use pre-vetted templates, and inline citations are validated against the retrieved passages.

MedBridge combines hybrid dense and BM25 retrieval, reciprocal rank fusion and cross-encoder reranking over {{corpus.totals.documents|int}} curated documents ({{corpus.totals.chunks|comma}} passages), with all inference on locally hosted open-weight models. Patient text is redacted before any model call, encrypted per session with erasure by key destruction, and recorded in an append-only audit trail. The gates are evaluated on MedBridge-HalluBench, {{hallubench.n_items|int}} synthetic single- and multi-turn queries, through a paired ablation of six conditions and two extensions, with hallucination judged by a cross-family LLM judge and an entailment model and conditions compared by exact McNemar tests with Holm correction.

[[include:abstract_results.md]]

**Keywords:** retrieval-augmented generation; hallucination; answerability; abstention; clinical decision support; hypertension; local large language models; patient privacy
