<!--
Abstract and index terms. IEEE asks for a single paragraph of at most about 250 words,
with no citations. Rates and kappas are {{...}} values from evaluation/results/summary.json;
the two Holm-adjusted p-values are typed by hand from summary.json tests.mcnemar[7]
(largest of the four C5 vs C1/C2 tests, 2.79e-9) and tests.mcnemar[11] (V2 vs C5,
4.85e-9). The NLI judge was stopped early by the project owner's decision (it covers
C0, C1, C1G and 76 C2 responses) and is used only for judge agreement.
-->

#* Abstract

In patient-facing care, a fluent answer to a question the evidence does not cover is a safety risk, yet standard retrieval-augmented generation (RAG) answers every question. MedBridge, a local assistant for hypertension and cardiovascular care, makes answerability an explicit, auditable decision: a deterministic emergency fast-path, a Context Gate that asks for missing personal facts and, after hybrid retrieval, an Evidence Gate that decides whether to answer, generalise, abstain or escalate. On {{hallubench.by_split.test}} test conversations of the synthetic MedBridge-HalluBench, with a 4B-parameter open-weight model and an LLM judge from another model family, the dual gate lowered the response-level hallucination rate to {{study.per_condition.C5.llm.hallucination|ci}} from {{study.per_condition.C1.llm.hallucination|pct}} for vanilla RAG and {{study.per_condition.C2.llm.hallucination|pct}} for prompt-only safety, and the unsafe-answer rate on unanswerable turns to {{study.per_condition.C5.llm.unsafe_answer|pct}} from {{study.per_condition.C1.llm.unsafe_answer|pct}} and {{study.per_condition.C2.llm.unsafe_answer|pct}} (exact McNemar, Holm-adjusted p ≤ 2.8×10⁻⁹). The gain came mostly from selective abstention: coverage of answerable turns fell from {{study.per_condition.C1.coverage_answerable|pct}} to {{study.per_condition.C5.coverage_answerable|pct}}, and responses with clinical claims were hallucinated as often as before. A post-generation grounding verifier lowered hallucination further, to {{study.per_condition.V2.llm.hallucination|pct}} (p = 4.9×10⁻⁹), mainly by shortening answers ({{study.per_condition.V2.llm.claims_per_response|f2}} against {{study.per_condition.C5.llm.claims_per_response|f2}} claims per response, unsupported-claim ratio unchanged) and at a further cost in coverage ({{study.per_condition.V2.coverage_answerable|pct}}), without significantly reducing unsafe answers; a calibrated Evidence Gate had no measurable effect. A second, NLI-based judge, stopped early, scored only ungated outputs (agreement κ = {{study.judge_agreement.kappa_response_hallucination.kappa|f2}} per response, {{study.judge_agreement.kappa_binary.kappa|f2}} per claim).

#* Index Terms

Retrieval-augmented generation, hallucination, abstention, clinical decision support, hypertension, large language models, patient safety
