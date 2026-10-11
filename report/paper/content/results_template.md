<!--
TEMPLATE for Section V (Results). Not rendered on its own: copy this file to
results.md once evaluation/results/summary.json exists, then edit results.md.

Every [[table:...]] below is generated from summary.json by report/lib/tables.py and
shows [TABLE PENDING: ...] until the file exists. Figure paths are those written by
evaluation/analyze.py. {{study....}} values follow the summary.json schema, e.g.
{{study.per_condition.C5.llm.hallucination|ci}} or
{{study.per_condition.C5.llm.unsafe_answer|ci}}. Tables with many columns are set
across both columns automatically; add ".wide" to a label ({#tab:x .wide}) to force it.

Page budget: the paper is about 5 pages without this section and most venues allow 6
(ICITIIT: 6, up to 8 with a fee; EMBC: 4 + 1). Keep 2 tables and 1-2 figures here and
move the rest to the report.
-->

## Hallucination and Unsafe Answers

[TO BE FILLED: two or three sentences describing the hallucination rates of C5 against C0, C1 and C2 for the LLM judge (the NLI judge scored no gated condition), with the confidence intervals from the table.]

[[table:hallucination_by_condition|Hallucination by condition on the test split (unit: item-turn) {#tab:hallucination}]]

[[figure:reports/figures/hallucination_rate_by_condition.png|Response-level hallucination rate per condition with Wilson 95% intervals (LLM judge; NLI bars only where it scored the whole condition). {#fig:hallucination}|width=3.4]]

[TO BE FILLED: unsafe answers and appropriate abstention on unanswerable items.]

[[table:unsafe_answer|Unsafe answers and appropriate abstention on unanswerable items {#tab:unsafe}]]

## Paired Comparisons

[TO BE FILLED: McNemar results with Holm-adjusted p-values and effect sizes; say explicitly which comparisons are and are not significant.]

[[table:mcnemar|Paired exact McNemar tests with Holm correction {#tab:mcnemar}]]

## Routing, Coverage and Emergencies

[TO BE FILLED: routing accuracy, coverage/over-abstention on answerable items, emergency recall and false escalations.]

[[table:routing_by_condition|Routing accuracy and emergency handling by condition {#tab:routing}]]

## Ablation, Variants and Cost

[TO BE FILLED: contribution of each gate (C3 and C4 against the C1G anchor), effect of V1/V2, and latency per condition; LLM judge only for gated conditions.]

[[figure:reports/figures/ablation_waterfall.png|Change in hallucination rate as each gate is added. {#fig:waterfall}|width=3.4]]

## Judge Agreement

[TO BE FILLED: Cohen's kappa between the LLM and NLI judges on the test split, stating that the NLI judge labels the LLM judge's claims against the same evidence with thresholds tuned on dev to maximise this agreement (dev values {{calibration.nli.kappa_binary|f2}} binary, {{calibration.nli.kappa_3class|f2}} three-class), so κ measures the consistency of two complementary judges, not independent validation; human agreement if the annotation sheet has been completed.]

[[table:judge_agreement|Agreement between the LLM judge and the NLI judge {#tab:agreement}]]
