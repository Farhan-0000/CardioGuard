<!-- Included by appendices.md (Appendix 1). All tables are generated from evaluation/results/
     (summary.json, calibration.json, runs/, judgments/) by report/lib/tables.py. Further
     generated views exist and can be added if the page budget allows: claim_profile,
     routing_per_class, category_tests, sensitivity_rates, judge_confusion,
     latency_by_condition, degraded_turns and study_runs. -->

#### Hallucination Study by Item Category

The gates help mainly on unanswerable and out-of-domain questions; on general-topic questions only Gate 3 (V2, V12) lowers hallucination.

[[table:category_breakdown|Hallucination rate (LLM judge, %) by item category and condition]]

[[table:category_safety|Unsafe answers, appropriate abstention and coverage by item category]]

#### McNemar Tests under Every Analysis View

The table lists the unadjusted and Holm-adjusted p-values of all 18 headline tests, followed by the Holm-adjusted p-values in each sensitivity view. Only one conclusion changes: C5 against C4 on hallucination is not significant at item level.

[[table:mcnemar_sensitivity|McNemar tests under the sensitivity analyses]]

#### Degraded Turns, Failed Judgments and Provenance

Three of the 2,390 test turns were degraded and excluded from the headline metrics: in C0, HB-0205 lost its generator call, and in V1 and V12 turn 1 of HB-0084 lost all of its Gate 2 votes. Re-runs with the failing stage's cache bypassed stayed degraded, and including them changes no conclusion. The LLM judge failed on two non-medical C0 replies (HB-0207, HB-0213), so C0 is judged on {{study.per_condition.C0.llm.n_judged|int}} of {{study.per_condition.C0.n_valid|int}} valid turns.

[[table:judge_coverage|Provenance, judge coverage and failed judgments per condition]]

#### Run Manifests, Models and Seeds

Every run wrote a manifest with the commit, dataset checksum, model digests, seeds, settings and calibration status. The test split was generated on 10 October 2026 (06:28 to 15:32 UTC) in two interrupted-and-resumed runs at `82359eb` and follow-up runs at `085eb53` and `3f58b59` for C1G, V2, V12 and the retries; the LLM judge finished at 05:42 UTC on 11 October, and the early-concluded NLI run left no manifest.

[[table:study_models|Models, versions, hardware and seeds of the hallucination study]]
