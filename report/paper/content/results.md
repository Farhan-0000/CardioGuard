<!--
Section V (Results), included by 50_results.md. Written from the hallucination study
(python -m evaluation.analyze, 2026-10-11).

Sources of every number:
- {{study....}} values and the generated table (report/paper/paper_tables.py,
  paper_outcomes: rates with Wilson intervals and the Holm-adjusted McNemar p-value of each
  condition against C5) read evaluation/results/summary.json at build time.
- Fig. 2 is drawn from summary.json by report/paper/figures/make_results_figure.py.
- Typed by hand (no filter can print them): the risk differences in percentage points and
  the Holm-adjusted p-values, copied from summary.json tests.mcnemar[1, 2, 4, 6, 7, 10, 11,
  14, 16, 17] (the p-values match Table I, which is generated; the full test table with
  b / c counts and risk-difference intervals is the generator paper_tests, kept for a
  supplement); the item-level p-values from tests.sensitivity_item_level[1, 4]; the
  prompt-overlap sensitivity p-value from sensitivity.without_prompt_overlap.tests[1]; the
  Cochran's Q p-value bound from tests.cochran_q; the counts of degraded turns
  (degraded.per_condition), of C0 judge failures (C0 n_valid 238 - n_judged 236); NLI
  CONTRADICTED 2,606 vs LLM 25 from judge_agreement.confusion_llm_rows_nli_cols (column /
  row sums); the Cochran's Q complete-case counts (n = 236 and 63) from
  tests.cochran_q[*].n; the 83.9-86.3% range of hallucination_given_claims over C1, C1G,
  C2, C3, C4, C5 and V1 from per_condition.
- The NLI judge's C2 result (partial subset, 76 answerable responses) is deliberately not
  reported as a condition result.
-->

## Hallucination and Unsafe Answers

Of the {{study.per_condition.C5.n_turns}} test turns per condition, three in all fell back to a default after an error (one each in C0, V1 and V12) and are excluded, and the LLM judge failed on two non-medical C0 replies. [@tab:outcomes] lists the primary outcomes with the paired test of each condition against C5.

[[table:paper_outcomes|Primary outcomes on the test split (LLM judge) and paired exact McNemar tests against C5 {#tab:outcomes .column}]]

The dual gate (C5) produced a hallucinated response in {{study.per_condition.C5.llm.hallucination|ci}} of turns, against {{study.per_condition.C1.llm.hallucination|pct}} for vanilla RAG (C1), {{study.per_condition.C2.llm.hallucination|pct}} for prompt-only safety (C2) and {{study.per_condition.C0.llm.hallucination|pct}} for the direct LLM (C0). On unanswerable turns its unsafe-answer rate was {{study.per_condition.C5.llm.unsafe_answer|ci}}, against {{study.per_condition.C1.llm.unsafe_answer|pct}} (C1) and {{study.per_condition.C2.llm.unsafe_answer|pct}} (C2). The NLI judge, which scored C0, C1 and C1G completely and no gated condition, rated them at least as harshly ({{study.per_condition.C0.nli.hallucination|pct}}, {{study.per_condition.C1.nli.hallucination|pct}} and {{study.per_condition.C1G.nli.hallucination|pct}} for C0, C1 and C1G).

## Paired Tests and the Decision Rule

C5 reduced hallucination by 35.6 percentage points (pp) against C1 (odds ratio 0.15) and by 30.5 pp against C2, and unsafe answers by 69.7 and 47.0 pp, with Holm-adjusted p ≤ 2.8×10⁻⁹ in all four tests (all McNemar p-values below are Holm-adjusted), so the fallback of our decision rule is not triggered. Cochran's Q confirms that C0 to C5 (C1G not included) differ on both outcomes (Q = {{study.tests.cochran_q.0.q|f1}} and {{study.tests.cochran_q.1.q|f1}}, df = 5, n = 236 and 63 complete cases, p < 10⁻³⁰). In sensitivity analyses (no C1, C2 or C5 turn was degraded, so including degraded turns changes none of the four tests), all four tests stay significant when the {{study.sensitivity.without_prompt_overlap.n_items_removed}} test items resembling few-shot examples of earlier prompt versions are removed (C5 vs C1 hallucination: p = 1.3×10⁻¹³) and per item instead of per turn (p = 1.1×10⁻¹¹), and both hallucination tests when the non-clinical claim filter is disabled.

## Where the Reduction Comes From

The gates lowered the rate mainly by producing fewer responses with clinical content, not better-grounded ones ([@fig:composition]). C5 returned clinical claims in {{study.per_condition.C5.llm.hallucination_given_claims.n}} of {{study.per_condition.C5.n_turns}} turns, against {{study.per_condition.C1.llm.hallucination_given_claims.n}} for C1, but among those responses the hallucination rate was unchanged ({{study.per_condition.C5.llm.hallucination_given_claims|pct}} against {{study.per_condition.C1.llm.hallucination_given_claims|pct}}), and the pooled unsupported-claim ratios overlap ({{study.per_condition.C5.llm.unsupported_claim_ratio|cif}} against {{study.per_condition.C1.llm.unsupported_claim_ratio|cif}}). The price is coverage: C5 answered {{study.per_condition.C5.coverage_answerable|pct}} of answerable turns against {{study.per_condition.C1.coverage_answerable|pct}} for C1, and abstained on {{study.per_condition.C5.over_abstention|pct}}.

[[figure:report/paper/figures/response_composition.png|Composition of the judged test responses (LLM judge). Hatched: response-level hallucination rate; right column: hallucination rate among responses with clinical claims. {#fig:composition}|width=3.1]]

## Gate Ablation and Variants

With the production prompt but no gates, C1G hallucinated about as often as C1 ({{study.per_condition.C1G.llm.hallucination|pct}}); against this anchor, C5 lowered hallucination by 36.8 pp (p = 2.8×10⁻²⁰) and unsafe answers by 65.2 pp (p = 2.6×10⁻¹²), so the effect is due to the gates, not to the prompt. Descriptively (not tested), Gate 1 alone (C3) lowered unsafe answers only to {{study.per_condition.C3.llm.unsafe_answer|pct}} (from {{study.per_condition.C1G.llm.unsafe_answer|pct}}), Gate 2 alone (C4) to {{study.per_condition.C4.llm.unsafe_answer|pct}}. Adding Gate 1 to Gate 2 (C5 vs C4) reduced hallucination by a further 6.3 pp (p = 0.0059), but not significantly per item (p = 0.12) or on unsafe answers (p = 1.0), and lowered appropriate abstention ({{study.per_condition.C5.appropriate_abstention|pct}} against {{study.per_condition.C4.appropriate_abstention|pct}}), as Gate 1 turned some abstentions into clarifications.

V1 had no measurable effect (−0.8 pp, p = 0.73) and was about 1.6 times slower (imputed study latency, relative only). V2 lowered hallucination to {{study.per_condition.V2.llm.hallucination|ci}} (−15.9 pp [−20.9, −10.8], p = 4.9×10⁻⁹) and, with V12, which includes it, was the only condition with a clearly lower rate among claim-bearing responses ({{study.per_condition.V2.llm.hallucination_given_claims|pct}} and {{study.per_condition.V12.llm.hallucination_given_claims|pct}}, against 83.9–86.3% for every other condition with retrieval; not tested). Its unsafe-answer reduction was not significant ({{study.per_condition.V2.llm.unsafe_answer|pct}} against {{study.per_condition.C5.llm.unsafe_answer|pct}}, p = 0.75); it answered fewer answerable turns ({{study.per_condition.V2.coverage_answerable|pct}}, not tested) and its responses carried fewer claims ({{study.per_condition.V2.llm.claims_per_response|f2}} against {{study.per_condition.C5.llm.claims_per_response|f2}}), while its unsupported-claim ratio stayed within C5's interval ({{study.per_condition.V2.llm.unsupported_claim_ratio|cif}}), so V2 lowered the response-level rate largely by shortening answers, not by better supporting the remaining claims; on the development split its sentence verdicts agreed only weakly with the LLM judge's (κ = {{calibration.v2.kappa_sentence|f2}}, {{calibration.v2.n_sentences}} sentences). V12 behaved like V2 ({{study.per_condition.V12.llm.hallucination|pct}}); the two were not compared directly.

## Routing, Emergencies and Judge Agreement

C5 chose the gold action in {{study.per_condition.C5.routing_system.accuracy_strict|pct}} of turns ({{study.per_condition.C5.routing_system.accuracy_lenient|pct}} counting acceptable actions; macro-F1 {{study.per_condition.C5.routing_system.macro_f1|f2}}); its weakest boundary is ANSWER versus GENERALIZE (ANSWER recall {{study.per_condition.C5.routing_system.per_class.ANSWER.recall|f2}}). In every condition with the fast-path, all {{study.per_condition.C5.emergency.recall_system.n}} emergency turns were escalated and no other turn was. On the {{study.judge_agreement.n_responses|comma}} ungated responses that both judges scored, response-level agreement was substantial (κ = {{study.judge_agreement.kappa_response_hallucination|cif}}), but claim-level agreement over {{study.judge_agreement.n_claims|comma}} claims was only fair (κ = {{study.judge_agreement.kappa_binary|cif}}): the NLI judge labelled 2,606 claims CONTRADICTED, the LLM judge 25.
