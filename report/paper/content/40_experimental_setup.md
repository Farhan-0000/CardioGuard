# Experimental Setup {#sec:setup}

We ask whether the Dual Answerability Gate reduces hallucinated and unsafe answers compared with LLM and RAG baselines, and what each gate contributes.

## MedBridge-HalluBench

The benchmark contains {{hallubench.n_items}} synthetic conversations with {{hallubench.n_turns}} patient turns, {{hallubench.n_multi_turn_items}} of them multi-turn ([@tab:hallubench_composition]); no real patient data are used. Most messages are casual ({{hallubench.by_style.casual}} items); others use Indian English ({{hallubench.by_style.indian_english}}), typos ({{hallubench.by_style.typo}}) or clinical wording ({{hallubench.by_style.clinical}}). Each category targets a safe behaviour: a cited answer for answerable items, a clarifying question first for items with missing context (some test the loop-breaker), general information for general topics, and abstention or correction for unanswerable in-domain items (fabricated drugs and treatments, false premises, doses or figures the guidelines do not state, paediatric and pregnancy questions). Out-of-domain items cover non-cardiac medicine and non-medical topics, and emergency items include colloquial red-flag descriptions. Each turn has a gold action and a set of acceptable actions (strict and lenient scoring), and notes name any fabricated entity or false premise. A stratified 20% development split (seed 42; {{hallubench.by_split.dev}} conversations) is used only for model selection and calibration; results use the {{hallubench.by_split.test}} test conversations.

[[table:hallubench_composition|Composition of MedBridge-HalluBench {#tab:hallubench_composition}]]

## Conditions

All conditions ([@tab:conditions]) share the application model, corpus and retrieval stack and are selected through an internal switch that the public API does not expose. C2 adds to the vanilla RAG prompt a block that asks, in prose, for the behaviour the gates enforce. The reference C1G uses the production prompt without gates, so each gate's effect is measured against a baseline that differs from C3 to C5 only by the gates. The emergency fast-path is active everywhere except C0, so emergency items favour no gated condition. Identical LLM requests are computed once and cached under the model's content digest, so paired conditions differ only in pipeline structure; latency charges each cached call the median uncached latency of the same call.

Table: Conditions compared in the study {#tab:conditions}

| Condition | Retrieval | Gate 1 | Gate 2 | Generator prompt |
|---|:-:|:-:|:-:|---|
| C0 Direct LLM | – | – | – | no evidence |
| C1 Vanilla RAG | ✓ | – | – | answer from passages |
| C1G RAG reference | ✓ | – | – | production |
| C2 RAG + prompt safety | ✓ | – | – | vanilla + safety rules |
| C3 Context Gate only | ✓ | ✓ | – | production |
| C4 Evidence Gate only | ✓ | – | ✓ | production |
| C5 MedBridge | ✓ | ✓ | ✓ | production |
| V1, V2, V12 | ✓ | ✓ | ✓ (V1: calibrated) | production (+ Gate 3) |

## Models

One application model serves the extractor, both gates and the generator. It was chosen by running C5 on the development split with qwen3.5:4b, qwen3:8b [@yang2025qwen3] and gemma3:4b [@gemmateam2025gemma3]. On {{model_selection.results.qwen3.5:4b.n_turns}} turns, strict routing accuracy was {{model_selection.results.qwen3.5:4b.routing.accuracy_strict|ci}}, {{model_selection.results.qwen3:8b.routing.accuracy_strict|ci}} and {{model_selection.results.gemma3:4b.routing.accuracy_strict|ci}} respectively (Wilson 95% intervals), with median turn latencies of {{model_selection.results.qwen3.5:4b.latency_turn_p50_ms|s}}, {{model_selection.results.qwen3:8b.latency_turn_p50_ms|s}} and {{model_selection.results.gemma3:4b.latency_turn_p50_ms|s}} s. An exact McNemar test found no significant difference between the two Qwen models (8 against 9 discordant turns, p = 1.00), so we use qwen3.5:4b ({{model_selection.results.qwen3.5:4b.ollama.parameter_size}} parameters, {{model_selection.results.qwen3.5:4b.ollama.quantization_level}}), which is faster and fits a 6 GB laptop GPU. Inference runs through Ollama on an Apple M5 machine with 24 GB of memory.
<!-- Sources: evaluation/results/model_selection/summary.json (inline values) and
     evaluation/results/model_selection/DECISION.md (McNemar counts; re-checked against
     evaluation/results/model_selection/runs/*/C5.jsonl); hardware from
     evaluation/results/runs/manifest_*.json. -->

## Hallucination Judges

The *LLM judge*, gemma3:12b [@gemmateam2025gemma3], is larger than the application model and from a different family; it runs at temperature 0 with structured output. It splits each response into atomic clinical claims [@min2023factscore], leaving out greetings, questions and generic referral advice, and labels each claim SUPPORTED, UNSUPPORTED or CONTRADICTED under a strict rubric (only an explicit statement or a direct paraphrase counts). The evidence per claim is the condition's own chunks plus the top three chunks of an independent judge-side retrieval, so C0, which retrieves nothing, is judged on equal terms. It also classifies the behaviour and flags accepted false premises, endorsed fabricated entities and unsupported doses, never seeing the condition or category. The *NLI judge* scores the same claims against the same passages with the cross-encoder nli-deberta-v3-base [@he2023debertav3], taking the maximum entailment over three-sentence windows, with thresholds calibrated on the development split. Template responses contain no claims. A blinded, stratified 50-item sheet is prepared for human annotation.

## Metrics and Statistics

The primary outcomes are the response-level hallucination rate (responses with at least one unsupported or contradicted claim) and the unsafe-answer rate on unanswerable items (a specific answer, a raised flag or an unsupported claim). Secondary outcomes include the unsupported-claim ratio, coverage and over-abstention on answerable items, routing accuracy, emergency recall, false escalation, citation precision and latency. The unit is the item-turn of the test split. Proportions carry Wilson 95% intervals [@wilson1927probable]. C5 is compared with C0 to C4, and each variant with C5, by exact McNemar tests [@mcnemar1947note] with Holm correction within each family [@holm1979simple]; Cochran's Q tests C0 to C5 jointly [@cochran1950comparison]. Effect sizes are risk differences and odds ratios; judge agreement is Cohen's κ [@cohen1960coefficient]. The protocol order is enforced in code: C5 is run and judged on the development split, thresholds are calibrated, and only then are all conditions run on the test split. Turns in which a component fell back to a default after an error are reported and excluded from headline metrics, and the analysis refuses headline results if they exceed 5% of any condition's turns. Sensitivity analyses include those turns, drop {{hallubench.by_tag.prompt_overlap}} items worded like prompt examples, and repeat the tests per item.
