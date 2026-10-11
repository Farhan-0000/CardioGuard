# Experimental Setup {#sec:setup}

## MedBridge-HalluBench

The benchmark contains {{hallubench.n_items}} synthetic conversations with {{hallubench.n_turns}} patient turns, {{hallubench.n_multi_turn_items}} of them multi-turn; no real patient data are used. Most messages are casual; others use Indian English, typos or clinical wording. Six categories each target a safe behaviour: a cited answer (answerable items), a clarifying question first (missing context, including loop-breaker tests), general information (general topics), abstention or correction (unanswerable in-domain items: fabricated drugs and treatments, false premises, doses the guidelines do not state, paediatric and pregnancy questions), abstention (out-of-domain items) and escalation (emergencies, including colloquial red flags). Each turn has a gold action and a set of acceptable actions (strict and lenient scoring). A stratified 20% development split (seed 42; {{hallubench.by_split.dev}} conversations) serves only model selection and calibration; results use the {{hallubench.by_split.test}} test conversations ({{study.per_condition.C5.n_turns}} turns).

## Conditions

Ten conditions are compared: C0, the application model without retrieval; C1, vanilla RAG; C1G, RAG with the production generator prompt; C2, vanilla RAG whose prompt asks, in prose, for the behaviour the gates enforce; C3, Context Gate only; C4, Evidence Gate only; C5, MedBridge with both gates; and the extensions V1 (C5 with the calibrated Gate 2), V2 (C5 with Gate 3) and V12 (both). All share the application model; all but C0 also share the corpus, the retrieval stack, the emergency fast-path and the small-talk handler. C3 to C5 differ from C1G only by their gates, so C1G anchors the gate effects, while C1 and C2 are the baselines of the decision rule. Identical LLM requests are cached by model digest, so paired conditions differ only in pipeline structure and prompt.

## Models

One application model serves the extractor, both gates and the generator. Run as C5 on the development split, qwen3.5:4b routed about as accurately as qwen3:8b [@yang2025qwen3] (strict accuracy {{model_selection.results.qwen3.5:4b.routing.accuracy_strict|ci}} against {{model_selection.results.qwen3:8b.routing.accuracy_strict|ci}}; exact McNemar p = 1.00) and ahead of gemma3:4b [@gemmateam2025gemma3] ({{model_selection.results.gemma3:4b.routing.accuracy_strict|ci}}), so we use qwen3.5:4b ({{model_selection.results.qwen3.5:4b.ollama.parameter_size}} parameters, {{model_selection.results.qwen3.5:4b.ollama.quantization_level}}), which is faster (median development turn {{model_selection.results.qwen3.5:4b.latency_turn_p50_ms|s}} against {{model_selection.results.qwen3:8b.latency_turn_p50_ms|s}} s) and whose 3.3 GB of weights should, unlike qwen3:8b's 5.2 GB, fit a 6 GB laptop GPU (not measured: every run used Ollama on an Apple M5 with 24 GB).
<!-- Sources: evaluation/results/model_selection/summary.json (inline values) and
     evaluation/results/model_selection/DECISION.md (McNemar counts; re-checked against
     evaluation/results/model_selection/runs/*/C5.jsonl); hardware from
     evaluation/results/runs/manifest_*.json. -->

## Hallucination Judges

The *LLM judge*, gemma3:12b [@gemmateam2025gemma3], is larger than the application model and from another family. At temperature 0 it splits each response into atomic clinical claims [@min2023factscore], leaving out greetings, questions and generic referral advice, and labels each claim SUPPORTED, UNSUPPORTED or CONTRADICTED under a strict rubric (only an explicit statement or a direct paraphrase counts). Each claim is labelled against the three passages a cross-encoder ranks highest in a pool of the condition's chunks and the production retriever's top three for the claim, so C0, which retrieves nothing, is still judged against evidence (from a smaller pool). Blind to condition and category but given each item's reference notes, it also classifies the behaviour and flags accepted false premises, endorsed fabricated entities and unsupported doses.

The *NLI judge* scores the same claims against the same passages with nli-deberta-v3-base [@he2023debertav3] (maximum entailment over three-sentence windows), with thresholds calibrated on the development split to agree with the LLM judge, so the two are not independent. To save compute, the project owner stopped it early after C0, C1, C1G and 76 C2 responses (no gated condition), so it serves only to measure agreement; all primary metrics and tests use the LLM judge.
<!-- Human annotation sheet: evaluation/annotation/human_annotation_sheet.xlsx (50 items,
     exported; labels to be filled by the team). -->

## Metrics and Statistics

The primary outcomes are the response-level hallucination rate (responses with at least one unsupported or contradicted claim) and the unsafe-answer rate on unanswerable items (a specific answer, a raised flag or an unsupported claim). Secondary outcomes include the unsupported-claim ratio, coverage and over-abstention on answerable items, routing accuracy and emergency recall. The unit is the item-turn; proportions carry Wilson 95% intervals [@wilson1927probable]. Exact McNemar tests [@mcnemar1947note] compare C5 with C0 to C4, each variant with C5, and C5 with C1G, with Holm correction within each family and outcome [@holm1979simple]; Cochran's Q tests C0 to C5 jointly [@cochran1950comparison]. Effect sizes are paired risk differences and odds ratios; judge agreement is Cohen's κ [@cohen1960coefficient].

A decision rule fixed before the study makes the variants the fallback if C5 does not significantly reduce (Holm-adjusted p < 0.05) the hallucination or unsafe-answer rate against both C1 and C2. All thresholds (NLI judge, V1, V2) are calibrated on C5's development-split runs before any test run. Turns in which a component fell back to a default after an error are excluded from headline metrics (the analysis refuses to run if they exceed 5% of any condition's turns).
