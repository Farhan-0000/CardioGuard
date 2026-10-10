# Discussion {#sec:discussion}

<!-- Sentences marked [TO BE FILLED after the study] must be written from the results;
     the rest discusses the design and does not depend on study numbers. -->

**Why two gates.** The gates decide with different information at different times. Gate 1 sees only the conversation and the recorded facts, so it can ask for a missing medicine or reading before any evidence is fetched; Gate 2 sees the passages and can tell a well-covered question from one the guidelines do not answer. A single post-retrieval gate would treat "I need to know your medicine" and "the guidelines do not say" as the same failure, although they need different replies. C3 and C4 measure each gate against the C1G reference. [TO BE FILLED after the study: whether the ablation shows complementary contributions, citing the paired tests.]

**Templates and coverage.** Serving ABSTAIN and ESCALATE from fixed texts removes generation, and with it hallucination, from the paths where an error costs most, and makes those replies easy to review clinically. The price is helpfulness when a gate abstains too readily, which is why coverage and over-abstention are reported beside the hallucination rates. [TO BE FILLED after the study: coverage cost of C5 relative to C1 and C2, and C5 against the prompt-only safety baseline C2.]

**Small local models.** Each gate is a narrow classification with a JSON schema, which suits small local models: in model selection a 4B model routed no significantly worse than an 8B model and was faster. Every gate adds a model call; starting retrieval with Gate 1 hides part of that cost.

**Expected failure modes.** Gate 2 may accept related but non-specific evidence as sufficient (V1 targets this); the generator may add details beyond its cited passages after a correct ANSWER decision, which citation validation cannot catch (V2 targets this); and the fast-path escalates some benign messages that mention a red-flag symptom, a cost we accept for an unsupervised tool. [TO BE FILLED after the study: which failure mode dominated, with counts from evaluation/results/error_analysis.md.]
