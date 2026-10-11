# Related Work {#sec:related}

<!-- Trimmed for the 6-page limit (11 Oct 2026): Chain-of-Verification, SelfCheckGPT, the
     second LLM-judge self-preference paper, the NCD-RisC trend analysis, the MiniLM
     reranking paper, the prompt-injection paper and the IBM "no answer" patent are no
     longer cited (they remain in report/references/references.json). -->

**Grounding and verification.** Self-RAG fine-tunes a generator to decide when to retrieve and whether its output is supported [@asai2024selfrag], and Corrective RAG scores retrieved documents and replaces them when confidence is low [@yan2024crag]. Post-hoc methods such as RARR retrieve evidence per claim and revise unsupported content [@gao2023rarr]; FActScore measures factual precision over atomic facts [@min2023factscore]. These need a trained generator, answer every input or act only after generation; MedBridge decides *before* generation, and its Gate 3 uses the post-hoc idea only to make an output more conservative.

**Abstention.** Selective classification rejects inputs below a confidence threshold calibrated to bound the risk on accepted ones [@geifman2017selective]. Such methods give one answer-or-refuse signal; MedBridge separates missing context (ask) from insufficient evidence (generalise or abstain) and adds escalation.

**Evaluation and prior art.** MultiMedQA and MIRAGE consist mainly of exam-style questions that all expect an answer [@singhal2023llmclinical; @xiong2024medrag]; our benchmark labels each turn with the expected behaviour, including asking, abstaining and escalating. Because LLM judges can favour their own generations [@zheng2023judging], our judge comes from another model family. Patents cover clarifying questions for ambiguous queries [@zhang2024intuitdisambiguity] and RAG-based clinical decision support over curated guidelines [@paul2025glasshealthrag]; none we reviewed combines a pre-retrieval context gate, a post-retrieval evidence gate and a deterministic emergency override.
