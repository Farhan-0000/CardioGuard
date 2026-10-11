# Literature and Patent Review

This chapter reviews the clinical guidance that defines a correct answer, grounded generation and its failures, abstention and self-correction, medical language models and benchmarks, hallucination evaluation, retrieval, security and privacy, the Review-1 literature and related patents, and closes with a comparison of existing approaches and the research gap.

## Review Method

The review followed the questions raised by the design, from safe clinical sources to security obligations. Peer-reviewed papers, official guidelines and primary legal or standards texts were preferred, foundational papers were kept for retrieval and statistics, and LLM work dates mainly from 2022 onwards. Preprints are used only where no peer-reviewed version exists and are marked as such, and every record was checked against its publisher, proceedings, registry or patent entry. The six literature findings of the Review-1 presentation are matched to publications in Section 2.9.

## Clinical Background: Hypertension Guidance and Its Variability

A guideline-grounded assistant inherits both the strengths and the inconsistencies of its sources. The 2025 AHA/ACC multisociety guideline is the current United States reference for measurement, classification, thresholds, targets and severe hypertension, with several published corrections [@jones2025aha]. The 2017 ACC/AHA guideline separates a hypertensive emergency, severely raised BP (typically above 180/120 mmHg) with new or worsening target-organ damage, from urgency without it [@whelton2018acc]. The 2024 ESC guideline adds an elevated-BP category and a treated systolic target of 120 to 129 mmHg where tolerated [@mcevoy2024esc], whereas the 2023 ESH guideline keeps the classic grades [@mancia2023esh]; the 2020 ISH guideline separates "essential" from "optimal" care by resource setting [@unger2020ish], and the WHO pharmacological guideline excludes pregnancy [@who2021pharm]. NICE NG136 refers a clinic BP of 180/120 mmHg or higher the same day when signs such as new confusion, chest pain or heart failure are present [@nice2019ng136]. Comorbidity changes the answer further: KDIGO recommends a systolic target below 120 mmHg in CKD [@kdigo2021bp], the ADA revises its cardiovascular recommendations yearly [@ada2026cvd], and the heart failure [@heidenreich2022hf], atrial fibrillation [@joglar2024af], chronic coronary disease [@virani2023ccd] and primary prevention [@arnett2019primary] guidelines assume clinician oversight of medicines. Hence answers must cite their specific source; facts such as kidney disease, diabetes or pregnancy must be known before some questions are answered, which motivates a context check; emergency criteria are explicit enough for deterministic rules; clinician text must be rephrased for patients without losing its grounding; and each document's edition must be recorded.

## Retrieval-Augmented Generation and Hallucination

RAG couples a generator with a dense retriever over an external collection, so that knowledge can be updated without retraining and answers are more factual than those of purely parametric models [@lewis2020rag]. For a safety-critical setting its limitation is structural: retrieval is always performed and an answer always produced. MedBridge is a modular RAG system with query expansion, drug-name normalisation, fused retrievers with reranking and decision modules before and after retrieval.

Hallucinations may contradict the source (*intrinsic*) or add unverifiable content (*extrinsic*) [@ji2023survey]; for LLMs they divide into *factuality* and *faithfulness* errors that persist under retrieval [@huang2025survey]. Retrieval failure and unfaithful generation are therefore separate risks, addressed in MedBridge by the Evidence Gate before generation and by evidence-only generation, citation validation and V2 claim verification afterwards. Kim et al. define a medical hallucination as output that could alter clinical decisions and report, in a preprint, that general-purpose models gave hallucination-free answers more often than medically specialised ones (median 76.6 against 51.3 per cent) [@kim2025medhallucination].

Grounding lowers but does not eliminate hallucination. Wołk compared twelve RAG pipelines on 250 de-identified cases: hybrid retrieval with reciprocal rank fusion and cross-encoder reranking retrieved best, yet the best variant still hallucinated in 5.8 per cent of answers; confidence thresholds, verification and on-premises deployment are recommended without defining when to answer, abstain or escalate [@wolk2025ragcds]. Asgari et al. found 1.47 per cent hallucinated and 3.45 per cent omitted sentences among 12,999 clinician-annotated sentences of generated notes despite the full transcript being available [@asgari2025creola], and a preprint review recommends hybrids of RAG and fine-tuning [@jubair2026ragft]. A grounded system still needs an explicit decision about whether its evidence suffices.

## Abstention, Selective Prediction and Self-Correcting Generation

Selective classification calibrates a confidence threshold so that the error rate on accepted inputs stays below a target [@geifman2017selective]; rejecting more lowers risk but reduces coverage, so both must be reported. Conformal prediction gives finite-sample, marginal coverage guarantees under exchangeability [@angelopoulos2023conformal], but for generation it needs repeated sampling [@quach2024conformal], which is costly locally. Because calibration of self-assessed confidence weakens out of distribution and for smaller models (a preprint) [@kadavath2022know], MedBridge's gates reason over explicit inputs, the snapshot and the passages, and V1 calibrates evidence thresholds on the development split without formal guarantees. Evaluation of abstention, including over-abstention, remains immature [@wen2025abstention], so Chapter 4 reports both appropriate abstention and over-abstention.

Chen et al. treat RAG as risk control, refusing questions whose confidence, estimated by counterfactually varying the retrieved passages, is low [@chen2024controlling]; this is the closest prior work to ABSTAIN, but it targets factoid questions and cannot request missing context or escalate. Self-RAG trains a model to emit reflection tokens that decide whether to retrieve and whether passages are relevant and supported [@asai2024selfrag]; it is the closest prior work to gating, but needs a specially trained generator, makes learned rather than auditable decisions and has no abstain or escalate action. Corrective RAG (CRAG, a preprint) scores retrieved documents and refines them or falls back to web search [@yan2024crag]; its post-retrieval decision is the nearest analogue to the Evidence Gate, but it always answers, uses the open web and ignores patient context.

Post-hoc methods let the model answer first: Chain-of-Verification drafts, verifies with independent questions and revises [@dhuliawala2024cove]; RARR edits unsupported claims and attaches an attribution report [@gao2023rarr]; SelfCheckGPT flags sentences inconsistent across samples [@manakul2023selfcheckgpt]; and FActScore reports the fraction of atomic facts supported by a trusted source [@min2023factscore], the basis of this project's claim-level evaluation. They act only after an unsafe draft exists, can introduce new errors and cost several calls per query, so MedBridge decides *before* generation and adds verification (V2) only as a third gate.

## Medical Language Models, Medical RAG and Benchmarks

Med-PaLM 2 reached up to 86.5 per cent on MedQA and was preferred to physician answers on most axes [@singhal2025medpalm2], but it is a proprietary cloud model evaluated mainly on examination-style questions, and reviews weigh efficiency against inaccuracy, fabrication, misuse and privacy [@thirunavukarasu2023llmmedicine; @tian2024chatgptbiomed].

Almanac grounds an LLM in curated medical resources and beat general chatbots on factuality and adversarial safety in a clinician evaluation of 314 questions [@zakka2024almanac], but depends on a cloud LLM and has no context check, abstention or escalation. MedCite improves citation precision and recall [@wang2025medcite], while SourceCheckup found 50 to 90 per cent of responses from seven LLMs not fully supported by their own citations [@wu2025sourcecheckup]; neither decides whether to answer. MIRAGE and MedRAG compare 41 corpus, retriever and model combinations on 7,663 multiple-choice questions, with gains of up to 18 per cent from retrieval [@xiong2024medrag], but every item must be answered. Med-HALT probes false confidence and fake questions closed-book [@pal2023medhalt], and RABBITS shows accuracy drops of about 1 to 10 per cent when brand and generic names are swapped [@gallifant2024rabbits], which motivates BM25, a brand-to-generic table with Indian brands and a live RABBITS test in MedBridge. None measures answerability, abstention or patient context in dialogue, which MedBridge-HalluBench targets.

For local deployment, Qwen3 offers dense and mixture-of-experts models with switchable "thinking" [@yang2025qwen3] and Gemma 3 lightweight open models [@gemmateam2025gemma3]; the application model, `qwen3.5:4b`, belongs to the later Qwen3.5 generation and the judge, `gemma3:12b`, to Gemma 3. Both reports come from the vendors, so behaviour in cardiology is measured in Chapter 4. Ollama serves models locally [@ollama2026] and Qdrant stores dense and sparse vectors [@qdrant2026]; because Ollama also offers cloud-hosted models, MedBridge's local-only mode accepts only a loopback Ollama endpoint and refuses cloud-offloaded model tags.

## Evaluating Hallucination: Judges, Entailment and Statistics

A strong LLM judge agrees with human preferences about as well as humans agree with each other but shows position, verbosity and self-enhancement biases [@zheng2023judging], and LLMs favour their own outputs [@panickssery2024llm]. Hence the LLM judge here comes from a different model family, labels individual claims against evidence under a strict rubric, and is paired with an entailment judge and a human annotation subset. Natural language inference (NLI) models, usually trained on MultiNLI [@williams2018mnli] and often built on DeBERTaV3 [@he2023debertav3], check consistency best when texts are split into sentences and pairwise scores aggregated [@laban2022summac]; applied to atomic claims [@min2023factscore], this defines the NLI judge, whose threshold is calibrated on the development split because MultiNLI contains no clinical text.

Because every condition answers the same queries, comparisons are paired: the exact McNemar test uses the discordant pairs [@mcnemar1947note], Cochran's Q extends it to several conditions [@cochran1950comparison] and Holm's procedure controls the family-wise error rate [@holm1979simple]. Proportions take Wilson intervals, well behaved near zero or one [@wilson1927probable], and claim ratios and retrieval scores bootstrap intervals [@efron1993bootstrap]. Judge agreement uses Cohen's kappa [@cohen1960coefficient], read against the bands of Landis and Koch [@landis1977measurement] together with raw agreement, because kappa is low when one label is rare.

## Retrieval Components

BM25 scores documents by saturated term frequency with length normalisation [@robertson2009bm25]; it matches drug names and numbers exactly but cannot link "high blood pressure" to "hypertension". Dense Passage Retrieval beat BM25 by 9 to 19 points in top-20 accuracy on open-domain question answering [@karpukhin2020dpr], the BGE family provides compact English encoders [@xiao2024cpack], and hybrid pipelines with fusion did best on clinical cases [@wolk2025ragcds]. Reciprocal rank fusion (RRF) sums 1/(k + rank) over ranked lists, needs no score calibration and is commonly used with k = 60 [@cormack2009rrf]. A BERT cross-encoder reranking BM25 candidates topped the MS MARCO passage leaderboard (a preprint) [@nogueira2019passage], and distilled MiniLM models keep most of a teacher's accuracy at about half the computation [@wang2020minilm], making local reranking affordable; Chapter 4 tests whether this holds on cardiology text.

## Security, Privacy and Regulation

A model cannot reliably tell instructions from the data it processes. *Indirect* prompt injection plants instructions in content that an application later retrieves [@greshake2023notwhat], so retrieved passages are untrusted too; no single one of ten defences sufficed against five attacks [@liu2024formalizing]; and a review of 128 studies reports that architectural defences block up to about 95 per cent of known patterns while prompt-level defences carry no formal guarantees [@geng2026promptinjection]. Table 2.1 maps the relevant risks of the OWASP Top 10 for LLM Applications (2025) [@owasp2025llmtop10] to MedBridge's layered controls.

Table: OWASP LLM Top 10 (2025) risks and MedBridge controls {size=9}
| Risk (2025 identifier) | MedBridge control |
|---|---|
| LLM01 Prompt Injection | Patient text in untrusted-input XML blocks with tag neutralisation; deterministic emergency path and templates; adversarial test suite |
| LLM02 Sensitive Information Disclosure | Redaction before any model call and storage; local-only inference; encryption at rest; log scrubbing |
| LLM05 Improper Output Handling | Escape-first Markdown rendering; strict Content Security Policy |
| LLM06 Excessive Agency | Models have no tools; only the orchestrator accesses the database and vector store |
| LLM08 Vector and Embedding Weaknesses | Curated corpus pinned by checksums; write access only through offline ingestion |
| LLM09 Misinformation | Evidence Gate; citation validation and downgrade; ABSTAIN and GENERALIZE outcomes |
| LLM10 Unbounded Consumption | Rate limiting; 2,000-character message limit; request size limit |

India's DPDP Act, 2023 requires reasonable security safeguards (Section 8(5)) and erasure when consent is withdrawn or the purpose is served (Section 8(7)), and recognises a medical emergency as a legitimate use (Section 7(f)) [@dpdp2023india]. The U.S. HIPAA Privacy Rule does not bind an Indian project but serves as a design benchmark [@hhs_hipaa_privacy_rule]; its Safe Harbor list of eighteen identifier categories [@hhs2012deidentification] informed MedBridge's light, rule-based redactor, extended with Aadhaar, PAN and PIN codes. Event sourcing records every change as an appended, auditable event [@fowler2005eventsourcing] but does not say how personal data can be erased from an immutable log; cryptographic erase by key destruction [@nist2025sp80088r2] with key-management guidance [@barker2020sp80057pt1r5] resolves this per conversation (crypto-shredding). MedBridge is not a medical device, but its audit log, transparency panel and conservative actions follow WHO guidance on safety, transparency and human oversight [@who2021ethicsai].

## Literature Cited in the Review-1 Presentation

The Review-1 presentation summarised six literature findings by venue and year only; each was matched to a publication through bibliographic databases (Table 2.2). Row 3 matches only with low confidence: the closest record, a January 2026 preprint, presents RAG favourably rather than clearly supporting the row's claim, which rows 1 and 2 make with peer-reviewed evidence. [TO BE FILLED: team to confirm the Authorea source intended in Review-1 row 3, or remove the row] The article of row 4 was accepted in 2025 but appears in a 2026 issue, the year used in the references.

Table: Review-1 literature rows and matched publications {size=9}
| Review-1 row | Matched publication | Main finding | Bearing on MedBridge |
|---|---|---|---|
| 1. Electronics (MDPI) 2025: grounding reduces hallucination but is not sufficient alone | Wołk [@wolk2025ragcds] | Twelve RAG variants on 250 cases; best still hallucinated in 5.8 per cent | Hybrid retrieval and an explicit answer, abstain or escalate rule |
| 2. npj Digital Medicine 2025: hallucination persists in grounded summarisation | Asgari et al. [@asgari2025creola] | 1.47 per cent hallucination, 3.45 per cent omission in 12,999 sentences | Grounded output still needs checking; claim-level labels |
| 3. Authorea 2025: RAG alone does not guarantee safety without verification | Jubair [@jubair2026ragft] (low-confidence match) | Preprint review of RAG versus fine-tuning; recommends hybrids | Uncertain match; see note above |
| 4. CMC 2025: structural input isolation defends against prompt injection | Geng et al. [@geng2026promptinjection] | Review of 128 studies; instruction-data separation; no formal guarantees | XML isolation as one layer of a layered defence |
| 5. ACL Findings 2025: citation verifiability in medical generation | Wang et al., MedCite [@wang2025medcite] | Better citation precision and recall; automatic evaluation agrees with experts | Validation of citation markers against retrieved passages |
| 6. ACL/EMNLP Findings 2024: risk-controlled RAG generation | Chen et al. [@chen2024controlling] | Refusal of low-confidence questions via counterfactual prompting | Closest prior work to ABSTAIN; no context gate or escalation |

## Patent Review

A keyword search of Google Patents (clinical decision support with LLMs, answerability and abstention, hallucination detection, RAG verification) found six granted patents related to parts of the design, each checked against its public record (Table 2.3); the comparison positions the contribution and is not a legal opinion. The closest, US 12,142,371 B1 (Hippocratic AI) [@shah2024hippocraticconstellation], uses control logic, in its embodiments parallel LLMs, that can hand a "kickout" answer to a human, whereas MedBridge's emergency path is rule-based and precedes generation. The Intuit disambiguation patent [@zhang2024intuitdisambiguity] is the nearest analogue to SOFT-ASK, but is triggered by linguistic ambiguity rather than missing clinical facts and has no clarification cap or evidence gate.

Table: Related patents and how MedBridge differs {size=9}
| Patent (assignee, year granted) | Claimed subject matter | How MedBridge differs |
|---|---|---|
| US 12,142,371 B1 (Hippocratic AI, 2024) [@shah2024hippocraticconstellation]: *Low-latency conversational AI architecture with a parallelized in-depth analysis feedback loop* | Parallel control LLMs shape a patient conversation; "kickout" hand-off to a human | Deterministic emergency path before generation; dual pre-generation gates not claimed |
| US 12,243,646 B1 (Glass Health, 2025) [@paul2025glasshealthrag]: *Systems and methods for improving retrieval-augmented generation in clinical decision support* | Clinician-facing RAG over a validated guideline library producing differentials and plans | Patient-facing and local; never diagnoses; explicit abstain and escalate outcomes |
| US 9,740,769 B2 (IBM, 2017) [@bhagwat2017ibmnoanswer]: *Interpreting and distinguishing lack of an answer in a question answering system* | Ranked "no answer" candidate in a QA pipeline | Four graded evidence outcomes plus a separate context gate |
| US 12,505,299 B2 (Casetext, 2025) [@delevie2025casetextverification]: *Hallucination detection and remediation in text generation interface systems* | LLM checks each generated portion against its cited passages | Decides before generation; deterministic citation validation; verification only as an added gate |
| US 12,387,049 B2 (Intuit, 2025) [@zhang2025intuithallucination]: *Semantic aware hallucination detection for large language models* | Reference-free detection by consistency across rewritten prompts and two LLMs | Evidence-grounded decisions; single local model; no multi-sample cost |
| US 12,038,918 B1 (Intuit, 2024) [@zhang2024intuitdisambiguity]: *Disambiguity in large language models* | LLM detects query ambiguity and asks a clarifying question | Triggered by missing clinical context; capped by the loop-breaker; combined with evidence gate and emergency path |

## Comparison of Existing Approaches

Table 2.4 compares representative approaches with MedBridge. A *pre-generation answerability check* is any explicit decision, before answer text is produced, about whether the system has what it needs; an *evidence sufficiency check* judges the retrieved evidence; a *deterministic emergency path* is a rule-based route independent of a model; and *citation validation* checks that citations match supporting sources. "Partial" denotes a limited or learned form, a dash that the aspect is not described.

Table: Comparison of existing approaches with MedBridge {size=8}
| Approach | Pre-generation answerability check | Evidence sufficiency check | Abstain / escalate | Deterministic emergency path | Local / offline | Citation validation | Hallucination evaluation |
|---|---|---|---|---|---|---|---|
| RAG [@lewis2020rag] | No | No | No | No | — | No | Factuality vs. parametric models |
| Self-RAG [@asai2024selfrag] | Partial (learned retrieval token) | Partial (learned critique) | No | No | — | Partial | Factuality, citation accuracy |
| CRAG [@yan2024crag] | No | Yes (retrieval evaluator) | No | No | No (web search) | No | — |
| CoVe [@dhuliawala2024cove] | No | No | No | No | — | No | Fewer hallucinated facts |
| RARR [@gao2023rarr] | No | Partial (after generation) | No | No | No (web search) | Yes (attribution report) | Attribution |
| SelfCheckGPT [@manakul2023selfcheckgpt] | No | No | No | No | — | No | Sampling consistency |
| Risk-controlled RAG [@chen2024controlling] | Partial (counterfactual confidence) | Partial | Abstain only | No | — | No | Risk and carefulness |
| Med-PaLM 2 [@singhal2025medpalm2] | No | No | No | No | No (cloud) | No | Clinician ratings |
| MedRAG [@xiong2024medrag] | No | No | No | No | — | No | Accuracy only |
| Almanac [@zakka2024almanac] | No | No | No | No | No (cloud) | — | Clinician ratings |
| MedCite [@wang2025medcite] | No | No | No | No | — | Yes | Citation precision and recall |
| Clinical RAG variants [@wolk2025ragcds] | No | Recommended only | No | No | Recommended | — | Hallucination rate |
| US 12,142,371 B1 [@shah2024hippocraticconstellation] | Partial (control LLMs) | No | Escalate to human | No | — | No | — |
| US 9,740,769 B2 [@bhagwat2017ibmnoanswer] | No | Yes (no-answer candidate) | Abstain | No | — | No | — |
| **MedBridge (this work)** | **Yes (Context Gate)** | **Yes (Evidence Gate)** | **Yes (templates)** | **Yes (rule-based)** | **Yes (Ollama)** | **Yes (marker validation)** | **Paired ablation; LLM judge (NLI judge for agreement on ungated outputs)** |

The ingredients exist separately: evidence evaluation in CRAG and Self-RAG, refusal in risk-controlled RAG and the IBM patent, citation checking in MedCite and RARR, hand-off to humans in the Hippocratic AI patent and local deployment as a recommendation in clinical RAG studies. No approach combines a pre-generation check of *patient context* with a separate check of *evidence*, deterministic emergency handling, local inference and citation validation, or evaluates such gating on hallucination through a paired ablation with a cross-family LLM judge and an NLI judge.

## Limitations of Existing Work and Research Gap

Table 2.5 extends the Review-1 gap analysis: the limitations of existing work for a safe, patient-facing cardiovascular assistant, MedBridge's response and where it is evaluated.

Table: Limitations of existing work and the MedBridge response {size=9}
| Limitation of existing work | MedBridge response | Where evaluated |
|---|---|---|
| Answer-by-default designs and benchmarks; residual hallucination under grounding [@lewis2020rag; @xiong2024medrag; @wolk2025ragcds; @asgari2025creola] | Two explicit pre-generation gates routing every message to one of five actions | Hallucination study, C0 to C5 (Ch. 4) |
| Evidence-only answerability, no check of patient context [@asai2024selfrag; @yan2024crag] | Context Gate before retrieval with a loop-breaker after two SOFT-ASK turns | Each gate alone (C3, C4) against the full gate (C5) |
| Repair after the fact at several calls per query [@dhuliawala2024cove; @gao2023rarr; @manakul2023selfcheckgpt] | Decision before generation; a post-generation grounding verifier (Gate 3, V2) strips unsupported sentences, enabled by default since the study | V2 against C5 |
| Learned or opaque decisions; calibration weak for small models [@asai2024selfrag; @kadavath2022know; @wen2025abstention] | Gate rationales in an append-only audit log; V1 thresholds calibrated on the development split | Audit trace; V1 against C5 |
| Emergency handling depends on a model | Deterministic emergency fast-path before any LLM call | Emergency recall, false escalation, classifier latency |
| Unsupported citations [@wu2025sourcecheckup; @wang2025medcite] | Marker validation against retrieved passages; downgrade of uncited answers | Citation validity and supported-citation precision |
| Cloud dependence and web search conflict with health-data privacy [@singhal2025medpalm2; @zakka2024almanac; @yan2024crag; @gao2023rarr; @dpdp2023india] | Local-only inference, redaction, encryption at rest, crypto-shredding | Redaction on development and blind sets; privacy tests |
| Injection defences studied apart from clinical RAG [@liu2024formalizing; @geng2026promptinjection] | XML isolation in all prompts; templates outside the generator | Adversarial and injection suites |
| Narrow evaluation: multiple-choice accuracy or a single biased judge [@pal2023medhalt; @gallifant2024rabbits; @zheng2023judging; @panickssery2024llm] | Cross-family LLM judge, NLI judge, human sheet, paired statistics | Judge agreement and statistical tests (Ch. 4) |

The research gap can therefore be stated precisely. Within the literature reviewed, no system makes a **pre-generation, two-part answerability decision** that separates *"I do not know enough about you"* (insufficient patient context, decided before retrieval) from *"I do not have enough evidence"* (insufficient guideline support, decided after retrieval but before generation), routes the outcome to graded actions with deterministic refusal and emergency templates, and puts a deterministic emergency check in front of both. Nor has any study measured, through a paired ablation with a cross-family LLM judge (and an NLI judge as a check on agreement) on fully local open-weight models, whether such gating reduces hallucination and unsafe answers compared with retrieval alone and with prompt-only safety. MedBridge addresses this gap with the Dual Answerability Gate (Chapter 3) and the MedBridge-HalluBench study (Chapter 4).
