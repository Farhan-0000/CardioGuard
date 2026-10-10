# Introduction

Raised blood pressure develops silently over years, and the people who live with it make most of their health decisions alone: when to re-check a reading, whether a dizzy spell is related to a new tablet, what to eat, or whether a number on the home monitor means they should go to hospital. Conversational artificial intelligence (AI) appears to offer exactly the patient, always-available explainer that such people need. The same systems, however, can state an invented drug interaction with complete confidence, personalise advice to a patient they know almost nothing about, or treat a description of a hypertensive emergency as a routine question. This report describes **MedBridge**, a consultation assistant for hypertension and cardiovascular care that is designed around a simple principle: *before generating an answer, the system must decide whether it is entitled to give one*. The project is presented under the title *CardioGuard*; MedBridge is the name of the software system throughout this report.

This chapter sets out the clinical and technical background that motivates the work, states the problem and the research questions, lists the objectives and the scope of the project, summarises the expected outcomes and explains how the rest of the report is organised.

## Background and Motivation

### The Burden of Hypertension and Cardiovascular Disease

Hypertension is among the most widespread chronic conditions in the world. The first global report on hypertension published by the World Health Organization (WHO) estimates that about one adult in three is affected, that the number of people living with the condition doubled to roughly 1.3 billion between 1990 and 2019, and that around four out of five of them are not adequately treated [@who2023globalreport]. A pooled analysis by the NCD Risk Factor Collaboration, built on 1201 population-based studies from 200 countries, reaches a similar picture: the number of adults aged 30 to 79 years with hypertension approximately doubled over three decades, to about 626 million women and 652 million men in 2019, and the gaps in diagnosis, treatment and control are widest in low- and middle-income countries [@ncdrisc2021hypertension]. Uncontrolled blood pressure (BP) carries heavy health and economic consequences because it contributes to stroke, heart attack, heart failure and kidney disease, which is why the WHO report pairs its burden estimates with lessons from treatment programmes such as HEARTS [@who2023globalreport].

India carries a large share of this burden. The ICMR-INDIAB national survey, which examined 113,043 adults aged 20 years and above across 31 states and union territories, estimated hypertension prevalence at 35.5 per cent using JNC-8 criteria, alongside high levels of diabetes, obesity and dyslipidaemia [@anjana2023indiab]. The fifth National Family Health Survey (NFHS-5), which measured BP in a household survey of people aged 15 years and above, found hypertension in about 21 per cent of women and 24 per cent of men, and reported that only 67 per cent of women and 54 per cent of men had ever had their BP measured before the survey [@iips2021nfhs5]. Because the two surveys use different age ranges and definitions, their figures are not directly comparable, but both point to a large population of people who are either unaware of their condition or living with it outside regular specialist care. The India Hypertension Control Initiative (IHCI) has shown that simple, protocol-based treatment in public clinics can reach large numbers of patients: by early 2020 it had enrolled 570,365 patients across 26 districts in five states, with clinic-level BP control averaging 43 per cent [@kaur2023ihci]. Such programmes depend on patients who understand their readings, take their medicines and return for follow-up, which creates a clear need for trustworthy patient education between visits.

### Why Hypertension Guidance Is Context-Dependent

Hypertension management is highly individual. A reading of 150/95 mmHg can be routine in one person and urgent in another, depending on symptoms, age, pregnancy, kidney function and current medication, and guidelines encode this dependence explicitly: KDIGO recommends a systolic target below 120 mmHg in chronic kidney disease (CKD) where tolerated [@kdigo2021bp], diabetes changes the preferred targets and drug classes [@ada2026cvd], and the WHO guideline on pharmacological treatment excludes pregnancy altogether [@who2021pharm]. A correct answer to "what should my BP be?" therefore depends on facts that a chatbot may not have.

Guidance also differs between issuing bodies. The 2024 European Society of Cardiology (ESC) guideline introduces an "elevated BP" category and a treated systolic target of 120 to 129 mmHg where tolerated [@mcevoy2024esc], the 2023 European Society of Hypertension (ESH) guideline keeps the classic BP grades [@mancia2023esh], and the 2025 American Heart Association / American College of Cardiology (AHA/ACC) multisociety guideline sets its own thresholds [@jones2025aha]. An assistant that blends sources without saying which one it is using can give numerically inconsistent advice. These two observations shape MedBridge: dependence on patient context motivates a gate that checks whether enough is known about the patient, and divergence between sources motivates answers that cite the specific passage they rely on.

### Conversational AI as a Source of Health Information

Large language models (LLMs) have become a common first stop for health questions. In a widely cited study, licensed health professionals compared chatbot and physician replies to 195 patient questions from a public online forum and preferred the chatbot reply in 78.6 per cent of 585 evaluations, rating it higher for quality and empathy [@ayers2023physicianchatbot]; accuracy, however, was judged subjectively rather than checked against evidence, so perceived quality is not verified correctness. Reviews of LLMs in medicine and biomedicine describe large potential benefits alongside inaccuracy, fabricated content, misuse and patient-data privacy as the main concerns [@thirunavukarasu2023llmmedicine; @tian2024chatgptbiomed]. Even the strongest medical models need careful safety testing: Med-PaLM fell short of clinicians in human evaluation of harm, bias and factuality despite state-of-the-art examination scores [@singhal2023llmclinical], and Med-PaLM 2 was additionally tested on adversarial question sets [@singhal2025medpalm2]. These models are also proprietary and cloud-hosted, which makes them unsuitable for a privacy-preserving assistant that must run on a single consumer laptop.

### Risks of Large Language Models in Health Care

The literature and the earlier reviews of this project identify a recurring set of failure modes that matter when an LLM talks to patients. They are summarised below and in Table 1.1; Chapter 2 reviews the underlying work in detail.

**Hallucination.** Hallucinations may contradict the source (*intrinsic*) or add content the source cannot verify (*extrinsic*) [@ji2023survey], and LLM hallucinations are further divided into factuality and faithfulness errors, with retrieval-augmented systems showing failure modes of their own [@huang2025survey]. A medical hallucination is especially harmful when it could change a clinical decision [@kim2025medhallucination]. An automated audit found that 50 to 90 per cent of medical answers from seven LLMs were not fully supported by the sources they cited [@wu2025sourcecheckup], and a dedicated benchmark was needed to probe false confidence and fabricated references in medical models [@pal2023medhalt].

**Residual hallucination under grounding.** Supplying source material reduces fabrication but does not remove it: the best of twelve clinical RAG pipelines still hallucinated in 5.8 per cent of answers [@wolk2025ragcds], and LLM-generated clinical notes showed a 1.47 per cent hallucination rate even with the full transcript available [@asgari2025creola].

**Unsafe personalisation.** A response can be correct in general and still wrong for the person asking. The Review-1 presentation of this project gave the example of recommending a beta-blocker without knowing that the patient has asthma; likewise, BP targets and drug choices change with CKD, diabetes or pregnancy. The danger is not a fabricated fact but a correct fact applied without the patient-specific information that makes it applicable.

**Ungrounded and biased content.** Commercial chat models have reproduced debunked race-based clinical claims, with answers varying between runs [@omiye2023racebased], and swapping brand and generic drug names in medical benchmarks reduced model accuracy by about 1 to 10 per cent [@gallifant2024rabbits], which matters in India, where patients commonly name their medicines by brand.

**Missed emergencies.** Severely raised BP with symptoms such as chest pain, confusion or signs of heart failure calls for urgent assessment rather than self-management advice [@nice2019ng136; @peixoto2019acute]. A probabilistic generator gives no guarantee that such a message will be escalated, and any delay from retrieval and generation is undesirable precisely here.

**Prompt injection.** Instructions hidden in untrusted text, including retrieved text, can steer an LLM application [@greshake2023notwhat]; no single existing defence is sufficient [@liu2024formalizing], and prompt injection heads the industry list of LLM application risks [@owasp2025llmtop10].

**Privacy.** Sending health conversations, with their identifiers and clinical details, to a third-party API raises obligations under India's Digital Personal Data Protection Act, 2023 [@dpdp2023india] and conflicts with established principles for identifiable health information such as the U.S. HIPAA Privacy Rule [@hhs_hipaa_privacy_rule]; WHO guidance on AI for health likewise stresses safety, transparency and human oversight [@who2021ethicsai].

Table: Failure modes of unguarded LLM health assistants and the MedBridge safeguard that addresses each
| Failure mode | Illustrative situation | MedBridge safeguard |
|---|---|---|
| Hallucination | A dose or interaction that appears in no guideline is stated as fact | Evidence Gate before generation; answers restricted to retrieved passages; citation validation |
| Unsafe personalisation | A target or drug is recommended without knowing about CKD, pregnancy or current medicines | Context Gate (SOFT-ASK) before retrieval, with a loop-breaker after two requests |
| Out-of-scope or unanswerable question | A question about a fabricated drug, a false premise or a non-cardiac topic | Evidence Gate routes to ABSTAIN or GENERALIZE; ABSTAIN uses a pre-vetted template |
| Missed emergency | "BP is 200/120 and my chest feels tight" is answered with lifestyle advice | Deterministic emergency fast path before any LLM call; ESCALATE template |
| Brand-name confusion | A brand name common in India is not recognised as the generic drug it contains | Brand-to-generic normalisation and lexical (BM25) retrieval |
| Prompt injection | A message tries to override the system instructions | Isolation of patient text in dedicated XML blocks; gates and templates outside the generator |
| Privacy leakage | Names, phone numbers or ID numbers reach a model, a log or the database | PII redaction before any model call; local-only inference; encryption at rest; erasure by key destruction |

### Why Retrieval Alone Is Not Enough

Retrieval-augmented generation (RAG) pairs a language model with a retriever so that answers can draw on an external, updatable knowledge source [@lewis2020rag]. It is the natural starting point for a guideline-grounded assistant, and MedBridge builds on it. In its standard form, however, RAG retrieves and generates for every query: it has no explicit point at which the system can decline to answer. Later work has added retrieval evaluators and self-critique (reviewed in Chapter 2), but these mechanisms look only at the *evidence*. They do not ask whether the system knows enough about the *patient* to apply that evidence safely.

There are, in fact, two distinct reasons why a medical assistant should hold back. The first is "I do not know enough about you": the guideline may contain the answer, but the answer depends on facts the patient has not yet provided. The second is "I do not have enough evidence": the patient may have explained everything, but the knowledge base does not support a specific answer, the question rests on a false premise, or the topic lies outside cardiovascular care. The two situations call for different actions, namely asking a focused follow-up question in the first case and generalising or abstaining in the second, and they arise at different points in the pipeline, before and after retrieval respectively. MedBridge makes both decisions explicit through two separate gates, and places a deterministic emergency check in front of both.

## Problem Statement

LLM-based health assistants, including retrieval-augmented ones, generate an answer for almost every query. They do not separately establish whether the patient context is sufficient to personalise guidance, or whether the retrieved evidence is sufficient to support a specific claim, before they generate text. As a result they hallucinate unsupported clinical statements, personalise unsafely, answer questions they cannot support and may fail to escalate emergencies. In addition, most capable systems depend on cloud-hosted models, which is difficult to reconcile with the privacy obligations that apply to health conversations.

The problem addressed in this project is therefore:

*To design, implement and rigorously evaluate a locally deployable, evidence-grounded conversational assistant for hypertension and cardiovascular care that decides, before generating any answer, whether it has sufficient patient context and sufficient guideline evidence to answer safely, and that otherwise asks, generalises, abstains or escalates through deterministic and auditable paths.*

### Research Questions

The evaluation in this report is organised around four research questions.

1. **RQ1 (effect of the gate).** Does the full Dual Answerability Gate reduce the rate of hallucinated responses and the rate of unsafe answers to unanswerable questions, compared with a direct LLM, vanilla RAG and RAG with prompt-only safety instructions, when the same model, retrieval and benchmark are used?
2. **RQ2 (contribution of each gate).** What does each gate contribute on its own, as measured by an ablation in which only the Context Gate or only the Evidence Gate is active?
3. **RQ3 (extensions).** Do two variants that stay close to the original design, a calibrated Evidence Gate that combines the model's decision with quantitative evidence signals and a post-generation grounding verifier that checks each claim against its cited passage, improve on the base design, and at what cost in coverage and latency?
4. **RQ4 (feasibility).** Can the complete system, including the gates, retrieval, privacy layer and interface, run on consumer hardware with free, locally hosted models and acceptable response times?

## Objectives

The objectives of the project are listed below. Each one is stated in terms that can be checked against the implementation or measured in the evaluation; Table 1.2 shows where the evidence for each objective is reported.

1. **Dual Answerability Gate.** Implement a Context Gate before retrieval (SOFT-ASK or PROCEED, with a loop-breaker that forces GENERALIZE after two consecutive SOFT-ASK turns) and an Evidence Gate after retrieval (ANSWER, GENERALIZE, ABSTAIN or ESCALATE), both run at temperature 0 with a fixed seed, with every gate decision and its rationale recorded in an append-only audit log.
2. **Deterministic emergency handling.** Implement a rule-based emergency classifier that runs before any LLM call, handles colloquial, negated, misspelt and Indian-English phrasings of red-flag situations, completes within 10 ms per message, and escalates every emergency item of the evaluation benchmark.
3. **Curated knowledge base and hybrid retrieval.** Build a knowledge base of at least 15 authoritative cardiology and hypertension documents with a manifest recording source, licence and checksum, and retrieve from it with dense and BM25 retrieval fused by reciprocal rank fusion (k = 60) followed by cross-encoder reranking of the top 20 candidates to the top 5. Measure Recall@5, Recall@20, MRR and nDCG@5 for each retrieval configuration on a labelled query set.
4. **Grounded, cited generation with safe fallbacks.** Generate ANSWER, GENERALIZE and SOFT-ASK responses only from the retrieved passages, with inline citation markers that are validated against those passages, downgrade an answer left with no valid citation to GENERALIZE, and return pre-vetted templates without calling the generator for ABSTAIN and ESCALATE. Every internal failure must degrade to a safe response rather than an error.
5. **Privacy and security by design.** Redact personal identifiers before any model call and before storage, and report precision and recall on synthetic development and blind held-out sets; encrypt stored conversation content per session and support erasure by destroying the session key; protect sessions with signed, expiring tokens; add rate limiting, request size limits and security headers; and keep all inference on the local machine by default.
6. **Benchmark and hallucination study.** Construct MedBridge-HalluBench, a versioned synthetic benchmark of at least 240 queries across six categories with a 20 per cent development split reserved for calibration, and run a paired ablation of six conditions (C0 to C5) and two extensions (V1, V2). Judge hallucination with two independent judges, an LLM judge from a different model family and an NLI model, and report hallucination rate, unsafe-answer rate, coverage and routing macro-F1 with 95 per cent confidence intervals, exact McNemar tests with Holm correction and Cochran's Q.
7. **Local, efficient operation.** Select the application model through a model-selection experiment among open-weight models that fit in about 6 GB of GPU memory at 4-bit quantisation, and reduce response latency through model warm-up, concurrent embedding, caching and speculative retrieval that preserves the gate semantics, reporting latency before and after these optimisations.
8. **Usable, transparent interface and verified software.** Provide a browser-based chat interface with streamed progress, colour-coded action badges, a citation drawer, a "Why this answer?" panel that exposes the gate decisions, a BP log, a clinician summary export and an administrator dashboard, backed by an automated test suite with reported coverage.

Table: Objectives, measurable criteria and where the evidence is reported
| No. | Objective | Measurable criterion | Evidence |
|---|---|---|---|
| 1 | Dual Answerability Gate | Five actions with the stated routing; loop-breaker at two; gate decisions in the audit log | Chapter 3 (design); Chapter 4 (tests, routing results) |
| 2 | Emergency fast path | Runs before any LLM call; under 10 ms; recall on benchmark emergencies | Chapter 3; Chapter 4 |
| 3 | Knowledge base and retrieval | At least 15 documents with manifest; Recall@k, MRR, nDCG per configuration | Chapter 3 (corpus); Chapter 4 (retrieval evaluation) |
| 4 | Grounded generation | Citation validity; downgrade rule; template bypass; no unhandled errors | Chapter 3; Chapter 4 |
| 5 | Privacy and security | Redaction precision and recall (development and blind sets); encryption and erasure tests | Chapter 3; Chapter 4 |
| 6 | Hallucination study | Hallucination and unsafe-answer rates with confidence intervals and paired tests across C0 to C5, V1 and V2 | Chapter 3 (experimental design); Chapter 4 (results) |
| 7 | Local operation | Model-selection results; latency before and after optimisation | Chapter 4 |
| 8 | Interface and testing | Implemented features; test counts and coverage | Chapter 4; Appendices |

## Scope of the Project

### In Scope

- **Domain.** Cardiovascular care centred on hypertension: BP measurement and classification, hypertension treatment and lifestyle, hypertensive emergencies, heart failure, atrial fibrillation, coronary artery disease, dyslipidaemia and cardiovascular risk, BP in chronic kidney disease and diabetes, hypertensive disorders of pregnancy and cardiovascular prevention.
- **Nature of output.** Informational, guideline-grounded guidance written in plain language for patients, with inline citations to the source passage, together with clarifying questions, general (non-personalised) information, safe refusals and emergency referrals.
- **Users.** Adults seeking to understand their BP readings, medicines and lifestyle, and clinicians or reviewers who want to inspect how a response was produced (through the "Why this answer?" panel, the clinician summary and the administrator dashboard).
- **Knowledge.** A curated set of authoritative, publicly readable clinical guidelines, scientific statements and government patient-information pages, ingested locally from a manifest; no open web search and no unvetted literature scraping.
- **Patient context.** Structured facts extracted from the conversation (BP readings, medicines started or stopped, symptoms, demographics and laboratory results), stored as typed events from which the current patient state is rebuilt.
- **Processing location.** All language-model inference on the local machine through Ollama, with local-only mode enabled by default; development on Apple Silicon and a target deployment on a Windows laptop with a 6 GB GPU.
- **Language.** English, including casual, misspelt and Indian-English phrasing; the emergency classifier additionally recognises common Hinglish (Roman-script Hindi) red-flag expressions.
- **Data.** Synthetic conversations, vignettes and identifiers only, for development, testing and evaluation.

### Out of Scope

- Diagnosis, prescription, dose adjustment or any other individual treatment decision; the generator is instructed never to tell a patient to start, stop or change a medicine or dose, and to suggest discussing such changes with a clinician instead.
- Replacement of clinical or emergency triage services; the emergency path only directs the user to seek urgent care.
- Non-cardiovascular medicine and non-medical topics, which the system declines.
- Integration with electronic health records, BP monitors, wearables or other medical devices.
- Training or fine-tuning of foundation models; the project builds on existing open-weight models.
- Clinical validation, regulatory approval and large-scale or multi-node deployment; the system is a research prototype.
- Processing of real patient data.

## Expected Outcomes

At the end of the project the following outcomes are expected:

- A working, end-to-end prototype of MedBridge in which every message is routed to exactly one of the five actions (ANSWER, SOFT-ASK, GENERALIZE, ABSTAIN, ESCALATE) through the emergency fast path and the two gates, with every decision explained and audited.
- An empirical answer to RQ1 and RQ2 from a paired, statistically tested ablation on MedBridge-HalluBench, reported transparently whatever its direction. If the full gate does not significantly reduce hallucination or unsafe answers relative to vanilla RAG and prompt-only safety, the analysis explains why, and the calibrated gate and grounding-verifier extensions are evaluated under the same protocol (RQ3).
- Measured evidence for the supporting components: retrieval quality per configuration, emergency-classifier behaviour, PII redaction precision and recall, routing accuracy on a separate vignette benchmark, robustness to brand and generic drug names, resistance to prompt injection, load behaviour and latency before and after optimisation (RQ4).
- Reusable research artefacts: the HalluBench dataset with its development and test splits, the evaluation harness with two judges and the statistical analysis, a human annotation sheet for judge validation, and a curated corpus manifest.
- A privacy-preserving deployment that runs on a single consumer laptop with free, local models, together with documentation for installing and running it on Windows and macOS.
- A research paper based on this work, prepared for submission to a peer-reviewed venue.

## Organisation of the Report

The rest of this report is organised as follows. **Chapter 2** reviews the literature and patents on retrieval-augmented generation, hallucination, abstention and self-correcting generation, medical language models and benchmarks, evaluation with LLM judges and entailment models, retrieval components, prompt-injection security and health-data privacy, compares existing approaches with MedBridge and identifies the research gap. **Chapter 3** describes the methodology and system design: the architecture and its eight-step decision workflow, the individual modules, the gate and emergency algorithms, the knowledge base, the privacy layer, the tools used and the design of the experiments. **Chapter 4** presents the implementation, testing and validation, and the results of the retrieval evaluation, model selection, hallucination study and supporting benchmarks, followed by an analysis of the findings and their limitations. **Chapter 5** summarises the work, states the main findings and conclusions, and outlines limitations and future enhancements. The report ends with the references and appendices containing additional results, screenshots, test cases, dataset descriptions and the individual contribution statement.
