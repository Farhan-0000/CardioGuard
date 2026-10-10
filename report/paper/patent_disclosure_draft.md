# Invention Disclosure and Provisional Patent Application Outline (India) — DRAFT

> **DRAFT FOR INTERNAL REVIEW ONLY. NOT LEGAL ADVICE. NOT FOR FILING AS IS.**
>
> This document was prepared by the project team, with the help of an AI assistant, so that
> the team, the guide, the institution's Technology Transfer Office (TTO) / IPR cell and a
> registered Indian patent agent or attorney can decide whether to file a provisional
> application for the MedBridge answerability-gating method. Every statement about patent
> law, forms, fees and deadlines below must be checked by the patent agent against the
> Patents Act, 1970 and the Patents Rules as amended at the time of filing. The claims are a
> starting point for the agent, not final claim language.
>
> **Keep this document confidential until a decision is made** (see Section 1).

| Item | Entry |
|---|---|
| Working title | System and method for multi-stage answerability gating of a retrieval-augmented language-model assistant |
| Short name | MedBridge Dual (optionally Triple) Answerability Gate |
| Project | MedBridge (presentation/report title: CardioGuard), B.Tech final-year project, Year 4, Semester VII |
| Students | Vijay Aravindh S. (23BAI1160); Mohammad Farhan Rashid (23BAI1529) |
| Guide | Dr. Maheswari S. |
| Institution / applicant | [TO BE FILLED: institution name, department; applicant(s) as decided by the TTO under the institution's IP policy] |
| Inventors | [TO BE FILLED: to be decided by the TTO and the patent agent — inventorship follows contribution to the claimed inventive concept, not authorship of the paper or code] |
| Contact | [TO BE FILLED: name, e-mail, phone of the responsible inventor] |
| Date of this draft | [TO BE FILLED: date] |
| Earliest public disclosure so far | [TO BE FILLED: list every talk, poster, report, preprint, public code repository or paper submission and its date — see Section 1. Record in particular the visibility of the source repository github.com/Farhan-0000/MedBridge (public or private) and, if it is or was public, the date it became public] |

---

## 1. Decisions the Team Must Take First

1. **Paper or patent, or both, and in which order.** Publishing the paper, posting a preprint,
   presenting the work in a public review or making the source repository public can each
   count as a public disclosure that destroys novelty. Section 29–31 of the Patents Act give
   only narrow exceptions; for example, Section 31(d) covers a description in a paper read
   before a learned society or published in its transactions, provided the application is
   filed within twelve months. Whether a particular conference or journal qualifies is a
   question for the patent agent. **The safest order is: file the provisional application
   first, then submit or publish.** If the paper must be submitted quickly (see
   `report/paper/VENUES.md`), ask the agent whether a provisional filing can be made
   before the submission date.
2. **Whether the subject matter is patentable in India.** Two exclusions need attention:
   - Section 3(k) excludes "a mathematical or business method or a computer programme per se
     or algorithms". A computer-related invention must be claimed and described through its
     technical effect on the computing system, as explained in the Patent Office's
     Guidelines for Examination of Computer Related Inventions (the agent should apply the
     revision in force). Section 6 below lists the technical effects we believe can be
     argued.
   - Section 3(i) excludes methods of medical treatment and diagnosis. The claims must
     therefore be directed to controlling an information-retrieval and text-generation
     system, not to diagnosing or treating a patient. MedBridge itself never diagnoses or
     prescribes, which helps.
3. **Ownership.** Under most institutional IP policies the institution is the applicant for
   work done with institutional resources and supervision. [TO BE FILLED: confirm with the
   TTO; record any funding or third-party agreements.]
4. **Use of AI tools.** Parts of the software and of this draft were produced with an AI
   coding assistant. Only natural persons can be named as inventors; tell the agent which
   parts of the concept were conceived by which team member.

## 2. Filing Package (Indian Patent Office) — Outline

| Document | Purpose | Status |
|---|---|---|
| **Form 1** — Application for grant of patent | applicant(s), inventor(s), title, type of application (provisional), address for service | [TO BE FILLED by the TTO / agent] |
| **Form 2** — Provisional specification | title, field, background, summary, detailed description, drawings; claims optional at the provisional stage but recommended in outline | Sections 3–11 of this document are the raw material |
| **Form 3** — Statement and undertaking under Section 8 | foreign filings of the same invention | [TO BE FILLED]; none known so far |
| **Form 5** — Declaration as to inventorship | required with the complete specification | later |
| **Form 26** — Authorisation of a patent agent | if an agent files | [TO BE FILLED] |
| Fee category / proof | reduced fees exist for natural persons, small entities, startups and educational institutions; the TTO confirms the category, the form (for example Form 28) and the proof needed | [TO BE FILLED] |
| Drawings | black-and-white line drawings with reference numerals, on A4 sheets, per the Patents Rules | to be redrawn from the figures listed in Section 9 |
| **Complete specification** (Form 2, complete) | must be filed within **12 months** of the provisional filing date (Section 9(1)), with claims and abstract | later |
| Request for examination (Form 18), optional early publication (Form 9) | time limits are set by the Rules (they were shortened in the 2024 amendment); the agent confirms | later |

The application can be filed online through the Patent Office's e-filing portal by the
agent or the TTO.

---

## 3. Title of the Invention

System and method for multi-stage answerability gating of a retrieval-augmented
language-model assistant, with a deterministic pre-model emergency override, a pre-retrieval
context-sufficiency gate, a post-retrieval evidence-sufficiency gate and template bypass for
refusal and escalation outputs.

## 4. Field of the Invention

The invention relates to computer systems that generate natural-language responses with a
language model conditioned on documents retrieved from a curated store, and more
particularly to a computer-implemented control architecture that decides, through separate
and auditable decision stages, whether such a system may produce a generated answer at all,
must request missing information, must restrict itself to general information, must refuse,
or must escalate, in safety-critical information services such as patient-facing health
information.

## 5. Background and Prior Art

Language models can produce fluent statements that no source supports. Retrieval-augmented
generation (RAG) conditions the model on retrieved passages but, in its usual form, still
produces an answer for every input. Known improvements either (a) train the generator to
decide when to retrieve and to critique its own output, (b) score the retrieved documents
and refine or replace them, (c) check or revise text after it has been generated, or (d)
estimate a single confidence value to answer or refuse. In patient-facing health
information these approaches leave three situations unhandled as distinct cases: a question
that becomes answerable only once user-specific facts are supplied; a question that the
curated evidence cannot support (non-existent drugs, false premises, unstated doses,
out-of-scope topics); and a message that describes an emergency, where any generated reply
delays care.

### 5.1 Patent documents reviewed

The following documents were found by the team on Google Patents and checked against their
full text (`report/references/patents.json`). This is **not** a professional patentability
or freedom-to-operate search.

| Document | Assignee | What it covers | Difference from this invention |
|---|---|---|---|
| US 12,142,371 B1 (granted 2024-11-12) | Hippocratic AI (Healthgpt Inc) | multi-turn patient conversation with a primary LLM and parallel control logic (in embodiments, separate LLMs) that detects triggers, inserts follow-up questions and can hand certain cases to a human professional | no deterministic pre-model emergency override; no separate pre-retrieval context gate and post-retrieval evidence gate over retrieved passages; no template bypass of refusal/escalation |
| US 12,243,646 B1 (granted 2025-03-04) | Glass Health Inc | clinician-facing decision support: one LLM maps a problem representation to topics, a RAG module draws on a curated guideline library, a second LLM writes the assessment/plan | no claimed abstain, generalise or escalate outcome when evidence is weak; no context-sufficiency gate |
| US 9,740,769 B2 (granted 2017-08-22) | IBM | question answering that adds an explicit "no answer" candidate ranked with the same evidence scoring | single no-answer candidate inside one ranking; no graded four-way decision over reranked evidence, no pre-retrieval clarification gate, no emergency path |
| US 12,505,299 B2 (granted 2025-12-23) | Casetext Inc | generated text cites passage identifiers; each portion is sent back to an LLM with its cited passages to verify support | verification after generation only; no decision before generation whether to answer at all |
| US 12,387,049 B2 (granted 2025-08-12) | Intuit Inc | reference-free hallucination detection by semantic consistency across rephrased prompts and two LLMs | no retrieved evidence, no gating of the response route |
| US 12,038,918 B1 (granted 2024-07-16) | Intuit Inc | LLM decides whether a query is ambiguous and helps phrase a clarifying question | triggered by linguistic ambiguity of a general query, not by missing user-specific facts that change the safe answer; no post-retrieval evidence gate, no emergency override, no bounded clarification loop with forced fallback |

### 5.2 Non-patent literature (selection)

Self-RAG (reflection tokens in a fine-tuned generator), Corrective RAG (retrieval evaluator
with corrective actions), Chain-of-Verification, RARR and SelfCheckGPT (post-hoc
verification), selective classification and conformal methods (single answer-or-refuse
confidence), and MedRAG/MIRAGE (medical RAG benchmarks that always require an answer). Full
entries are in `report/references/references.json`.

### 5.3 Recommended further search (for the agent)

Search InPASS (Indian Patent Advanced Search System), WIPO PATENTSCOPE, Espacenet and Google
Patents, including IPC/CPC classes G06F 40/xx, G06F 16/33xx, G06N 3/xx, G06N 5/xx and
G16H 50/20, G16H 80/00, with query strings such as: "answerability" AND ("retrieval
augmented" OR RAG); ("clarifying question" OR "missing information") AND "language model"
AND (retrieval OR evidence); ("abstain" OR "refuse") AND "evidence sufficiency"; ("emergency"
OR "red flag") AND "rule-based" AND "before" AND "language model"; "template" AND
("abstain" OR "escalate") AND "language model".

## 6. Summary of the Invention

A computer system receives a user's natural-language message and routes it to exactly one
of five outputs: a specific grounded answer (ANSWER), a request for missing user-specific
information (SOFT-ASK), general information explicitly marked as such (GENERALIZE), a refusal
(ABSTAIN) or an emergency escalation (ESCALATE). The route is decided by separate stages at
fixed positions:

1. a **deterministic emergency stage** applied to the raw message before any language model
   is invoked, which on detection returns a pre-stored escalation response;
2. a **first answerability gate** (context gate) executed **before retrieval results are
   used**, which decides on the basis of a deterministically projected user state whether
   user-specific facts that would change the safe answer are missing, and if so requests
   them, with a counter that forces a general-information route after a predetermined number
   of consecutive requests;
3. a **second answerability gate** (evidence gate) executed **after retrieval**, which decides
   on the basis of the retrieved passages between answer, general information, refusal and
   escalation; and
4. **template bypass**: refusal and escalation outputs are pre-stored texts and never pass
   through the generative model, while answer and general-information outputs are generated
   from the passages only, with citation validation that downgrades an unsupported answer.

Optional stages add further checks: a **calibrated evidence gate** (repeated gate decisions
over deterministic permutations of the passage order with a conservative tie-break, plus
calibrated thresholds on retrieval scores and entity coverage) and a **post-generation
grounding verifier** (entailment checking of each generated claim against its cited passages,
with removal, regeneration as general information, or withdrawal to a refusal). The calibrated
thresholds and the grounding verifier can only move the decision towards a more conservative
route, while the permutation vote replaces the single evidence-gate decision with a majority
that has a conservative tie-break and can differ from the decision on the original passage
order in either direction.

**Technical effects** (relevant to Section 3(k)): the generative model is not invoked at all
on the emergency, refusal and escalation routes, so these outputs are deterministic and
cannot contain generated content; decision stages run with deterministic decoding and a
fixed seed and are logged, so the system's routing is reproducible and auditable; retrieval
is launched concurrently with the first gate and discarded if unused, which reduces response
latency without changing the routing; every component, including the models, runs on the
local device, so no user text is transmitted; each core stage has a defined conservative
fallback (redaction failure returns a safe error and nothing is sent or stored; first-gate
failure proceeds to the second gate; second-gate failure gives general information; generator
failure gives a pre-stored text), while a failure of the emergency classifier or of the
optional grounding verifier lets the turn continue (the second gate can still escalate, and
the unverified draft is kept); and the dialogue is bounded by the clarification counter.

## 7. Objects of the Invention

1. To prevent a retrieval-augmented language-model system from producing a generated answer
   when the retrieved evidence does not support one.
2. To distinguish, and handle differently, missing user-specific context and missing
   evidence.
3. To guarantee that emergency, refusal and escalation outputs are deterministic and contain
   no generated text.
4. To bound clarification dialogues and to keep every routing decision reproducible and
   auditable.
5. To achieve the above on a single local device without transmitting user data.

## 8. Detailed Description (Preferred Embodiment)

The embodiment below is the MedBridge implementation; reference numerals refer to the
drawings listed in Section 9. Parameter values are examples, not limitations.

**8.1 System (Fig. 1).** A computing device (100) with a processor, memory and optionally a
graphics processor runs: an input interface (102); a privacy redaction module (104); a
deterministic emergency classifier (106); a state store (108) holding append-only typed
events and an audit log; an extractor (110) using a language model; a context gate (112); a
retrieval module (114) with a document store of curated guideline passages, a dense encoder,
a sparse encoder, a rank-fusion unit and a cross-encoder reranker; an evidence gate (116); a
response generator (118); a citation validator (120); a template store (122); optional
calibration (124) and grounding verifier (126) units; and an orchestrator (130) that
sequences the stages and applies the fallbacks. In the embodiment all language models run
locally (for example a 4-billion-parameter open-weight model, quantised to 4 bits), and a
local-only setting prevents any remote model provider from being contacted.

**8.2 Method (Fig. 2).**

1. *Receive and redact.* The message is received; a redacted copy, with personal identifiers
   replaced by placeholders by deterministic rules, is used by every later stage except step 2.
2. *Emergency stage.* The classifier reads the raw message. It normalises text (case,
   repeated letters, a misspelling map, a bounded edit-distance matcher for red-flag
   keywords, and a lexicon for code-mixed Hindi-English), extracts blood-pressure readings in
   many written forms with plausibility checks, and applies red-flag rules (for example a
   systolic value of at least 180 mmHg or a diastolic value of at least 120 mmHg with
   blood-pressure wording; chest pain; stroke signs; fainting; severe breathlessness). Clause
   filters suppress negated, informational and dated historical mentions. On a match, the
   pre-stored escalation text is returned and recorded; no language model is invoked.
3. *Extraction and state.* The extractor returns typed delta events (reading, medication
   added or stopped, symptom, demographic, laboratory result), a search query and an intent
   summary in a fixed schema. The user state is a deterministic fold over stored events; the
   model never writes the state directly.
4. *Speculative retrieval (Fig. 6).* Retrieval with the search query is started
   concurrently with step 5.
5. *First gate.* The context gate, at temperature 0 and a fixed seed with a structured output
   schema, returns PROCEED or SOFT-ASK with missing fields from a fixed vocabulary. Its rules
   allow SOFT-ASK only when the question concerns the user's own care, a specific personal
   fact would change the safe answer, and that fact is unknown; general, out-of-scope,
   non-existent-entity, false-premise and otherwise unanswerable questions proceed. On
   SOFT-ASK with a counter below the limit (for example 2), the retrieval result is
   discarded, a clarification request is generated for the missing fields and the counter is
   incremented. On SOFT-ASK with the counter at the limit, the route is forced to GENERALIZE
   and step 7 is skipped (Fig. 3). On PROCEED the counter is reset.
6. *Retrieval.* Dense and sparse candidate lists are fused by reciprocal rank fusion (for
   example k = 60, 20 candidates) and reranked by a cross-encoder to the top passages (for
   example 5).
7. *Second gate.* The evidence gate, at temperature 0 and a fixed seed, applies ordered
   rules over the passages: ESCALATE for an emergency described in the present; ABSTAIN for
   out-of-scope questions, irrelevant or insufficient passages, a named product absent from
   the passages, a specific dose or number the passages do not state, or paediatric or
   pregnancy medicine decisions not explicitly covered; GENERALIZE for topics covered only in
   general terms or false premises contradicted by the passages; ANSWER only for direct and
   specific support, with GENERALIZE preferred when in doubt.
8. *Output.* For ABSTAIN and ESCALATE a pre-stored text is returned without invoking the
   generator. For ANSWER and GENERALIZE the generator, at a low temperature, writes from the
   passages only, ending each clinical sentence with a marker referring to its passage. The
   citation validator removes markers that do not refer to a retrieved passage and downgrades
   an ANSWER with no valid marker to GENERALIZE.
9. *Isolation.* All user-derived text is placed inside delimited elements (for example
   `<untrusted_user_input>`), with tag-like sequences escaped, and the passages inside a
   separate element, so that instructions inside user text or passages are treated as data.
10. *Persistence.* Events, messages, the route, each gate decision and its rationale are
    written in a single transaction; the clinical events and the audit record (route, gate
    decisions and rationales) go to append-only tables; session content is encrypted with
    a per-session key whose destruction erases the session while the audit records remain.
11. *Fallbacks.* Extractor failure: no events; the redacted message is used as the intent,
    and the redacted message expanded with clinical synonyms as the search query. First-gate
    failure:
    PROCEED. Second-gate failure: GENERALIZE. Generator failure: a pre-stored
    general-information text.

**8.3 Calibrated evidence gate (Fig. 4, optional).** The second gate is run once per distinct
deterministic ordering of the passages (original, reversed, rotations; for example three
orderings), each at temperature 0. The majority route is taken, ties going to the more
conservative route in the order ESCALATE, ABSTAIN, GENERALIZE, ANSWER. Features are computed
deterministically: the maximum and mean top-3 reranker scores and the fraction of the
question's salient entities (drug names, numbers with units, clinical keywords) found in the
passages. Thresholds calibrated on held-out data then lower ANSWER to GENERALIZE when the
maximum score or the coverage is below its threshold, and lower ANSWER or GENERALIZE to
ABSTAIN when the maximum score is below a lower threshold; ESCALATE is never changed and
ABSTAIN is never raised.

**8.4 Grounding verifier (Fig. 5, optional).** The draft is split into sentences;
non-claim sentences (greetings, referrals, emergency advice) are kept. Each claim sentence is
scored by a natural-language-inference cross-encoder against windows of about three
sentences from the passages it cites (all passages when uncited); it is supported when the
highest entailment probability reaches a threshold. If the unsupported fraction is at most a
limit τ, unsupported sentences are removed (in the current implementation a draft that
removal would leave empty is kept unchanged; [TO BE FILLED: decide with the team whether that
case should return the pre-stored refusal before the specification is drafted]). Otherwise an ANSWER is regenerated as
GENERALIZE and verified again; if it still fails, or the route was already GENERALIZE, the
pre-stored refusal is returned.

**8.5 Variations.** The gates may use different models from each other or from the
generator; the document store may cover any domain; the emergency rules may be replaced by
any deterministic classifier; the number of passages, the clarification limit, the number of
orderings and the thresholds may vary; and the optional stages may be used separately or
together.

## 9. Brief Description of the Drawings (to be prepared)

| Fig. | Content | Starting material in the repository |
|---|---|---|
| 1 | Block diagram of the system (100–130) | `report/figures/architecture_pipeline.png`, `report/figures/deployment.png` |
| 2 | Flow chart of the routing method (steps 1–11) | `report/paper/figures/gate_pipeline.png`; Algorithm 1 in the paper |
| 3 | State diagram of the clarification counter and forced fallback | `report/figures/fsm_soft_ask.png` |
| 4 | Flow chart of the calibrated evidence gate (orderings, vote, thresholds) | new |
| 5 | Flow chart of the grounding verifier (split, verify, remove / regenerate / withdraw) | new |
| 6 | Timing diagram of speculative retrieval overlapping the first gate | `report/figures/sequence_turn.png` |
| 7 | Data model: append-only events and audit log, per-session keys | `report/figures/er_diagram.png`, `report/figures/privacy_flow.png` |

Patent drawings must be redrawn as black-and-white line drawings with reference numerals.

## 10. Draft Claims (for the agent to refine)

**Independent method claim**

1. A computer-implemented method of controlling responses of a retrieval-augmented
   language-model system, comprising:
   (a) receiving a natural-language message from a user;
   (b) applying to the message, before any language model is invoked for the message, a
   deterministic classifier configured to detect an emergency condition, and, when the
   emergency condition is detected, outputting a pre-stored escalation response without
   invoking any language model;
   (c) when no emergency condition is detected, invoking a first answerability gate
   comprising a language model that determines, based on a user state and the message and
   before any retrieved passage is used, whether user-specific information required for a
   safe response is missing, and outputs either a clarification decision identifying the
   missing information or a proceed decision;
   (d) in response to the clarification decision, outputting a request for the missing
   information;
   (e) in response to the proceed decision, retrieving passages from a document store and
   invoking a second answerability gate, distinct from the first answerability gate,
   comprising a language model that classifies, based on the retrieved passages, the message
   into one of at least an answer category, a general-information category, an abstain
   category and an escalate category;
   (f) when the abstain category or the escalate category is determined, outputting a
   corresponding pre-stored response without invoking a generative language model; and
   (g) when the answer category or the general-information category is determined,
   generating a response with a generative language model conditioned on the retrieved
   passages.

**Dependent claims**

2. The method of claim 1, further comprising maintaining a count of consecutive
   clarification decisions and, when the count has reached a predetermined limit and the
   first answerability gate again outputs a clarification decision, assigning the
   general-information category without invoking the second answerability gate.
3. The method of claim 1, wherein the retrieving is started concurrently with the first
   answerability gate and its result is discarded when the clarification decision is output.
4. The method of claim 1, wherein generating the response comprises inserting markers that
   refer to retrieved passages, removing markers that do not refer to a retrieved passage,
   and changing the answer category to the general-information category when no valid marker
   remains.
5. The method of claim 1, wherein the second answerability gate is invoked a plurality of
   times on distinct deterministic orderings of the retrieved passages and the category is
   determined by majority, a tie being resolved towards the more conservative category.
6. The method of claim 1 or 5, further comprising computing an evidence-sufficiency signal
   comprising at least one of a reranker score of the retrieved passages and a fraction of
   entities of the message present in the retrieved passages, and changing the category only
   towards a more conservative category when the signal is below a threshold determined on
   held-out data.
7. The method of claim 1, further comprising, after generating the response and before
   outputting it, determining for each claim sentence an entailment score against the
   passages it cites, removing unsupported claim sentences when their fraction does not
   exceed a limit, and otherwise regenerating the response in the general-information
   category or outputting a pre-stored abstain response.
8. The method of claim 1, wherein the deterministic classifier extracts blood-pressure
   values and detects the emergency condition when a systolic value or a diastolic value
   reaches a threshold, and suppresses symptom mentions that are negated, informational or
   historical within the same clause.
9. The method of claim 1, wherein the user state is computed by a deterministic projection of
   typed events extracted from user messages by a language model and stored in an
   append-only store, the language model not writing the user state directly.
10. The method of claim 1, wherein the language models of the first and second answerability
    gates are executed with deterministic decoding and a fixed seed and return outputs
    conforming to a predefined schema, and each decision with its rationale is stored in an
    append-only audit record.
11. The method of claim 1, wherein user-derived text is enclosed in delimited elements with
    tag-like sequences escaped before being provided to any language model.
12. The method of claim 1, wherein personal identifiers are removed from the message before
    any language model is invoked, while the deterministic classifier operates on the
    unredacted message.
13. The method of claim 1, wherein a failure of the first answerability gate results in the
    proceed decision, a failure of the second answerability gate results in the
    general-information category, and a failure of the generative language model results in
    a pre-stored general-information response.
14. The method of claim 1, wherein all language models are executed on the device that
    receives the message and the message is not transmitted to any remote model provider.

**Independent system and medium claims**

15. A system comprising one or more processors and memory storing instructions that, when
    executed, cause the system to perform the method of any one of claims 1 to 14.
16. A non-transitory computer-readable medium storing instructions that, when executed by a
    processor, cause the processor to perform the method of any one of claims 1 to 14.

*Notes for the agent:* consider whether "language model" should be generalised to "trained
machine-learning model"; whether claim 1 should require the clarification counter (claim 2)
or the template bypass to distinguish US 12,038,918 B1 and US 12,142,371 B1 more clearly;
and how to phrase the technical effect for Section 3(k).

## 11. Abstract (draft, at most 150 words)

A system and method control a retrieval-augmented language-model assistant through
separate answerability stages. A deterministic classifier screens each message before any
model is invoked and returns a pre-stored escalation response on detecting an emergency. A
first gate decides, before retrieved passages are used, whether user-specific information
needed for a safe answer is missing and, if so, requests it, with a counter that forces a
general-information response after repeated requests. A second gate classifies the
retrieved passages into answer, general-information, abstain or escalate. Abstain and
escalate responses are pre-stored and bypass the generative model; answers are generated from
the passages with validated citation markers, and unsupported answers are downgraded.
Optional stages vote over passage orderings with a conservative tie-break, apply calibrated
evidence thresholds that can only lower the decision, and verify generated claims by
entailment. All models may run locally.

## 12. Supporting Evidence and Status

- Implementation: `medbridge/core/orchestrator.py` (routing, fallbacks, speculative
  retrieval, template bypass, citation validation), `medbridge/core/emergency_classifier.py`,
  `medbridge/ai/context_gate.py`, `medbridge/ai/evidence_gate.py`,
  `medbridge/ai/evidence_calibration.py`, `medbridge/ai/grounding_verifier.py`,
  `medbridge/retrieval/`, `medbridge/state/`, `medbridge/privacy/`; design decisions in
  `docs/architecture_decision_log.md`; dated history in the git log.
- Evaluation: the MedBridge-HalluBench study (`evaluation/`) is in progress; results, when
  available, are in `evaluation/results/summary.json`. **Do not state any performance claim
  in the specification until those results exist**, and then only as reported there.
- Related publication draft: `report/paper/` (see Section 1 on disclosure order).

[TO BE FILLED: TTO decision, agent's comments, filing date and application number.]
