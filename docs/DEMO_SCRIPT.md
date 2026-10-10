# MedBridge demo and viva script (Review 4 and Review 5)

A 10-minute live demonstration for the panel, followed by likely viva questions with short answers. Every result number in this file is filled in from the result files by `scripts/render_docs_results.py`; run it again on the morning of the review. Where it still says *pending*, the experiment has not produced its file: say so plainly and do not quote a number from memory.

Presenters: [TO BE FILLED: who presents which step] (suggestion: one student drives the laptop for steps 1–4, the other for steps 5–8 and the results; swap for the viva so both answer questions).

## 1. Setup checklist (30 minutes before)

Machine:

- [ ] Laptop on mains power, performance power mode, notifications off, other GPU-heavy apps closed.
- [ ] Docker Desktop running. `python -m medbridge.cli doctor` shows no failures.
- [ ] `.env` has `LOCAL_ONLY_MODE=true`, `WARMUP_ON_STARTUP=true`, `ADMIN_PASSWORD` set, `EMERGENCY_NUMBER=112`, `AMBULANCE_NUMBER=108`, and the model chosen in the model-selection table (`LLM_MODEL`, optionally `LLM_MODEL_GATES` / `LLM_MODEL_GENERATOR`).
- [ ] `GROUNDING_VERIFIER_ENABLED` and `EVIDENCE_GATE_CALIBRATION_ENABLED` set the way the report describes the production configuration ([TO BE FILLED: final values after the study]).
- [ ] Start the server at least 5 minutes early with `python -m medbridge.cli up`, so warm-up loads the embedders, the reranker and the Ollama model. Open <http://127.0.0.1:8000/health> and check `status: healthy` with every entry of `models_loaded` true.

Browser (one window, zoom about 125 %, light theme; switch to dark once during step 8):

- [ ] Tab 1: <http://127.0.0.1:8000> (accept the first-visit privacy notice beforehand).
- [ ] Tab 2: <http://127.0.0.1:8000/dashboard>, already logged in.
- [ ] Tab 3: the README "Results" section or the report's Chapter 4.
- [ ] Tab 4: the report PDF or the slides, opened at the architecture figure (`report/figures/architecture_pipeline.png`).

Rehearsal run (this is also the fallback):

- [ ] Run every step below once, each in its own conversation, and rename them "Backup 1 – emergency", "Backup 2 – soft-ask", and so on. If a live step is slow, open the matching backup from the sidebar: its answer and its "Why this answer?" panel are stored. Do this on the day of the review: session tokens expire after `SESSION_TOKEN_TTL_SECONDS` (7 days by default).
- [ ] For step 5, keep the backup conversation; during the demo erase only the fresh conversation you create live.
- [ ] The screenshots in `report/screenshots/` and the slides are the last fallback.

## 2. Ten-minute flow

The inputs below were chosen so that the deterministic parts are certain: the emergency, redaction, small-talk and drug-name outputs were checked against the code. Steps that depend on the language model say "expected"; if the model routes differently, show the "Why this answer?" panel and explain the decision rather than retrying.

### 0:00–1:00 · Problem and idea (slides)

- Generic chatbots invent doses and personalise without the facts that matter. Retrieval helps but never decides *whether* to answer.
- MedBridge decides twice: Gate 1 before retrieval ("do I know enough about this patient?") and Gate 2 after retrieval ("does the evidence support an answer?"). Five possible actions. Everything runs locally.
- Point at the architecture figure: emergency fast-path first, then the two gates, templates for ABSTAIN and ESCALATE.

### 1:00–2:00 · Step 1: emergency fast-path (ESCALATE)

1. New chat. Type: `my bp is 200/125 and my chest feels tight`
   - The red emergency card appears almost instantly, with 112 / 108.
   - Switch to **Clinician** mode and open **Why this answer?**: the emergency rules `bp_crisis` and `chest_pain` fired, and no gate or LLM stage ran.
2. Type: `seene mein dard ho raha hai aur saans nahi aa rahi` (Hinglish: chest pain, can't breathe)
   - ESCALATE again; rules `chest_pain_hi`, `breathlessness_hi`.
3. Type: `no chest pain or dizziness, I just want to know what foods help lower blood pressure`
   - **Not** escalated: negation is handled. This one goes through the full pipeline (expected ANSWER or GENERALIZE, with citations).

Say: "This classifier is pure rules, runs on the raw text before any model, and the tests enforce its latency budget. It errs towards escalating."

### 2:00–4:00 · Step 2: Context Gate, then grounded answer (SOFT-ASK → ANSWER)

1. New chat. Type: `should I increase my bp tablet dose?`
   - Expected **SOFT-ASK** (amber badge) asking for the current medicine and a recent reading.
   - Point at the progress stepper: the retrieval step is greyed out. It had started speculatively, and its result was **discarded** because Gate 1 said SOFT-ASK.
   - Why this answer?: Gate 1 decision, rationale and missing fields.
2. Reply: `I take amlodipine 5 mg, bp 150/96 this morning, I am 58`
   - Expected **ANSWER** (green) or GENERALIZE (blue), with `[1]`, `[2]` citation chips.
   - Click a chip: the source drawer shows guideline, organisation, year, section, page and excerpt.
   - Why this answer?: Gate 1 PROCEED (it judged the original question together with the follow-up), the top-5 passages with reranker scores, the Gate 2 decision and rationale, citation validation, and the per-stage latency bars.
   - Open **My health context** (pulse icon): amlodipine 5 mg with its drug class, BP 150/96 with its ACC/AHA category. Say: "This snapshot is a fold over stored events; the model never writes it directly."

### 4:00–5:00 · Step 3: out of scope and unknown drug (ABSTAIN)

1. New chat. Type: `what is the best antibiotic for a UTI?`
   - Expected **ABSTAIN** (grey): a fixed template, and the Why panel shows the generator was skipped.
2. Type: `is Cardiovexa 20 mg safe to take with my amlodipine?` ("Cardiovexa" is made up for this demo and appears nowhere in the corpus, prompts or benchmark.)
   - Expected ABSTAIN or a GENERALIZE that says the drug is not recognised. What matters is that no dose or safety claim is made for the invented drug.

### 5:00–6:00 · Step 4: GENERALIZE and the loop-breaker

1. In any chat, type: `thanks!`
   - **GENERALIZE** template in a few milliseconds: small talk is answered by a deterministic, claim-free template, without a model call.
2. Explain the loop-breaker with the state diagram (`report/figures/fsm_soft_ask.png`): after two SOFT-ASKs in a row the third turn is forced to GENERALIZE, Gate 2 is skipped and the counter resets. It is covered by `medbridge/tests/integration/test_e2e_pipeline.py::test_loop_breaker_forces_generalize`. Only if there is time, show it live with a vague question such as `what dose should I take?` and two vague replies; this depends on the model asking twice.

### 6:00–7:00 · Step 5: privacy (redaction and erasure)

1. New chat. Type: `my name is Ravi Kumar, phone 9876543210, my bp is 138/88 is that ok?`
   - The message gets a **PII redacted (2)** chip. The stored and model-visible text is `my name is [NAME], phone [PHONE], my bp is 138/88 is that ok?`; the BP value is untouched.
   - Dashboard tab: the newest audit entry shows only the redacted preview.
2. Open **My health context** → **Clear my data** → confirm.
   - The server destroys the session's encryption key (crypto-shredding). The audit rows stay, append-only but unreadable, and the conversation now reports that it is gone.

### 7:00–8:00 · Step 6: brand names, BP log and clinician summary

1. New chat. Type: `I take Telma 40 every morning. Is it okay with grapefruit juice?`
   - My health context lists **telmisartan** (brand Telma, angiotensin receptor blocker): Indian brand names are normalised to generics before retrieval.
2. In the BP log, **Add reading** `132/84`, then `145/92`. The trend chart shows the points against the ACC/AHA bands.
3. **Clinician summary** downloads a cited Markdown summary of the session.

### 8:00–9:00 · Step 7: dashboard

Dashboard tab: total turns and action distribution, Gate 1 and Gate 2 decision counts, safety-mechanism counters (fallbacks, loop-breaker, citation downgrades), median latency by action and by stage, feedback counts, and the audit feed with redacted request and response previews. Toggle dark mode once on the chat tab.

### 9:00–10:00 · Step 8: results and limits (README "Results" or report Chapter 4)

- Hallucination rate, C5 vs vanilla RAG: <!-- RESULT:test:C5:C1:hallucination:llm -->*pending* (result file not generated yet)<!-- /RESULT -->.
- Unsafe answers on unanswerable questions, C5 vs vanilla RAG: <!-- RESULT:test:C5:C1:unsafe:llm -->*pending* (result file not generated yet)<!-- /RESULT -->.
- Judge agreement: <!-- RESULT:kappa:3class -->*pending* (result file not generated yet)<!-- /RESULT -->.
- Retrieval MRR, hybrid: <!-- RESULT:retrieval:hybrid:mrr -->0.740 [0.650, 0.825] (95% CI)<!-- /RESULT -->.
- PII redaction on the blind set: <!-- RESULT:pii:blind_v2:strict -->precision 0.944, recall 0.744, F1 0.832 (strict span match, 203 gold spans)<!-- /RESULT -->.
- Close with two limitations (synthetic benchmark without clinician review; small local models still misroute) and one next step.

## 3. If something goes wrong

| Problem | What to do on stage |
|---|---|
| An LLM turn takes longer than about 45 s | Keep talking through the live stepper (each stage has a label). If it is still running, open the matching "Backup" conversation from the sidebar and continue from its stored answer and Why panel. |
| Model unloaded (first turn slow again) | Before the panel enters, send one throwaway question to reload it. `LLM_KEEP_ALIVE` keeps it resident afterwards. |
| Ollama crashed or out of memory | The emergency fast-path, small talk, redaction, BP log, summary and dashboard still work without the model. An LLM turn returns the safe fallback text instead of an error, which itself demonstrates the never-throw design. Restart Ollama between steps; switching to a smaller model in `.env` needs a server restart, so do it only before the demo. |
| A step routes differently from "expected" | Open Why this answer? and explain which gate decided and why. That is the point of the design; do not retry until it "works". |
| Browser storage blocked | The app falls back to in-memory storage; conversations disappear on reload, so do not reload. |
| Docker or the database down | `python -m medbridge.cli up` restarts the services and runs migrations; check `/health`. |
| Nothing works | Use the screenshots in `report/screenshots/` and the slides, and explain the flow with the architecture figure. |

## 4. Likely viva questions

**Q1. Why two gates?**
They answer different questions at different times. Gate 1 runs before retrieval and asks whether a missing personal fact (medicines, a reading, pregnancy, kidney disease) would change the safe answer. Gate 2 runs after retrieval and asks whether the retrieved guideline text supports an answer. Gate 1 cannot see evidence, and no amount of evidence supplies a missing personal fact, so each gate catches a failure the other cannot. Both are logged as separate decisions with their own rationale.

**Q2. Why not a single gate after retrieval?**
A single gate would mix "I do not know enough about you" with "the guidelines do not cover this". It would also have to retrieve before it knows whether it needs to ask a question, and it would be harder to audit. The ablation tests this directly: Context Gate only (C3) and Evidence Gate only (C4) against both gates (C5).
- Hallucination, C5 vs C3: <!-- RESULT:test:C5:C3:hallucination:llm -->*pending* (result file not generated yet)<!-- /RESULT -->.
- Hallucination, C5 vs C4: <!-- RESULT:test:C5:C4:hallucination:llm -->*pending* (result file not generated yet)<!-- /RESULT -->.
- Unsafe answers, C5 vs C3: <!-- RESULT:test:C5:C3:unsafe:llm -->*pending* (result file not generated yet)<!-- /RESULT -->.
- Unsafe answers, C5 vs C4: <!-- RESULT:test:C5:C4:unsafe:llm -->*pending* (result file not generated yet)<!-- /RESULT -->.

If a comparison is not significant after Holm correction, say so: the report's decision rule then requires an error analysis and the stronger variants (V1, Gate 3), and the honest conclusion is part of the result.

**Q3. Does the gate reduce hallucination compared with plain RAG?**
- Hallucination rate (LLM judge): C5 <!-- RESULT:rate:C5:llm.hallucination -->*pending* (result file not generated yet)<!-- /RESULT -->; C1 vanilla RAG <!-- RESULT:rate:C1:llm.hallucination -->*pending* (result file not generated yet)<!-- /RESULT -->; C2 RAG with written safety rules <!-- RESULT:rate:C2:llm.hallucination -->*pending* (result file not generated yet)<!-- /RESULT -->.
- Paired test C5 vs C2 (is a gate better than just telling the model to be careful?): <!-- RESULT:test:C5:C2:hallucination:llm -->*pending* (result file not generated yet)<!-- /RESULT -->.
- Across all six conditions: <!-- RESULT:cochran:hallucination:llm -->*pending* (result file not generated yet)<!-- /RESULT -->.
- The price is coverage on answerable questions: C5 <!-- RESULT:rate:C5:coverage_answerable -->*pending* (result file not generated yet)<!-- /RESULT --> vs C1 <!-- RESULT:rate:C1:coverage_answerable -->*pending* (result file not generated yet)<!-- /RESULT -->. The risk–coverage figure shows the trade-off.

**Q4. How exactly did you measure hallucination?**
Each reply is split into atomic clinical claims, and each claim is labelled SUPPORTED, UNSUPPORTED or CONTRADICTED against the passages the system retrieved plus passages the judge retrieves independently from the same corpus, so even the no-retrieval baseline C0 is judged fairly. A reply counts as hallucinated if at least one claim is unsupported or contradicted. Template replies (ABSTAIN, ESCALATE, small talk) have no claims. For unanswerable questions we also count unsafe answers: a specific answer, an accepted false premise, an endorsed invented drug or an unsupported dose. Results come from the test split only (<!-- RESULT:split -->*pending* (result file not generated yet)<!-- /RESULT -->); the development split was used only to calibrate thresholds.

**Q5. Who are the judges and do they agree?**
Two independent judges label the same claims. The first is an LLM judge from a different model family than the application model, run at temperature 0, because models tend to favour their own outputs (`panickssery2024llm` in `report/references/references.json`). The second is a deterministic NLI cross-encoder whose thresholds were calibrated on the development split. Study models: <!-- RESULT:models -->*pending* (result file not generated yet)<!-- /RESULT -->.
- Agreement, three-class: <!-- RESULT:kappa:3class -->*pending* (result file not generated yet)<!-- /RESULT -->; binary: <!-- RESULT:kappa:binary -->*pending* (result file not generated yet)<!-- /RESULT -->.
- Human check: a blinded 50-item sheet, status <!-- RESULT:human:status -->*pending* (result file not generated yet)<!-- /RESULT -->; consensus vs LLM judge <!-- RESULT:human:consensus_vs_llm -->*pending* (result file not generated yet)<!-- /RESULT -->.

**Q6. Why local LLMs instead of GPT-4 or Groq?**
Privacy first: with `LOCAL_ONLY_MODE` no patient text can leave the laptop, which removes a Review-1 limitation (dependence on external APIs) and fits the data-minimisation principles of India's DPDP Act and HIPAA. It also costs nothing, works offline, and is reproducible because model weights are pinned by digest. The trade-off is weaker models, so the model was chosen by experiment: <!-- RESULT:model_selection -->qwen3.5:4b: lenient routing 78.0%, macro-F1 0.71, turn p50 10.3 s; gemma3:4b: lenient routing 50.8%, macro-F1 0.47, turn p50 9.3 s; qwen3:8b: lenient routing 84.7%, macro-F1 0.74, turn p50 16.0 s<!-- /RESULT -->.

**Q7. Hybrid retrieval did as well as hybrid plus reranker. Why keep the reranker?**
- Retrieval MRR: dense <!-- RESULT:retrieval:dense:mrr -->0.677 [0.584, 0.766] (95% CI)<!-- /RESULT -->, sparse <!-- RESULT:retrieval:sparse:mrr -->0.584 [0.488, 0.676] (95% CI)<!-- /RESULT -->, hybrid <!-- RESULT:retrieval:hybrid:mrr -->0.740 [0.650, 0.825] (95% CI)<!-- /RESULT -->, hybrid + reranker <!-- RESULT:retrieval:hybrid+rerank:mrr -->0.698 [0.606, 0.785] (95% CI)<!-- /RESULT -->.
- Reranker ablation: <!-- RESULT:reranker -->MRR none (hybrid RRF k=60 only) 0.740 [0.650, 0.825]; cross-encoder/ms-marco-MiniLM-L-6-v2 (default) 0.698 [0.606, 0.785]; BAAI/bge-reranker-base 0.713 [0.623, 0.799]; cross-encoder/ms-marco-MiniLM-L-12-v2 0.708 [0.619, 0.794]; ncbi/MedCPT-Cross-Encoder 0.734 [0.643, 0.819]; all 95% CIs overlap with RRF alone<!-- /RESULT -->.

Read the confidence intervals, not the point estimates: with 70 queries the intervals are wide, and the line above states whether they overlap with fusion alone. We kept the cross-encoder because it is part of the approved architecture, it costs little next to the LLM calls (see the per-stage latency bars), and its score is the quantitative evidence signal used by the calibrated Evidence Gate (V1) and to rank the risk–coverage curve. Hybrid itself is justified: BM25 catches exact drug names and dense retrieval catches lay wording.

**Q8. Walk us through the privacy design. How can you erase data from an append-only audit log?**
Redaction happens first, so models, logs and the database only ever see placeholders such as `[NAME]`; only the deterministic emergency classifier reads the raw text, and it stores nothing. Each session has its own encryption key, wrapped by a master key. "Clear my data" deletes that key and the mutable rows: the immutable audit and event rows stay in place but can no longer be decrypted. That is crypto-shredding, so the right to erasure and the append-only invariant are both kept. Access needs a signed, expiring session token, and the dashboard has its own login.
- Redaction on the blind held-out set: <!-- RESULT:pii:blind_v2:strict -->precision 0.944, recall 0.744, F1 0.832 (strict span match, 203 gold spans)<!-- /RESULT -->; clinical values: <!-- RESULT:pii:blind_v2:clinical -->0 protected clinical values corrupted, 1 of 89 PII-free messages altered<!-- /RESULT -->.
- Weak spot: names without a cue word are missed; a named-entity model is the next step.

**Q9. What does Gate 3 add?**
Gate 3, the grounding verifier, runs after generation and before the patient sees anything. Each claim sentence is checked by a local NLI model against the passage it cites. Unsupported sentences are removed; if too many are unsupported, an ANSWER is regenerated as GENERALIZE or replaced by ABSTAIN. It can only make a reply more conservative, and its thresholds come from the development split.
- Hallucination, V2 vs C5: <!-- RESULT:test:V2:C5:hallucination:llm -->*pending* (result file not generated yet)<!-- /RESULT -->.
- Coverage cost, V2 <!-- RESULT:rate:V2:coverage_answerable -->*pending* (result file not generated yet)<!-- /RESULT --> vs C5 <!-- RESULT:rate:C5:coverage_answerable -->*pending* (result file not generated yet)<!-- /RESULT -->.

**Q10. How does the emergency classifier work, and how do you know it does not miss emergencies?**
Pure rules, compiled at import time: text normalisation (Unicode, apostrophes, a misspelling map and one-edit matching for red-flag words), blood-pressure extraction with crisis thresholds (systolic 180 or diastolic 120, with BP wording), 40 rule ids including 14 for Roman-script Hindi, a negation scope ("no chest pain"), and suppression of purely informational or historical mentions. It runs on the raw text before any model call, and unit tests enforce its latency budget. When in doubt it escalates.
- Emergency recall on the benchmark: <!-- RESULT:rate:C5:emergency.recall_system -->*pending* (result file not generated yet)<!-- /RESULT -->; false escalations: <!-- RESULT:rate:C5:emergency.false_escalation_system -->*pending* (result file not generated yet)<!-- /RESULT -->.
- Known misses: spelled-out numbers ("two hundred over one twenty") and other Indian languages. Known false positives: side-effect questions that name a red flag.

**Q11. How does event sourcing work here?**
The extractor never writes patient state. It emits typed events (BP reading, medicine added or stopped, symptom, demographic, lab result). The snapshot is computed as a fold over those events by a deterministic projector, and the events are stored append-only (database rules reject UPDATE and DELETE). Under concurrent writers the turn store locks the snapshot row and re-projects it, so the snapshot always equals the fold of the events. Benefits: a full audit trail, reproducible state and a clear erasure story.

**Q12. What is RABBITS and what did you do about it?**
RABBITS (`gallifant2024rabbits` in the references) showed that language models answer differently when a drug is named by its brand instead of its generic name. Indian patients usually say brand names ("Telma 40"). We map 189 cardiovascular brands, 65 of them Indian, to generics before retrieval, and we test whether the action, the stored medicine and the citations stay the same when only the drug name changes: <!-- RESULT:rabbits -->*pending* (result file not generated yet)<!-- /RESULT -->.

**Q13. Speculative retrieval sounds like it could change the routing. Does it?**
No. Retrieval starts while Gate 1 is still deciding, because it only needs the extractor's search query. If Gate 1 says SOFT-ASK, the task is cancelled and its results never reach Gate 2 or the generator; if Gate 1 says PROCEED, the results are used and Gate 1's time is saved. Unit tests check both paths and check that the output is identical with the flag on or off.

**Q14. What happens if the model or the database fails mid-turn?**
The orchestrator never raises. The specification's fallbacks apply: extractor failure → raw message and no events; Gate 1 failure → PROCEED; Gate 2 failure → GENERALIZE; generator failure → a pre-vetted template; retrieval outage → empty evidence, recorded as failed, and Gate 2 still decides. In the study such turns are flagged as degraded and excluded from headline numbers (C5: <!-- RESULT:degraded:C5 -->*pending* (result file not generated yet)<!-- /RESULT -->).

**Q15. How do you defend against prompt injection?**
Patient text is always wrapped in `<untrusted_user_input>` (and evidence in `<clinical_evidence>`), and tag-like text, including look-alike brackets and invisible characters, is escaped so it cannot close those blocks. Outputs are schema-constrained JSON. ABSTAIN and ESCALATE never reach the generator, and the emergency check runs before any model. Live adversarial suite: <!-- RESULT:adversarial -->*pending* (result file not generated yet)<!-- /RESULT -->.

**Q16. The benchmark is synthetic and you wrote it. Isn't that biased?**
It is synthetic by design: the project never uses real patient data. To limit bias we used a stratified development/test split and report only on test, decontaminated the prompts' few-shot examples against the benchmark (enforced by a test), tagged items that overlap the prompts and report results without them, used two independent judges, and prepared a human annotation sheet. The remaining weakness, single-author labels without clinician review, is stated in the limitations.

**Q17. Is it fast enough?**
Before/after optimisation: <!-- RESULT:perf -->*pending* (result file not generated yet)<!-- /RESULT -->. Per turn in the study (C5): <!-- RESULT:latency:C5 -->*pending* (result file not generated yet)<!-- /RESULT -->. The emergency path does not wait for any model. The rest is dominated by four local model calls, which is why warm-up, speculative retrieval, caching and structured outputs were added.

**Q18. What are the main limitations?**
A synthetic, single-author benchmark without clinician review; routing errors from small local models; answer text replayed after the turn rather than truly streamed; rule-based emergency and PII detection with known gaps; thin heart-failure and coronary coverage plus an unresolved licence question for one guideline; a single-node deployment; and no clinical validation. MedBridge is a research prototype, not a medical device.

**Q19. What would you do next?**
Clinician-reviewed labels and a small user study; human annotation at scale; a named-entity model for PII; the missing guidelines (heart failure, ESC atrial fibrillation and lipids); conformal or selective abstention on gate confidence (the V3 idea); true token streaming; regional-language input; and submitting the paper ([TO BE FILLED: venue]).

**Q20. What did each of you do?**
[TO BE FILLED: each student's own answer, matching the contribution statement in the report appendix]
