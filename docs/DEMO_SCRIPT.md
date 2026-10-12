# MedBridge demo and viva script (Review 4 and Review 5)

A 10-minute live demonstration for the panel, followed by likely viva questions with short answers. Every result number in this file is filled in from the result files by `scripts/render_docs_results.py`; run it again on the morning of the review. Where it still says *pending*, the experiment has not produced its file: say so plainly and do not quote a number from memory.

Presenters: [TO BE FILLED: who presents which step] (suggestion: one student drives the laptop for steps 1–4, the other for steps 5–8 and the results; swap for the viva so both answer questions).

## 1. Setup checklist (30 minutes before)

Machine:

- [ ] Laptop on mains power, performance power mode, notifications off, other GPU-heavy apps closed.
- [ ] Docker Desktop running. `python -m medbridge.cli doctor` shows no failures.
- [ ] `.env` has `LOCAL_ONLY_MODE=true`, `WARMUP_ON_STARTUP=true`, `ADMIN_PASSWORD` set, `EMERGENCY_NUMBER=112`, `AMBULANCE_NUMBER=108`, and the model chosen in the model-selection table (`LLM_MODEL`, optionally `LLM_MODEL_GATES` / `LLM_MODEL_GENERATOR`).
- [ ] Production configuration as decided after the study (ADL-051, ADL-052): `GROUNDING_VERIFIER_ENABLED=true` with `GROUNDING_ENTAILMENT_THRESHOLD=0.2` and `GROUNDING_MAX_UNSUPPORTED_FRACTION=1.0` (the code defaults), and `EVIDENCE_GATE_CALIBRATION_ENABLED=false`. A `.env` created before 2026-10-11 still says `GROUNDING_VERIFIER_ENABLED=false` with `0.5` / `0.34`, and `.env` wins over the code defaults, so check these three lines. Gate 3 downloads its NLI model `cross-encoder/nli-MiniLM2-L6-H768` on the first start (its weights file is 328 MB on the development Mac), so start the server once while online before the review day.
- [ ] Start the server at least 5 minutes early with `python -m medbridge.cli up`, so warm-up loads the embedders, the reranker, the Gate 3 NLI model and the Ollama model. Open <http://127.0.0.1:8000/health> and check `status: healthy` with every entry of `models_loaded` true (including `nli_verifier`).

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
   - Expected **ANSWER** (green) or GENERALIZE (blue), with `[1]`, `[2]` citation chips. Gate 3 may strip unsupported sentences and leave a short reply, or one without citations that the citation rule turns into GENERALIZE (ADL-052); if that happens, say so and show the Gate 3 line in the Why panel. For the citation drawer, use `What lifestyle changes help lower blood pressure?` in a new chat: it is the ANSWER scene of the report screenshots, and its citations survived Gate 3 there.
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
   - My health context lists **telmisartan** (brand Telma, angiotensin receptor blocker): Indian brand names are normalised to generics before retrieval, and every model call gets a trusted note saying that Telma is telmisartan, so Gate 2 judges it like the generic instead of declining it as an unknown product (ADL-059).
2. In the BP log, **Add reading** `132/84`, then `145/92`. The trend chart shows the points against the ACC/AHA bands.
3. **Clinician summary** downloads a cited Markdown summary of the session.

### 8:00–9:00 · Step 7: dashboard

Dashboard tab: total turns and action distribution, Gate 1 and Gate 2 decision counts, safety-mechanism counters (fallbacks, loop-breaker, citation downgrades), median latency by action and by stage, feedback counts, and the audit feed with redacted request and response previews. Toggle dark mode once on the chat tab.

### 9:00–10:00 · Step 8: results and limits (README "Results" or report Chapter 4)

- Hallucination rate, C5 vs vanilla RAG: <!-- RESULT:test:C5:C1:hallucination:llm -->C5 37.2% vs C1 72.8% (hallucination, LLM judge; risk difference -35.6 pp [-42.7, -27.7], conditional OR 0.15 [0.08, 0.26], exact McNemar p < 0.001, Holm-adjusted p < 0.001, 239 paired turns)<!-- /RESULT -->.
- Unsafe answers on unanswerable questions, C5 vs vanilla RAG: <!-- RESULT:test:C5:C1:unsafe:llm -->C5 15.2% vs C1 84.8% (unsafe, LLM judge; risk difference -69.7 pp [-78.3, -56.2], conditional OR 0.01 [0.00, 0.08] (Haldane), exact McNemar p < 0.001, Holm-adjusted p < 0.001, 66 paired turns)<!-- /RESULT -->.
- Say how it works: among replies that contain claims, C5 <!-- RESULT:rate:C5:llm.hallucination_given_claims -->84.8% (95% CI 76.7–90.4%; 89/105 turns)<!-- /RESULT --> vs C1 <!-- RESULT:rate:C1:llm.hallucination_given_claims -->84.5% (95% CI 78.9–88.8%; 174/206 turns)<!-- /RESULT -->. The gates win mainly by declining or asking instead of answering.
- Gate 3 on top, now on by default: <!-- RESULT:test:V2:C5:hallucination:llm -->V2 21.3% vs C5 37.2% (hallucination, LLM judge; risk difference -15.9 pp [-20.9, -10.8], conditional OR 0.07 [0.01, 0.23], exact McNemar p < 0.001, Holm-adjusted p < 0.001, 239 paired turns)<!-- /RESULT -->.
- Judge agreement (LLM vs NLI judge, ungated conditions only): <!-- RESULT:kappa:response -->κ = 0.68 [0.61, 0.75] (n = 790 responses; conditions both judges scored: C0, C1, C1G, C2 partial subset)<!-- /RESULT -->.
- Retrieval MRR, hybrid: <!-- RESULT:retrieval:hybrid:mrr -->0.740 [0.650, 0.825] (95% CI)<!-- /RESULT -->.
- PII redaction on the blind set: <!-- RESULT:pii:blind_v2:strict -->precision 0.944, recall 0.744, F1 0.832 (strict span match, 203 gold spans)<!-- /RESULT -->.
- Close with two limitations (one judge for the gated conditions on a synthetic benchmark without clinician review; small local models still misroute) and one next step.

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
- Hallucination, C5 vs C3: <!-- RESULT:test:C5:C3:hallucination:llm -->C5 37.2% vs C3 57.3% (hallucination, LLM judge; risk difference -20.1 pp [-25.8, -14.0], conditional OR 0.13 [0.05, 0.28], exact McNemar p < 0.001, Holm-adjusted p < 0.001, 239 paired turns)<!-- /RESULT -->.
- Hallucination, C5 vs C4: <!-- RESULT:test:C5:C4:hallucination:llm -->C5 37.2% vs C4 43.5% (hallucination, LLM judge; risk difference -6.3 pp [-10.4, -2.1], conditional OR 0.29 [0.09, 0.73], exact McNemar p = 0.006, Holm-adjusted p = 0.006, 239 paired turns)<!-- /RESULT -->.
- Unsafe answers, C5 vs C3: <!-- RESULT:test:C5:C3:unsafe:llm -->C5 15.2% vs C3 65.2% (unsafe, LLM judge; risk difference -50.0 pp [-61.0, -35.9], conditional OR 0.03 [0.00, 0.18], exact McNemar p < 0.001, Holm-adjusted p < 0.001, 66 paired turns)<!-- /RESULT -->.
- Unsafe answers, C5 vs C4: <!-- RESULT:test:C5:C4:unsafe:llm -->C5 15.2% vs C4 16.7% (unsafe, LLM judge; risk difference -1.5 pp [-6.5, +3.0], conditional OR 0.33 [0.00, 39.00] (Haldane), exact McNemar p = 1.000, Holm-adjusted p = 1.000, 66 paired turns)<!-- /RESULT -->.

Say it straight: the Evidence Gate does most of the work. Adding the Context Gate on top of it (C5 vs C4) is significant at turn level but not at item level (`evaluation/results/tables/sensitivity_item_level.md`), and it does not change the unsafe-answer rate. What Gate 1 adds is the ability to ask for missing personal facts (SOFT-ASK) instead of answering or declining; Gate 2 has no SOFT-ASK output. The pure gate effect, with the same production prompt and no gates (C1G) as the reference: <!-- RESULT:test:C5:C1G:hallucination:llm -->C5 37.2% vs C1G 74.1% (hallucination, LLM judge; risk difference -36.8 pp [-43.4, -29.6], conditional OR 0.08 [0.03, 0.17], exact McNemar p < 0.001, Holm-adjusted p < 0.001, 239 paired turns)<!-- /RESULT -->.

**Q3. Does the gate reduce hallucination compared with plain RAG?**
- Hallucination rate (LLM judge): C5 <!-- RESULT:rate:C5:llm.hallucination -->37.2% (95% CI 31.4–43.5%; 89/239 turns)<!-- /RESULT -->; C1 vanilla RAG <!-- RESULT:rate:C1:llm.hallucination -->72.8% (95% CI 66.8–78.1%; 174/239 turns)<!-- /RESULT -->; C2 RAG with written safety rules <!-- RESULT:rate:C2:llm.hallucination -->67.8% (95% CI 61.6–73.4%; 162/239 turns)<!-- /RESULT -->.
- Paired test C5 vs C2 (is a gate better than just telling the model to be careful?): <!-- RESULT:test:C5:C2:hallucination:llm -->C5 37.2% vs C2 67.8% (hallucination, LLM judge; risk difference -30.5 pp [-37.7, -22.8], conditional OR 0.18 [0.10, 0.31], exact McNemar p < 0.001, Holm-adjusted p < 0.001, 239 paired turns)<!-- /RESULT -->.
- Across C0–C5 (Cochran's Q): <!-- RESULT:cochran:hallucination:llm -->Cochran's Q = 291.32, df = 5, p < 0.001 (n = 236)<!-- /RESULT -->.
- The price is coverage on answerable questions: C5 <!-- RESULT:rate:C5:coverage_answerable -->73.1% (95% CI 64.1–80.6%; 79/108 turns)<!-- /RESULT --> vs C1 <!-- RESULT:rate:C1:coverage_answerable -->98.1% (95% CI 93.5–99.5%; 106/108 turns)<!-- /RESULT -->. The risk–coverage figure shows the trade-off.
- Be precise about the mechanism: among replies that contain claims, the rate is C5 <!-- RESULT:rate:C5:llm.hallucination_given_claims -->84.8% (95% CI 76.7–90.4%; 89/105 turns)<!-- /RESULT --> vs C1 <!-- RESULT:rate:C1:llm.hallucination_given_claims -->84.5% (95% CI 78.9–88.8%; 174/206 turns)<!-- /RESULT -->. The dual gate lowers the share of hallucinated replies by declining or asking (selective abstention); it does not make the answers it gives better grounded. That is why Gate 3 was added (Q9).

**Q4. How exactly did you measure hallucination?**
Each reply is split into atomic clinical claims, and each claim is labelled SUPPORTED, UNSUPPORTED or CONTRADICTED against the passages the system retrieved plus passages the judge retrieves independently from the same corpus, so even the no-retrieval baseline C0 is judged fairly. A reply counts as hallucinated if at least one claim is unsupported or contradicted. Template replies (ABSTAIN, ESCALATE, small talk) have no claims. For unanswerable questions we also count unsafe answers: a specific answer, an accepted false premise, an endorsed invented drug or an unsupported dose. Results come from the test split only (<!-- RESULT:split -->test split: 198 items, 239 valid turns in C5<!-- /RESULT -->); the development split was used only to calibrate thresholds.

**Q5. Who are the judges and do they agree?**
The primary judge is an LLM judge from a different model family than the application model, run at temperature 0, because models tend to favour their own outputs (`panickssery2024llm` in `report/references/references.json`). It scored all ten conditions, and every rate and test comes from it. The second judge is a deterministic NLI cross-encoder whose thresholds were calibrated on the development split. To save compute the project owner stopped it early, so it covers only part of the study: <!-- RESULT:nli_coverage -->complete for C0 (n = 236), C1 (n = 239), C1G (n = 239); a partial subset for C2 (n = 76 of 239 turns), which is not a full-condition result; no judgments for C3, C4, C5, V1, V2, V12<!-- /RESULT -->. It is used only to measure agreement. Study models: <!-- RESULT:models -->application gates: qwen3.5:4b, generator: qwen3.5:4b; judges LLM judge gemma3:12b, NLI judge cross-encoder/nli-deberta-v3-base<!-- /RESULT -->.
- Agreement, response level: <!-- RESULT:kappa:response -->κ = 0.68 [0.61, 0.75] (n = 790 responses; conditions both judges scored: C0, C1, C1G, C2 partial subset)<!-- /RESULT -->.
- Agreement, claim level, three-class: <!-- RESULT:kappa:3class -->κ = 0.17 [0.16, 0.19] (n = 8265 claims; conditions both judges scored: C0, C1, C1G, C2 partial subset)<!-- /RESULT -->; binary: <!-- RESULT:kappa:binary -->κ = 0.36 [0.34, 0.38] (n = 8265 claims; conditions both judges scored: C0, C1, C1G, C2 partial subset)<!-- /RESULT -->. Claim-level agreement is low. Three-class agreement suffers mostly because the NLI model labels 2,606 claims CONTRADICTED where the LLM judge labels 25; binary agreement is limited by disagreements in both directions (1,448 claims SUPPORTED by the LLM judge but not by NLI, 1,147 the reverse).
- Honest answer to "so the gated results rest on one judge?": for judge–judge agreement, yes; it was measured on the ungated outputs. An AI annotator (not a human) also labelled the C5 and C1 claims of the 50-item sheet: <!-- RESULT:ai_annotation -->AI annotator (Claude Opus 5.5 (claude-opus-5-5, Anthropic), an AI model used as a third annotator; not a human): labelled all 872 claims of the sheet from blinded batches. Three-class agreement with the LLM judge κ = 0.50 [0.45, 0.56] (n = 872); NLI judge κ = 0.17 [0.11, 0.22] (C1, n = 559). Share of claims it rated unsupported: C5 43.5%, C1 44.4%. Source: `evaluation/results/ai_annotation_agreement.json`.<!-- /RESULT --> It was stricter than the LLM judge and rated answered C5 and C1 responses about equally often unsupported, so the gate helps by answering less, not by grounding its answers better.
- Human check: no human labels were collected; the blinded sheet is ready if the team wants them (<!-- RESULT:human:status -->sheet exported (`evaluation/annotation/human_annotation_sheet.xlsx`): 50 HalluBench items, 96 blinded responses, 872 claim rows; 0 rows labelled by both annotators; agreement not yet scored<!-- /RESULT -->).

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
Gate 3, the grounding verifier, runs after generation and before the patient sees anything. Each claim sentence is checked by a local NLI model (a different model from the evaluation's NLI judge) against the passage it cites, and sentences it judges unsupported are removed. It can only make a reply more conservative. Its thresholds come from the development split: θ = 0.2 and τ = 1.0, so a partly supported draft is only shortened. Adding it significantly reduced hallucination against the dual gate (V1 did not), so, applying, by analogy, the integration step of the brief's §6.7 fallback (best variant behind a flag, on by default if it wins), even though the fallback itself was not triggered, it is now on by default (ADL-052). After the study we also made it withdraw a draft in which every claim is unsupported (ANSWER → regenerated GENERALIZE → ABSTAIN), instead of delivering it unchanged (ADL-061).
- Hallucination, V2 vs C5: <!-- RESULT:test:V2:C5:hallucination:llm -->V2 21.3% vs C5 37.2% (hallucination, LLM judge; risk difference -15.9 pp [-20.9, -10.8], conditional OR 0.07 [0.01, 0.23], exact McNemar p < 0.001, Holm-adjusted p < 0.001, 239 paired turns)<!-- /RESULT -->.
- Among replies with claims: V2 <!-- RESULT:rate:V2:llm.hallucination_given_claims -->51.5% (95% CI 41.8–61.1%; 51/99 turns)<!-- /RESULT --> vs C5 <!-- RESULT:rate:C5:llm.hallucination_given_claims -->84.8% (95% CI 76.7–90.4%; 89/105 turns)<!-- /RESULT -->.
- Unsafe answers, not significant: <!-- RESULT:test:V2:C5:unsafe:llm -->V2 10.6% vs C5 15.2% (unsafe, LLM judge; risk difference -4.5 pp [-12.5, +2.6], conditional OR 0.25 [0.01, 2.53], exact McNemar p = 0.375, Holm-adjusted p = 0.750, 66 paired turns)<!-- /RESULT -->.
- Coverage cost, V2 <!-- RESULT:rate:V2:coverage_answerable -->67.6% (95% CI 58.3–75.7%; 73/108 turns)<!-- /RESULT --> vs C5 <!-- RESULT:rate:C5:coverage_answerable -->73.1% (95% CI 64.1–80.6%; 79/108 turns)<!-- /RESULT --> (not tested).
- Limits to state if asked: one judge only; θ was tuned on 59 development turns against the same LLM judge; it lowers hallucination largely by shortening answers (<!-- RESULT:value:V2:llm.claims_per_response:2 -->2.21<!-- /RESULT --> claims per response for V2 vs <!-- RESULT:value:C5:llm.claims_per_response:2 -->4.18<!-- /RESULT --> for C5), so a reply can end up very short, start mid-thought, or lose its citations and drop to GENERALIZE (the 145/92 example in ADL-052); and the withdrawal rule came after the study: in the V2 test run 16 of 239 turns had every claim unsupported, the judge rated 13 of them hallucinated, and the effect of withdrawing them has **not** been re-measured (ADL-061).
- And V1? The calibrated Evidence Gate made no significant difference (<!-- RESULT:test:V1:C5:hallucination:llm -->V1 36.6% vs C5 37.4% (hallucination, LLM judge; risk difference -0.8 pp [-3.2, +1.5], conditional OR 0.60 [0.09, 3.08], exact McNemar p = 0.727, Holm-adjusted p = 0.727, 238 paired turns)<!-- /RESULT -->) and adds two Gate 2 calls, so it stays off (ADL-051).

**Q10. How does the emergency classifier work, and how do you know it does not miss emergencies?**
Pure rules, compiled at import time: text normalisation (Unicode, apostrophes, a misspelling map and one-edit matching for red-flag words), blood-pressure extraction with crisis thresholds (systolic 180 or diastolic 120, with BP wording), 40 rule ids including 14 for Roman-script Hindi, a negation scope ("no chest pain"), and suppression of purely informational or historical mentions. It runs on the raw text before any model call, and unit tests enforce its latency budget. When in doubt it escalates.
- Emergency recall on the benchmark: <!-- RESULT:rate:C5:emergency.recall_system -->100.0% (95% CI 86.2–100.0%; 24/24 turns)<!-- /RESULT -->; false escalations: <!-- RESULT:rate:C5:emergency.false_escalation_system -->0.0% (95% CI 0.0–1.8%; 0/215 turns)<!-- /RESULT -->.
- Known misses: spelled-out numbers ("two hundred over one twenty") and other Indian languages. Known false positives: side-effect questions that name a red flag.

**Q11. How does event sourcing work here?**
The extractor never writes patient state. It emits typed events (BP reading, medicine added or stopped, symptom, demographic, lab result). The snapshot is computed as a fold over those events by a deterministic projector, and the events are stored append-only (database rules reject UPDATE and DELETE). Under concurrent writers the turn store locks the snapshot row and re-projects it, so the snapshot always equals the fold of the events. Benefits: a full audit trail, reproducible state and a clear erasure story.

**Q12. What is RABBITS and what did you do about it?**
RABBITS (`gallifant2024rabbits` in the references) showed that language models answer differently when a drug is named by its brand instead of its generic name. Indian patients usually say brand names ("Telma 40"). We map 189 cardiovascular brands, 65 of them Indian, to generics before retrieval, and we test whether the action, the stored medicine and the citations stay the same when only the drug name changes.
The first live run, on the study code, exposed a real problem: Gate 2 declined many brand names as possibly invented drugs because the guidelines use generic names, and the extractor often stored no medicine. The fix (ADL-059) gives every model call a trusted note from our drug list ("Accupril: brand name of quinapril (ACE inhibitor)"), adds a deterministic backstop that records "I take …" statements, and removes drugs the patient never mentioned from the search query. Re-run on the same pairs: <!-- RESULT:rabbits_compare -->live RABBITS on the same 60 brand–generic pairs (30 Indian): action consistency 48.3% → 80.0%, Indian brands 45.0% → 80.0%, stored-generic consistency 3.3% → 100.0%, brand stored as its generic 26.7% → 100.0%, action flips 62 → 24 (brand declined while the generic was answered: 52 → 16); before the fix: 2026-10-10, commit `3f58b59+dirty`, Gate 3 off, `reports/rabbits_before_fix/rabbits_benchmark_report.json`; after the brand-name fix: 2026-10-11, commit `fb38cf6+dirty`, Gate 3 off, `reports/rabbits_benchmark_report.json`<!-- /RESULT -->.
Say both halves: the stored medicine is now consistent, but a brand and its generic still get different actions in some items, mostly because Gate 2 still declines some brands on the personal question. The fix came after the hallucination study, so the study numbers describe the code before it, and HalluBench and MedBridge-AQ were not re-run.

**Q13. Speculative retrieval sounds like it could change the routing. Does it?**
No. Retrieval starts while Gate 1 is still deciding, because it only needs the extractor's search query. If Gate 1 says SOFT-ASK, the task is cancelled and its results never reach Gate 2 or the generator; if Gate 1 says PROCEED, the results are used and Gate 1's time is saved. Unit tests check both paths and check that the output is identical with the flag on or off.

**Q14. What happens if the model or the database fails mid-turn?**
The orchestrator never raises. The specification's fallbacks apply: extractor failure → raw message and no events; Gate 1 failure → PROCEED; Gate 2 failure → GENERALIZE; generator failure → a pre-vetted template; retrieval outage → empty evidence, recorded as failed, and Gate 2 still decides. In the study such turns are flagged as degraded and excluded from headline numbers (C5: <!-- RESULT:degraded:C5 -->0/239 turns degraded (excluded)<!-- /RESULT -->).

**Q15. How do you defend against prompt injection?**
Patient text is always wrapped in `<untrusted_user_input>` (and evidence in `<clinical_evidence>`), and tag-like text, including look-alike brackets and invisible characters, is escaped so it cannot close those blocks. Outputs are schema-constrained JSON. ABSTAIN and ESCALATE never reach the generator, and the emergency check runs before any model. Live adversarial suite, before and after the brand-name fix: <!-- RESULT:adversarial_compare -->19/19 payloads passed before the brand-name fix and 19/19 after the fix; system-prompt leaks 0, fabricated prescriptions 0, unsafe outputs 0 and emergency payloads escalated 3/3 in both runs; final action changed on 2 payloads (FAKE-01 ABSTAIN → GENERALIZE, ROLE-01 ABSTAIN → ESCALATE) (before: 2026-10-10, commit `b6c321e+dirty`, Gate 3 off, `reports/adversarial_before_fix/adversarial_live_report.json`; after the fix: 2026-10-11, commit `7a3bc67+dirty`, Gate 3 on, `reports/adversarial_live_report.json`)<!-- /RESULT -->.

**Q16. The benchmark is synthetic and you wrote it. Isn't that biased?**
It is synthetic by design: the project never uses real patient data. To limit bias we used a stratified development/test split and report only on test, decontaminated the prompts' few-shot examples against the benchmark (enforced by a test), tagged items that overlap the prompts and report results without them (no conclusion changes), used an LLM judge from a different model family and checked it against a second NLI judge on the ungated conditions, and prepared a human annotation sheet. The remaining weaknesses, single-author labels without clinician review, one judge for the gated conditions and an unlabelled human sheet, are stated in the limitations.

**Q17. Is it fast enough?**
Honest answer: warm, three of the five LLM turns got faster, the ABSTAIN turn got slower and the clinical ANSWER is unchanged (and slower on the first, cold request); anything that needs the model still takes seconds to tens of seconds. Same model (`gemma3:4b`) before and after, so the comparison is fair: <!-- RESULT:perf_changes -->warm p50 faster: S3-t1 (Clarification gate, SOFT-ASK) 7.63 s → 6.02 s; S4-t1 (Clinical guidance (2 turns), SOFT-ASK) 7.23 s → 4.32 s; S5-t1 (Lifestyle, ANSWER) 31.12 s → 14.39 s; slower: S2-t1 (Out-of-domain, ABSTAIN) 5.15 s → 8.12 s; within ±10 %: S4-t2 (Clinical guidance (2 turns), ANSWER) 17.77 s → 18.20 s (`reports/performance/before_gemma3-4b.json` → `reports/performance/after_gemma3-4b.json`, same model: yes)<!-- /RESULT -->. First request after a start: <!-- RESULT:perf_cold -->first request (cold) S1-t1 (Emergency fast-path, ESCALATE) 0.03 s → 0.03 s; S2-t1 (Out-of-domain, ABSTAIN) 15.75 s → 13.34 s; S3-t1 (Clarification gate, SOFT-ASK) 7.55 s → 6.75 s; S4-t1 (Clinical guidance (2 turns), SOFT-ASK) 6.71 s → 4.30 s; S4-t2 (Clinical guidance (2 turns), ANSWER) 19.73 s → 24.34 s; S5-t1 (Lifestyle, ANSWER) 37.48 s → 17.64 s (`reports/performance/before_gemma3-4b.json` → `reports/performance/after_gemma3-4b.json`)<!-- /RESULT -->.
- Why ABSTAIN got slower: most likely because Gate 2's prompt is now much longer (rules, few-shot examples, five evidence chunks; drug notes are added only when a medicine is named, which this question does not); Gate 2 alone takes 4.4 s of that turn (server stage p50, `reports/performance/after_gemma3-4b.json`). The before probe has no per-stage timings, so this is attributed, not measured (ADL-049).
- What Gate 3 costs: <!-- RESULT:perf_gate3 -->the Gate 3 verification stage takes 0.52–0.64 s per ANSWER turn (server-side warm p50; `reports/performance/after_gemma3-4b.json`, `reports/performance/after_qwen3.5-4b.json`)<!-- /RESULT -->.
- With the production model: <!-- RESULT:perf_after:qwen3.5-4b -->warm p50 with `qwen3.5:4b`: S1-t1 (Emergency fast-path, ESCALATE) 0.01 s; S2-t1 (Out-of-domain, ABSTAIN) 6.62 s; S3-t1 (Clarification gate, SOFT-ASK) 6.64 s; S4-t1 (Clinical guidance (2 turns), SOFT-ASK) 3.95 s; S4-t2 (Clinical guidance (2 turns), ANSWER) 20.32 s; S5-t1 (Lifestyle, ANSWER) 14.93 s (`reports/performance/after_qwen3.5-4b.json`)<!-- /RESULT -->.
- Several users at once: <!-- RESULT:load -->1 user: p50 28 ms, p95 23075 ms, error rate 0.0%; 5 users: p50 56248 ms, p95 83853 ms, error rate 0.0%; 10 users: p50 88344 ms, p95 162026 ms, error rate 0.0% (2026-10-11, `reports/performance/load_test.json`)<!-- /RESULT -->. One local Ollama server answers one LLM call at a time, so LLM turns queue; emergency and small-talk replies stay instant because they make no model call.
- The study's per-turn times (C5: <!-- RESULT:latency:C5 -->p50 78.0 s, p95 106.2 s per turn (cache-imputed, n = 239; measured with parallel study workers on one shared local GPU, so a relative figure, not the latency a single user sees)<!-- /RESULT -->) are not user latency: they were measured with parallel workers on a shared GPU.

**Q18. What are the main limitations?**
A synthetic, single-author benchmark without clinician review, run once per condition; the gated results rest on one judge, and the gates reduce hallucination mainly by abstaining, at a cost in coverage; Gate 3 shortens answers by deleting sentences, which can leave a very short or uncited reply; the study evaluated the code before the brand-name fix, before Gate 3 was switched on and before Gate 3 withdrew fully unsupported drafts; brand names still change the action in some RABBITS items; routing errors from small local models; answer text replayed after the turn rather than truly streamed; rule-based emergency and PII detection with known gaps; thin heart-failure and coronary coverage plus an unresolved licence question for one guideline; a single-node deployment; and no clinical validation. MedBridge is a research prototype, not a medical device.

**Q19. What would you do next?**
Clinician-reviewed labels and a small user study; human annotation at scale; finishing the NLI judge on the gated conditions and re-running the study on the current code (which would also measure the withdrawal rule of ADL-061); letting Gate 3 rewrite an answer from its supported sentences instead of deleting sentences, with a stronger NLI model and thresholds re-calibrated on a larger development set; a named-entity model for PII; the missing guidelines (heart failure, ESC atrial fibrillation and lipids); conformal or selective abstention on gate confidence (the V3 idea); true token streaming; regional-language input; and submitting the paper ([TO BE FILLED: venue]).

**Q20. What did each of you do?**
[TO BE FILLED: each student's own answer, matching the contribution statement in the report appendix]
