# MedBridge-AQ vignette changes (finalisation round, task I2)

All vignettes are synthetic. Before: 225 vignettes; after: 275 (50 new).
Every change below is checked by `medbridge/tests/unit/test_benchmark_runner.py`.

## 1. Citation labels now name corpus documents

`expected_citations` lists the acceptable sources for an ANSWER. Labels must
identify a document in `data/corpus_manifest.json`, in one of three forms
(see `tests/benchmark/citations.py`):

* `"ORG YEAR"` equal to the citation's `org` + `year` (the pipeline's own
  source label), e.g. `ESC 2024`, `AHA/ACC 2025`, `KDIGO 2021`, `ACC/AHA 2023`;
* a `guideline_id`, e.g. `NICE_NG133`, `CCD_GUIDELINE_REVIEW_2024` — used when
  `ORG YEAR` is ambiguous (NICE 2023 is both NG133 and NG238);
* an organisation name contained in `org`, e.g. `MedlinePlus`.

Citation precision matches returned citations on `org`/`year`/`title`/`guideline_id`
(not on the free-text source string). `test_every_expected_citation_label_names_a_corpus_document`
fails if a label names no corpus document.

| Old label | Occurrences | New label | Reason |
| :--- | :---: | :--- | :--- |
| `ESC/ESH 2024` | 26 | `ESC 2024` | corpus org/year of `ESC_HTN_2024` |
| `KDIGO 2024` (ANS-003) | 1 | `KDIGO 2021` | the corpus has the 2021 KDIGO BP-in-CKD guideline |
| `ACOG 2024` (CONTRA-001, -006, -011, -018, -023, -026) | 6 | `NICE_NG133` | ACOG is not in the corpus; NICE NG133 (hypertension in pregnancy) covers ACE inhibitor/ARB risk (rec 1.3.2), labetalol/nifedipine (rec 1.3.10) and pre-conception advice |
| `ACOG 2024` (CONTRA-030, atenolol in pregnancy) | 1 | `ESC 2024` | NG133 only mentions atenolol post-natally; ESC 2024 states atenolol should be avoided in pregnancy (foetal growth restriction) |
| `ADA 2025` (ANS-032) | 1 | removed | ADA is not in the corpus; the vignette already lists `ESC 2024`, which recommends SGLT2 inhibitors |

`MedlinePlus` (16 occurrences) and `AHA/ACC 2025` (74) already matched corpus documents.

## 2. CONTRA-025 expected action: ANSWER → ESCALATE

The query reports BP 190/110. The deterministic emergency fast-path escalates
any systolic reading >= 180 mmHg (rule `bp_crisis`) before any LLM call, so the
correct expected action is ESCALATE; `expected_citations` is now empty and the
rationale explains why. `test_vignettes_agree_with_the_deterministic_emergency_rule`
asserts that no vignette expects a non-ESCALATE action for a query the
fast-path escalates (CONTRA-025 was the only conflict).

## 3. New vignettes (natural / casual phrasing, broader cardiology scope)

| Subset | New ids | Count | Topics |
| :--- | :--- | :---: | :--- |
| complete_context_answer | ANS-047 … ANS-056 | 10 | HFrEF therapy, AF anticoagulation and rate control, post-MI lipids and antiplatelets, CKD BP target, pregnancy BP target, heart-failure salt/fluid, chronic coronary disease aspirin, diabetes + CKD SGLT2 |
| complex_comorbid_contraindications | CONTRA-031 … CONTRA-038 | 8 | mechanical valve + DOAC, statin before conception, enalapril while breastfeeding, dual RAS blockade in diabetic CKD, DOAC in pregnancy, triple antithrombotic therapy after PCI on apixaban, aspirin for pre-eclampsia, potassium salt substitutes in heart failure |
| emergency_escalate | EMERG-039 … EMERG-046 | 8 | ACS, stroke (FAST), AF with pre-syncope, acute pulmonary oedema, severe pre-eclampsia, syncope with bradycardia, thunderclap headache, rest angina after stents |
| incomplete_context_soft_ask | SOFT-043 … SOFT-050 | 8 | statins, cholesterol, palpitations, blood thinners, diuretic dose, home readings, pregnancy (SOFT-050 starts at soft_ask_count = 1) |
| lifestyle_and_generalize | GEN-036 … GEN-043 | 8 | cholesterol diet, coffee, heart-failure warning signs, resting heart rate, AF risk factors, antenatal BP checks; GEN-042/043 start at soft_ask_count = 2 (loop-breaker) |
| out_of_domain_abstain | ABST-035 … ABST-042 | 8 | veterinary, computing, tax, antibiotics, sports injury, dermatology, antidepressants, cooking |

Checks run on the new vignettes:

* every new ESCALATE vignette is escalated by the deterministic classifier, and
  no new non-ESCALATE vignette triggers it; none is caught by the small-talk detector;
* every new SOFT-ASK vignette has `soft_ask_count < 2`, every vignette with
  `soft_ask_count >= 2` expects GENERALIZE.

### Answerability of the new ANSWER vignettes

Checked by searching the parsed text of the files in `data/guidelines/`
(PyMuPDF text for PDFs, `medbridge.ingestion.html_parser` for HTML) — not by
running the retriever. The supporting passage of each vignette:

| Vignette | Document | Passage found |
| :--- | :--- | :--- |
| ANS-047 | NICE NG106 rec 1.4.1; ESC 2024 | ACE inhibitor + beta-blocker + MRA + SGLT2 inhibitor for HFrEF |
| ANS-048 | NICE NG196 rec 1.2.1, 1.6.x; MedlinePlus AFib | CHA2DS2-VASc stroke-risk score; DOAC anticoagulation; AF raises stroke risk |
| ANS-049 | NICE NG238 rec 1.7.1, 1.6.13; AHA/ACC 2018 | secondary prevention LDL <= 2.0 mmol/L; ezetimibe added to statin |
| ANS-050 | KDIGO 2021 rec 3.1.1 | SBP < 120 mmHg when tolerated, standardised office BP |
| ANS-051 | NICE NG133 rec 1.3.9 | target 135/85 mmHg on treatment in pregnancy |
| ANS-052 | NICE NG185 rec 1.4.14–1.4.15 | dual antiplatelet therapy up to 12 months after MI, aspirin indefinitely |
| ANS-053 | NICE NG196 rec 1.7.2 | standard beta-blocker or rate-limiting CCB as initial rate control |
| ANS-054 | NICE NG106 rec 1.9.10 | do not routinely restrict sodium or fluid in heart failure |
| ANS-055 | CCD guideline review 2024 | single antiplatelet therapy with low-dose aspirin 75–100 mg in CCD |
| ANS-056 | ESC 2024 | SGLT2 inhibitors recommended in hypertensive CKD with eGFR > 20 |
| CONTRA-031 | ACC/AHA 2023 AF | warfarin first-line with mechanical heart valves; DOACs not for valve recipients |
| CONTRA-032 | NICE NG238 rec 1.5.9–1.5.10 | statins contraindicated in pregnancy; stop 3 months before conceiving |
| CONTRA-033 | NICE NG133 rec 1.9.4 | offer enalapril post-natally with renal function and potassium monitoring |
| CONTRA-034 | KDIGO 2021 rec 3.3.1 | avoid any combination of ACEi, ARB and DRI in CKD |
| CONTRA-035 | ACC/AHA 2023 AF | DOACs contraindicated during pregnancy and breastfeeding |
| CONTRA-036 | NICE NG185 rec 1.4.21–1.4.22 | on anticoagulation after PCI: anticoagulant + clopidogrel up to 12 months; triple therapy raises bleeding risk |
| CONTRA-037 | NICE NG133 rec 1.1.2 | aspirin 75–150 mg from 12 weeks for women at high risk of pre-eclampsia (incl. chronic hypertension) |
| CONTRA-038 | NICE NG106 rec 1.9.11 | avoid potassium-containing salt substitutes in heart failure |
