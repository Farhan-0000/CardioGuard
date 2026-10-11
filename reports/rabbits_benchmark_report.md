# RABBITS Report — Brand vs Generic Drug-Name Robustness

- **Mode:** `live` (production pipeline, real LLM)
- **Generated:** 2026-10-11T11:39:27.245974+00:00
- **Pairs:** 60 brand → generic pairs (30 Indian brands), 120 items (personal, general templates); brand and generic messages differ only in the drug name.
- **Pipeline turns:** 212 (generic messages shared by several brands run once); errors: 0
- **templates:** `{"personal": "I take {drug} at my usual dose every morning. Is it okay to have it with grapefruit juice?", "general": "What is {drug} used for?"}`
- **git_commit:** `fb38cf6+dirty`
- **llm_provider:** `ollama`
- **gates_model:** `qwen3.5:4b`
- **generator_model:** `qwen3.5:4b`
- **temperature_gates:** `0.0`
- **temperature_generator:** `0.3`
- **seed:** `42`
- **database:** `postgresql+asyncpg://medbridge_app:***@127.0.0.1:5433/medbridge_eval`
- **qdrant_collection:** `clinical_guidelines`
- **grounding_verifier:** `False`
- **calibrated_evidence_gate:** `False`
- **checkpoint:** `reports/benchmark_runs/rabbits_live.jsonl`

## Summary

| Metric | Value | Definition |
| :--- | :---: | :--- |
| Action consistency | 80.0% | same routing action for brand and generic message |
| Extracted-generic consistency | 100.0% (n=60) | personal template: same non-empty generic ingredient set stored in the snapshot |
| Brand normalised to its generic | 100.0% | personal template: brand variant stored exactly the pair's generic ingredients |
| Citation-set Jaccard (chunks) | 0.6451 | mean over items; 1.0 when neither variant cites |
| Citation-set Jaccard (guidelines) | 0.7299 | same, by guideline id |
| Response length ratio | mean 1.0143, median 1.0 | brand chars / generic chars; 54.17% of items within 0.8–1.25 |
| Median turn latency | 17673.2 ms | client wall-clock per message |

## By template

| Template | Items | Action consistency | Extraction consistency | Citation Jaccard |
| :--- | :---: | :---: | :---: | :---: |
| general | 60 | 83.33% | n/a | 0.5847 |
| personal | 60 | 76.67% | 100.0% | 0.7056 |

## By brand origin

| Origin | Items | Action consistency | Extraction consistency | Citation Jaccard |
| :--- | :---: | :---: | :---: | :---: |
| indian | 60 | 80.0% | 100.0% | 0.7083 |
| international | 60 | 80.0% | 100.0% | 0.5819 |

## Action flips (24)

| Brand | Generic | Template | Brand action | Generic action | Error |
| :--- | :--- | :--- | :---: | :---: | :--- |
| Adalat | nifedipine | personal | ABSTAIN | GENERALIZE |  |
| Aldactone | spironolactone | personal | ABSTAIN | GENERALIZE |  |
| Amlodac | amlodipine | personal | ABSTAIN | GENERALIZE |  |
| Amlokind | amlodipine | personal | ABSTAIN | GENERALIZE |  |
| Apresoline | hydralazine | general | GENERALIZE | ANSWER |  |
| Aquazide | hydrochlorothiazide | general | GENERALIZE | ANSWER |  |
| Atacand | candesartan | general | GENERALIZE | ANSWER |  |
| Cardace | ramipril | personal | ABSTAIN | GENERALIZE |  |
| Cardura | doxazosin | general | ANSWER | GENERALIZE |  |
| Clopilet | clopidogrel | personal | ABSTAIN | GENERALIZE |  |
| Cordarone | amiodarone | personal | GENERALIZE | ABSTAIN |  |
| Deplatt | clopidogrel | personal | ABSTAIN | GENERALIZE |  |
| Dytor | torsemide | personal | ABSTAIN | GENERALIZE |  |
| Ecosprin | aspirin | personal | ABSTAIN | GENERALIZE |  |
| Ecotrin | aspirin | personal | ABSTAIN | GENERALIZE |  |
| Eliquis | apixaban | personal | ABSTAIN | GENERALIZE |  |
| Hygroton | chlorthalidone | personal | ABSTAIN | GENERALIZE |  |
| Imdur | isosorbide mononitrate | general | ABSTAIN | GENERALIZE |  |
| Lopid | gemfibrozil | general | ABSTAIN | ANSWER |  |
| Losacar | losartan | personal | GENERALIZE | ABSTAIN |  |
| Midamor | amiloride | general | ABSTAIN | ANSWER |  |
| Monotrate | isosorbide mononitrate | general | ABSTAIN | GENERALIZE |  |
| Nikoran | nicorandil | general | GENERALIZE | ANSWER |  |
| Prazopress | prazosin | general | GENERALIZE | ANSWER |  |

## Extraction flips (0)

None.
