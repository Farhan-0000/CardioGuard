# RABBITS Report — Brand vs Generic Drug-Name Robustness

- **Mode:** `live` (production pipeline, real LLM)
- **Generated:** 2026-10-10T17:47:22.731696+00:00
- **Pairs:** 60 brand → generic pairs (30 Indian brands), 120 items (personal, general templates); brand and generic messages differ only in the drug name.
- **Pipeline turns:** 212 (generic messages shared by several brands run once); errors: 0
- **templates:** `{"personal": "I take {drug} at my usual dose every morning. Is it okay to have it with grapefruit juice?", "general": "What is {drug} used for?"}`
- **git_commit:** `3f58b59+dirty`
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
| Action consistency | 48.33% | same routing action for brand and generic message |
| Extracted-generic consistency | 3.33% (n=60) | personal template: same non-empty generic ingredient set stored in the snapshot |
| Brand normalised to its generic | 26.67% | personal template: brand variant stored exactly the pair's generic ingredients |
| Citation-set Jaccard (chunks) | 0.4903 | mean over items; 1.0 when neither variant cites |
| Citation-set Jaccard (guidelines) | 0.4958 | same, by guideline id |
| Response length ratio | mean 0.8606, median 1.0 | brand chars / generic chars; 51.67% of items within 0.8–1.25 |
| Median turn latency | 24943.6 ms | client wall-clock per message |

## By template

| Template | Items | Action consistency | Extraction consistency | Citation Jaccard |
| :--- | :---: | :---: | :---: | :---: |
| general | 60 | 28.33% | n/a | 0.2722 |
| personal | 60 | 68.33% | 3.33% | 0.7083 |

## By brand origin

| Origin | Items | Action consistency | Extraction consistency | Citation Jaccard |
| :--- | :---: | :---: | :---: | :---: |
| indian | 60 | 45.0% | 0.0% | 0.4472 |
| international | 60 | 51.67% | 6.67% | 0.5333 |

## Action flips (62)

| Brand | Generic | Template | Brand action | Generic action | Error |
| :--- | :--- | :--- | :---: | :---: | :--- |
| Accupril | quinapril | personal | ANSWER | ABSTAIN |  |
| Accupril | quinapril | general | ABSTAIN | GENERALIZE |  |
| Aldactone | spironolactone | personal | ABSTAIN | GENERALIZE |  |
| Aldactone | spironolactone | general | ABSTAIN | ANSWER |  |
| Aldomet | methyldopa | personal | ABSTAIN | GENERALIZE |  |
| Aldomet | methyldopa | general | ABSTAIN | ANSWER |  |
| Alphadopa | methyldopa | personal | ABSTAIN | GENERALIZE |  |
| Alphadopa | methyldopa | general | ABSTAIN | ANSWER |  |
| Amlodac | amlodipine | general | ABSTAIN | GENERALIZE |  |
| Amlokind | amlodipine | general | ABSTAIN | GENERALIZE |  |
| Angispan-TR | nitroglycerin | personal | ABSTAIN | ANSWER |  |
| Angispan-TR | nitroglycerin | general | ABSTAIN | ANSWER |  |
| Apresoline | hydralazine | personal | ABSTAIN | ANSWER |  |
| Apresoline | hydralazine | general | ABSTAIN | GENERALIZE |  |
| Arkamin | clonidine | general | ABSTAIN | ANSWER |  |
| Aten | atenolol | personal | ANSWER | ABSTAIN |  |
| Aten | atenolol | general | ABSTAIN | GENERALIZE |  |
| Atocor | atorvastatin | personal | ABSTAIN | ANSWER |  |
| Atocor | atorvastatin | general | ABSTAIN | ANSWER |  |
| Atorlip | atorvastatin | personal | ABSTAIN | ANSWER |  |
| Atorlip | atorvastatin | general | ABSTAIN | ANSWER |  |
| Betacard | atenolol | general | ABSTAIN | GENERALIZE |  |
| Brilinta | ticagrelor | general | ABSTAIN | ANSWER |  |
| Bumex | bumetanide | general | ABSTAIN | GENERALIZE |  |
| Bystolic | nebivolol | personal | ABSTAIN | ANSWER |  |
| Bystolic | nebivolol | general | ABSTAIN | GENERALIZE |  |
| Calan | verapamil | general | ABSTAIN | GENERALIZE |  |
| Calaptin | verapamil | general | ABSTAIN | GENERALIZE |  |
| Cardace | ramipril | general | ABSTAIN | ANSWER |  |
| Clopilet | clopidogrel | general | ABSTAIN | GENERALIZE |  |
| Cordarone | amiodarone | general | ABSTAIN | ANSWER |  |
| Corlanor | ivabradine | general | ABSTAIN | ANSWER |  |
| Coumadin | warfarin | personal | GENERALIZE | ABSTAIN |  |
| Coumadin | warfarin | general | ABSTAIN | ANSWER |  |
| Deplatt | clopidogrel | personal | ANSWER | ABSTAIN |  |
| Deplatt | clopidogrel | general | ABSTAIN | GENERALIZE |  |
| Dilzem | diltiazem | general | ABSTAIN | GENERALIZE |  |
| Dytor | torsemide | general | ABSTAIN | ANSWER |  |
| Ecosprin | aspirin | general | ABSTAIN | ANSWER |  |
| Ecotrin | aspirin | general | ABSTAIN | ANSWER |  |
| Eliquis | apixaban | personal | ABSTAIN | GENERALIZE |  |
| Eliquis | apixaban | general | ABSTAIN | ANSWER |  |
| Envas | enalapril | personal | ABSTAIN | ANSWER |  |
| Envas | enalapril | general | ABSTAIN | ANSWER |  |
| Ezetrol | ezetimibe | general | ABSTAIN | ANSWER |  |
| Glucophage | metformin | personal | ANSWER | ABSTAIN |  |
| Glucophage | metformin | general | ABSTAIN | ANSWER |  |
| Glycomet | metformin | personal | GENERALIZE | ABSTAIN |  |
| Glycomet | metformin | general | ABSTAIN | ANSWER |  |
| Hygroton | chlorthalidone | general | ABSTAIN | GENERALIZE |  |
| Imdur | isosorbide mononitrate | personal | ANSWER | ABSTAIN |  |
| Ivabrad | ivabradine | personal | ANSWER | ABSTAIN |  |
| Kerendia | finerenone | general | ABSTAIN | ANSWER |  |
| Lanoxin | digoxin | general | ABSTAIN | ANSWER |  |
| Lorvas | indapamide | general | ABSTAIN | ANSWER |  |
| Losacar | losartan | general | ABSTAIN | GENERALIZE |  |
| Losar | losartan | general | ABSTAIN | GENERALIZE |  |
| Midamor | amiloride | general | ABSTAIN | ANSWER |  |
| Natrilix | indapamide | general | ABSTAIN | ANSWER |  |
| Nikoran | nicorandil | general | GENERALIZE | ABSTAIN |  |
| Ozempic | semaglutide | general | ABSTAIN | GENERALIZE |  |
| Prazopress | prazosin | personal | GENERALIZE | ANSWER |  |

## Extraction flips (58)

| Brand | Expected generic | Stored (brand msg) | Stored (generic msg) |
| :--- | :--- | :--- | :--- |
| Accupril | quinapril | quinapril | — |
| Acitrom | acenocoumarol | acenocoumarol | — |
| Adalat | nifedipine | — | — |
| Aldactone | spironolactone | — | — |
| Aldomet | methyldopa | — | — |
| Alphadopa | methyldopa | — | — |
| Amlodac | amlodipine | amlodipine | — |
| Amlokind | amlodipine | — | — |
| Angispan-TR | nitroglycerin | — | — |
| Apresoline | hydralazine | — | — |
| Aquazide | hydrochlorothiazide | — | hydrochlorothiazide |
| Arkamin | clonidine | — | — |
| Atacand | candesartan | candesartan | — |
| Aten | atenolol | — | — |
| Atocor | atorvastatin | — | — |
| Atorlip | atorvastatin | — | — |
| Betacard | atenolol | atenolol | — |
| Brilinta | ticagrelor | — | — |
| Bumex | bumetanide | bumetanide | — |
| Bystolic | nebivolol | — | — |
| Calan | verapamil | — | — |
| Calaptin | verapamil | — | — |
| Cardace | ramipril | — | — |
| Cardura | doxazosin | — | — |
| Clopilet | clopidogrel | — | — |
| Cordarone | amiodarone | — | — |
| Corlanor | ivabradine | — | — |
| Coumadin | warfarin | — | — |
| Crestor | rosuvastatin | — | — |
| Deplatt | clopidogrel | clopidogrel | — |
| Dilzem | diltiazem | — | — |
| Dytor | torsemide | — | — |
| Ecosprin | aspirin | — | — |
| Ecotrin | aspirin | aspirin | — |
| Eliquis | apixaban | apixaban | — |
| Envas | enalapril | — | — |
| Ezetrol | ezetimibe | — | — |
| Farxiga | dapagliflozin | — | — |
| Flavedon | trimetazidine | — | — |
| Glucophage | metformin | metformin | — |
| Glycomet | metformin | metformin | — |
| Imdur | isosorbide mononitrate | — | — |
| Ivabrad | ivabradine | ivabradine | — |
| Kerendia | finerenone | — | — |
| Lanoxin | digoxin | — | digoxin |
| Leqvio | inclisiran | inclisiran | — |
| Lopid | gemfibrozil | — | — |
| Lorvas | indapamide | — | — |
| Losacar | losartan | — | — |
| Losar | losartan | — | — |
| Midamor | amiloride | — | — |
| Monotrate | isosorbide mononitrate | — | — |
| Natrilix | indapamide | Natriuretic peptide antagonist (implied) | — |
| Nexletol | bempedoic acid | — | — |
| Nikoran | nicorandil | — | — |
| Ozempic | semaglutide | — | — |
| Prazopress | prazosin | prazosin | — |
| Warf | warfarin | — | — |
