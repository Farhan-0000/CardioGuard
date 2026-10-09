# RABBITS Report — Brand vs Generic Drug-Name Robustness

- **Mode:** `live` (production pipeline, real LLM)
- **Generated:** 2026-10-09T16:13:49.412248+00:00
- **Pairs:** 1 brand → generic pairs (0 Indian brands), 2 items (personal, general templates); brand and generic messages differ only in the drug name.
- **Pipeline turns:** 4 (generic messages shared by several brands run once); errors: 0
- **templates:** `{"personal": "I take {drug} at my usual dose every morning. Is it okay to have it with grapefruit juice?", "general": "What is {drug} used for?"}`
- **git_commit:** `4817e42+dirty`
- **llm_provider:** `ollama`
- **gates_model:** `qwen3:8b`
- **generator_model:** `qwen3:8b`
- **temperature_gates:** `0.0`
- **temperature_generator:** `0.3`
- **seed:** `42`
- **database:** `postgresql+asyncpg://medbridge_app:***@127.0.0.1:5433/medbridge_eval`
- **qdrant_collection:** `clinical_guidelines`
- **grounding_verifier:** `False`
- **calibrated_evidence_gate:** `False`
- **checkpoint:** `reports/benchmark_runs/rabbits_live_dryrun.jsonl`

## Summary

| Metric | Value | Definition |
| :--- | :---: | :--- |
| Action consistency | 50.0% | same routing action for brand and generic message |
| Extracted-generic consistency | 100.0% (n=1) | personal template: same non-empty generic ingredient set stored in the snapshot |
| Brand normalised to its generic | 100.0% | personal template: brand variant stored exactly the pair's generic ingredients |
| Citation-set Jaccard (chunks) | 0.0 | mean over items; 1.0 when neither variant cites |
| Citation-set Jaccard (guidelines) | 0.0 | same, by guideline id |
| Response length ratio | mean 1.1321, median 1.1321 | brand chars / generic chars; 50.0% of items within 0.8–1.25 |
| Median turn latency | 25702.0 ms | client wall-clock per message |

## By template

| Template | Items | Action consistency | Extraction consistency | Citation Jaccard |
| :--- | :---: | :---: | :---: | :---: |
| general | 1 | 100.0% | n/a | 0.0 |
| personal | 1 | 0.0% | 100.0% | 0.0 |

## By brand origin

| Origin | Items | Action consistency | Extraction consistency | Citation Jaccard |
| :--- | :---: | :---: | :---: | :---: |
| international | 2 | 50.0% | 100.0% | 0.0 |

## Action flips (1)

| Brand | Generic | Template | Brand action | Generic action | Error |
| :--- | :--- | :--- | :---: | :---: | :--- |
| Accupril | quinapril | personal | ABSTAIN | ANSWER |  |

## Extraction flips (0)

None.
