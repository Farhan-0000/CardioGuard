# RABBITS Report — Brand vs Generic Drug-Name Robustness

> **MOCK MODE — LLM calls stubbed (the extractor echoes the drug name as written; Gate 1, retrieval, Gate 2 and the generator follow fixed rules). Real components: redaction, emergency classifier, extractor post-processing, drug normaliser, query expansion, citation validation. This checks the deterministic normalisation path and is not evidence of model behaviour.**

- **Mode:** `mock` (stubbed LLM, hermetic)
- **Generated:** 2026-10-09T16:17:40.913520+00:00
- **Pairs:** 189 brand → generic pairs (65 Indian brands), 378 items (personal, general templates); brand and generic messages differ only in the drug name.
- **Pipeline turns:** 602 (generic messages shared by several brands run once); errors: 0
- **templates:** `{"personal": "I take {drug} at my usual dose every morning. Is it okay to have it with grapefruit juice?", "general": "What is {drug} used for?"}`

## Summary

| Metric | Value | Definition |
| :--- | :---: | :--- |
| Action consistency | 100.0% | same routing action for brand and generic message |
| Extracted-generic consistency | 100.0% (n=189) | personal template: same non-empty generic ingredient set stored in the snapshot |
| Brand normalised to its generic | 100.0% | personal template: brand variant stored exactly the pair's generic ingredients |
| Citation-set Jaccard (chunks) | 0.9881 | mean over items; 1.0 when neither variant cites |
| Citation-set Jaccard (guidelines) | 1.0 | same, by guideline id |
| Response length ratio | mean 1.0065, median 1.0 | brand chars / generic chars; 98.68% of items within 0.8–1.25 |
| Median turn latency | 0.6 ms | client wall-clock per message |

## By template

| Template | Items | Action consistency | Extraction consistency | Citation Jaccard |
| :--- | :---: | :---: | :---: | :---: |
| general | 189 | 100.0% | n/a | 0.9894 |
| personal | 189 | 100.0% | 100.0% | 0.9868 |

## By brand origin

| Origin | Items | Action consistency | Extraction consistency | Citation Jaccard |
| :--- | :---: | :---: | :---: | :---: |
| indian | 130 | 100.0% | 100.0% | 0.9808 |
| international | 248 | 100.0% | 100.0% | 0.9919 |

## Action flips (0)

None.

## Extraction flips (0)

None.
