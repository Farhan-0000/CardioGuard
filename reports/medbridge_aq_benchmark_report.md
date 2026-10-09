# MedBridge-AQ Routing Benchmark Report

> **MOCK MODE — routing simulator — not evidence of model behaviour.** Actions come from the real emergency classifier, the loop-breaker rule and the vignette's own subset label; citations are pseudo-citations built from the corpus manifest. Use these numbers only to check the harness. Run `--mode live` for pipeline results.

- **Generated:** 2026-10-09 16:17:33 UTC
- **Mode:** `mock` (routing simulator)
- **Vignettes evaluated:** 275
- **Errors (turns that raised in the harness):** 0
- **subset:** `all`
- **limit:** `none`

## 1. Summary against acceptance targets

| Metric | Target | Measured | Status | Basis |
| :--- | :---: | :---: | :---: | :--- |
| Routing accuracy | >= 90.0% | 100.0% | PASS | 275/275 expected actions matched |
| Citation precision | >= 90.0% | 100.0% | PASS | macro-average over 93 vignettes that returned >= 1 citation and list expected sources |
| Emergency recall | 100.0% | 100.0% | PASS | 47/47 expected-ESCALATE vignettes escalated (47 by the deterministic fast-path) |
| Out-of-domain ANSWER rate | 0.0% | 0.0% | PASS | 0 of 42 out-of-domain vignettes not ABSTAINed; ANSWER on out-of-domain is a routing proxy, not a hallucination judgement |

Overall: **all acceptance targets met** (4/4 targets met; targets without applicable vignettes are not judged).

## 2. Results by subset

| Subset | Vignettes | Correct | Routing accuracy | Citation precision (n) | Mean latency | Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `complete_context_answer` | 56 | 56 | 100.0% | 100.0% (56) | 0.21 ms | 0 |
| `complex_comorbid_contraindications` | 38 | 38 | 100.0% | 100.0% (37) | 0.19 ms | 0 |
| `emergency_escalate` | 46 | 46 | 100.0% | n/a (0) | 0.08 ms | 0 |
| `incomplete_context_soft_ask` | 50 | 50 | 100.0% | n/a (0) | 0.05 ms | 0 |
| `lifestyle_and_generalize` | 43 | 43 | 100.0% | n/a (0) | 0.05 ms | 0 |
| `out_of_domain_abstain` | 42 | 42 | 100.0% | n/a (0) | 0.04 ms | 0 |

## 3. Safety invariants (computed from this run)

- **Emergency triage:** 47 of 47 expected-ESCALATE vignettes were escalated; 47 by the deterministic fast-path before any LLM call.
- **Loop-breaker (soft_ask_count >= 2):** 10 vignettes start at or above the threshold; 10 ended in GENERALIZE (100.0%); 10 were forced by the loop-breaker (Gate 1 returned SOFT-ASK at the threshold).
- **Out-of-domain:** 42 of 42 out-of-domain vignettes were ABSTAINed; 0 received an ANSWER.
- **Citations on expected-ANSWER vignettes:** 93/93 returned >= 1 citation (100.0%).

## 4. Action confusion matrix

Rows: expected action; columns: actual action.

| Expected \ Actual | ANSWER | SOFT-ASK | GENERALIZE | ABSTAIN | ESCALATE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ANSWER** | 93 | 0 | 0 | 0 | 0 |
| **SOFT-ASK** | 0 | 50 | 0 | 0 | 0 |
| **GENERALIZE** | 0 | 0 | 43 | 0 | 0 |
| **ABSTAIN** | 0 | 0 | 0 | 42 | 0 |
| **ESCALATE** | 0 | 0 | 0 | 0 | 47 |

## 5. Who decided the final route

| Decision source | Vignettes |
| :--- | :---: |
| `fastpath` | 47 |
| `loop_breaker` | 10 |
| `simulated` | 218 |

## 6. Latency

- Client wall-clock per turn (excluding harness errors): mean 0.11 ms, p50 0.07 ms, p95 0.23 ms, p99 0.29 ms.

## 7. Misrouted or failed vignettes (0)

None.
