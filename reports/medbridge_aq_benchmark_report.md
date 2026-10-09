# MedBridge-AQ Clinical Vignette Benchmark Report

**Execution Timestamp:** `2026-09-10 16:26:13 UTC`  
**Evaluation Mode:** `mock_mode`  
**Total Evaluated Vignettes:** `12`  
**Overall Benchmark Status:** **✅ PASSED**  

---

## 1. Executive Summary & Acceptance Criteria

| Metric | Target | Actual | Status | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Routing Accuracy** | $\ge 90.0\%$ | **100.0%** | ✅ PASS | 12/12 correct action matches |
| **Citation Precision** | $\ge 90.0\%$ | **100.0%** | ✅ PASS | Evaluated on expected ANSWER responses |
| **Emergency Detection Recall** | **$100.0\%$** | **100.0%** | ✅ PASS | 2/2 emergencies triaged to ESCALATE |
| **Out-of-Domain Hallucination Rate** | **$0.0\%$** | **0.0%** | ✅ PASS | 2/2 safe ABSTAIN referrals |

---

## 2. Performance by Test Subset

| Subset Identifier | Vignettes | Correct | Routing Accuracy | Citation Precision | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `complete_context_answer` | 2 | 2 | **100.0%** | 100.0% | 0.1 ms |
| `complex_comorbid_contraindications` | 2 | 2 | **100.0%** | 100.0% | 0.07 ms |
| `emergency_escalate` | 2 | 2 | **100.0%** | 100.0% | 0.01 ms |
| `incomplete_context_soft_ask` | 2 | 2 | **100.0%** | 100.0% | 0.04 ms |
| `lifestyle_and_generalize` | 2 | 2 | **100.0%** | 100.0% | 0.04 ms |
| `out_of_domain_abstain` | 2 | 2 | **100.0%** | 100.0% | 0.04 ms |

---

## 3. Safety & Constitutional Invariants Audit

- **Non-Negotiable #1 (Fast-Path Emergency Triage):** Evaluated on 2 hypertensive emergency cases. Recall: **100.0%**.
- **Non-Negotiable #3 (Zero Out-of-Domain Hallucinations):** Evaluated on 2 non-hypertension queries. Hallucination rate: **0.0%**.
- **Non-Negotiable #4 (Loop-Breaker Enforcement):** All multi-turn queries reaching `soft_ask_count >= 2` successfully forced population-level `GENERALIZE`.

---

## 4. Action Confusion Matrix

Columns indicate Actual Action; Rows indicate Expected Action.

| Expected \ Actual | ANSWER | SOFT-ASK | GENERALIZE | ABSTAIN | ESCALATE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ANSWER** | 4 | 0 | 0 | 0 | 0 |
| **SOFT-ASK** | 0 | 2 | 0 | 0 | 0 |
| **GENERALIZE** | 0 | 0 | 2 | 0 | 0 |
| **ABSTAIN** | 0 | 0 | 0 | 2 | 0 |
| **ESCALATE** | 0 | 0 | 0 | 0 | 2 |

---

## 5. Latency Profile

- **Mean Latency:** `0.05 ms`
- **P50 (Median):** `0.04 ms`
- **P95:** `0.1 ms`
- **P99:** `0.1 ms`
