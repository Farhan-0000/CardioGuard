# PII/PHI redaction — evaluation

Generated 2026-10-09T10:57:59+00:00 by `scripts/pii_eval.py` on `tests/privacy/pii_testset.json` (v1.1), reference date 2026-10-09.

Span-level metrics. *Strict*: exact character offsets and category; *overlap*: any overlap with a gold span of the same category. The `dev` split was written together with the detector (in-sample); the `heldout` split was written after the detector was frozen and never used for tuning.

| Split | Examples | Negatives | Gold spans | P (strict) | R (strict) | F1 (strict) | P (overlap) | R (overlap) | Negatives altered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 240 | 90 | 183 | 1.000 | 0.962 | 0.981 | 1.000 | 0.962 | 0 |
| dev | 185 | 65 | 147 | 1.000 | 0.986 | 0.993 | 1.000 | 0.986 | 0 |
| heldout | 55 | 25 | 36 | 1.000 | 0.861 | 0.925 | 1.000 | 0.861 | 0 |

Clinical-value protection: 0 of 90 clinically dense negative messages altered; 0 protected clinical values (BP readings, doses, drug names, lab values, ages) corrupted across all examples.

Mean detection latency: 0.025 ms per message (3.11.15, arm64).

## Per category (all examples, strict)

| Category | Support | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AADHAAR | 8 | 8 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| ADDRESS | 14 | 14 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| AGE | 13 | 13 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| EMAIL | 11 | 11 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| INSURANCE_ID | 13 | 13 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| IP | 6 | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| MRN | 13 | 13 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| NAME | 49 | 44 | 0 | 5 | 1.000 | 0.898 | 0.946 |
| PAN | 6 | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| PHONE | 21 | 20 | 0 | 1 | 1.000 | 0.952 | 0.976 |
| PINCODE | 18 | 17 | 0 | 1 | 1.000 | 0.944 | 0.971 |
| SSN | 6 | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| URL | 5 | 5 | 0 | 0 | 1.000 | 1.000 | 1.000 |

## Per category (held-out split, strict)

| Category | Support | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AADHAAR | 2 | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| ADDRESS | 3 | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| AGE | 3 | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| EMAIL | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| INSURANCE_ID | 3 | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| IP | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| MRN | 2 | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| NAME | 9 | 6 | 0 | 3 | 1.000 | 0.667 | 0.800 |
| PAN | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| PHONE | 4 | 3 | 0 | 1 | 1.000 | 0.750 | 0.857 |
| PINCODE | 5 | 4 | 0 | 1 | 1.000 | 0.800 | 0.889 |
| SSN | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| URL | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |

## Errors (strict)

- **P026** — FN NAME `Rahul`
- **P027** — FN NAME `Suresh`
- **H001** — FN NAME `Rajesh Iyer`
- **H003** — FN NAME `Arun Pillai`
- **H005** — FN PHONE `+91 80 4123 4567`
- **H022** — FN NAME `Meghna Bose`
- **H029** — FN PINCODE `560034`
