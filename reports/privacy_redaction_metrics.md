# PII/PHI redaction — evaluation

Generated 2026-10-09T16:04:57+00:00 by `scripts/pii_eval.py`; reference date 2026-10-09 (DOB → age); detector `medbridge/privacy/redaction.py` SHA-256 `e75b67bc708d0522…` at commit `4817e42` (plus working-tree changes when the hash differs from the committed file).

## Protocol

1. The detector (`medbridge/privacy/redaction.py`) is rule based; it was developed only against the development set `tests/privacy/pii_testset.json` (v1.1, both its `dev` split and its earlier `heldout` split, which became development data once its misses were studied) plus self-authored probe sentences in `tests/privacy/test_redaction.py`.
2. The v1 held-out misses (names after 'its me X', 'my husband's name is X', 'Sign: X'; a +91 landline written with spaces; a PIN after a comma-separated locality) were fixed by general rules, not string-specific ones, and clinical values were re-checked.
3. The detector was then frozen; its SHA-256 is recorded below.
4. The blind held-out set `tests/privacy/pii_testset_blind_v2.json` was written by another author who did not read the detector, this script or its metrics. Only its header metadata (format, conventions, category counts) was read before the run, to make the loader accept its start/end offsets; no detector rule was added for anything mentioned only there.
5. The blind set was evaluated once with the frozen detector. The detector was not changed after seeing the results; problems found are listed as open issues instead. The report was regenerated afterwards with the same detector hash only to add the category-agnostic masking metrics and the post-hoc error analysis; every span metric is identical to the first run.
6. Development-set numbers are in-sample and optimistic; the blind numbers are the estimate of generalisation.

Span-level metrics. *Strict*: exact character offsets and category; *overlap*: any overlap with a gold span of the same category. Precision is 1.0 by convention when nothing is predicted.

## Summary

| Dataset / split | Role | Examples | Negatives | Gold spans | P (strict) | R (strict) | F1 (strict) | P (overlap) | R (overlap) | F1 (overlap) | Negatives altered | Protected values corrupted |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| development | development | 240 | 90 | 183 | 1.000 | 0.989 | 0.995 | 1.000 | 0.989 | 0.995 | 0 | 0 |
| development / dev | development | 185 | 65 | 147 | 1.000 | 0.986 | 0.993 | 1.000 | 0.986 | 0.993 | 0 | 0 |
| development / heldout | development | 55 | 25 | 36 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| blind_v2 | blind_heldout | 237 | 89 | 203 | 0.944 | 0.744 | 0.832 | 0.988 | 0.778 | 0.871 | 1 | 0 |

### Privacy masking (category-agnostic)

Whatever the predicted label: *touched* = some redaction overlaps the identifier; *fully masked* = no letter or digit of the identifier survives redaction.

| Dataset / split | Gold spans | Touched | Touched rate | Fully masked | Fully masked rate |
|---|---:|---:|---:|---:|---:|
| development | 183 | 181 | 0.989 | 181 | 0.989 |
| development / dev | 147 | 145 | 0.986 | 145 | 0.986 |
| development / heldout | 36 | 36 | 1.000 | 36 | 1.000 |
| blind_v2 | 203 | 159 | 0.783 | 152 | 0.749 |

## Development set (in-sample): `development`

`tests/privacy/pii_testset.json` — MedBridge synthetic PII/PHI redaction test set v1.1, SHA-256 `f6199442757ce83d…`, 240 examples (150 positive, 90 negative), 183 gold spans. Mean detection latency 0.063 ms per message.

### Clinical-value protection

- Negative (PII-free, clinically dense) messages altered: **0** of 90
- Protected clinical values (BP, labs, doses, drug names, ages, reading dates) corrupted across all 240 examples: **0**

### Per category

| Category | Support | TP | FP | FN | P (strict) | R (strict) | F1 (strict) | P (overlap) | R (overlap) | F1 (overlap) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AADHAAR | 8 | 8 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ADDRESS | 14 | 14 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AGE | 13 | 13 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| EMAIL | 11 | 11 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| INSURANCE_ID | 13 | 13 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| IP | 6 | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| MRN | 13 | 13 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| NAME | 49 | 47 | 0 | 2 | 1.000 | 0.959 | 0.979 | 1.000 | 0.959 | 0.979 |
| PAN | 6 | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PHONE | 21 | 21 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PINCODE | 18 | 18 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SSN | 6 | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| URL | 5 | 5 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **overall** | 183 | 181 | 0 | 2 | 1.000 | 0.989 | 0.995 | 1.000 | 0.989 | 0.995 |

#### Split `dev`

| Category | Support | TP | FP | FN | P (strict) | R (strict) | F1 (strict) | P (overlap) | R (overlap) | F1 (overlap) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AADHAAR | 6 | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ADDRESS | 11 | 11 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AGE | 10 | 10 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| EMAIL | 10 | 10 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| INSURANCE_ID | 10 | 10 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| IP | 5 | 5 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| MRN | 11 | 11 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| NAME | 40 | 38 | 0 | 2 | 1.000 | 0.950 | 0.974 | 1.000 | 0.950 | 0.974 |
| PAN | 5 | 5 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PHONE | 17 | 17 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PINCODE | 13 | 13 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SSN | 5 | 5 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| URL | 4 | 4 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **overall** | 147 | 145 | 0 | 2 | 1.000 | 0.986 | 0.993 | 1.000 | 0.986 | 0.993 |

#### Split `heldout`

| Category | Support | TP | FP | FN | P (strict) | R (strict) | F1 (strict) | P (overlap) | R (overlap) | F1 (overlap) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AADHAAR | 2 | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ADDRESS | 3 | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| AGE | 3 | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| EMAIL | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| INSURANCE_ID | 3 | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| IP | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| MRN | 2 | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| NAME | 9 | 9 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PAN | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PHONE | 4 | 4 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PINCODE | 5 | 5 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SSN | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| URL | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **overall** | 36 | 36 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### Errors (strict)

- **P026** — FN NAME `Rahul` — _out of scope: name without a contextual cue (expected miss)_
- **P027** — FN NAME `Suresh` — _out of scope: name without a contextual cue (expected miss)_

## Blind held-out set (evaluated once, frozen detector): `blind_v2`

`tests/privacy/pii_testset_blind_v2.json` — MedBridge synthetic PII/PHI redaction test set - blind held-out v2 v2.0, SHA-256 `eefa4b15d601daa8…`, 237 examples (148 positive, 89 negative), 203 gold spans. Mean detection latency 0.043 ms per message.

### Clinical-value protection

- Negative (PII-free, clinically dense) messages altered: **1** of 89 (B2N072: FP NAME `Google`)
- Protected clinical values (BP, labs, doses, drug names, ages, reading dates) corrupted across all 237 examples: **0**

### Per category

| Category | Support | TP | FP | FN | P (strict) | R (strict) | F1 (strict) | P (overlap) | R (overlap) | F1 (overlap) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AADHAAR | 8 | 8 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ADDRESS | 18 | 10 | 7 | 8 | 0.588 | 0.556 | 0.571 | 1.000 | 0.944 | 0.971 |
| AGE | 15 | 14 | 0 | 1 | 1.000 | 0.933 | 0.966 | 1.000 | 0.933 | 0.966 |
| EMAIL | 9 | 9 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| INSURANCE_ID | 12 | 11 | 0 | 1 | 1.000 | 0.917 | 0.957 | 1.000 | 0.917 | 0.957 |
| IP | 7 | 7 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| MRN | 11 | 8 | 1 | 3 | 0.889 | 0.727 | 0.800 | 0.889 | 0.727 | 0.800 |
| NAME | 63 | 27 | 1 | 36 | 0.964 | 0.429 | 0.593 | 0.964 | 0.429 | 0.593 |
| PAN | 7 | 7 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PHONE | 21 | 20 | 0 | 1 | 1.000 | 0.952 | 0.976 | 1.000 | 0.952 | 0.976 |
| PINCODE | 18 | 17 | 0 | 1 | 1.000 | 0.944 | 0.971 | 1.000 | 0.944 | 0.971 |
| SSN | 6 | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| URL | 8 | 7 | 0 | 1 | 1.000 | 0.875 | 0.933 | 1.000 | 0.875 | 0.933 |
| **overall** | 203 | 151 | 9 | 52 | 0.944 | 0.744 | 0.832 | 0.988 | 0.778 | 0.871 |

### Post-hoc error analysis

Written after the single blind run of 2026-10-09T16:02:05+00:00 (detector SHA-256 `e75b67bc708d0522…`). The detector was **not** changed in response; the groups below are the open issues. Counts are recomputed from this run's errors.

| # | Error group | Examples | FN spans | FP spans |
|---:|---|---:|---:|---:|
| 1 | NAME with no supported cue: the name opens the message, sits in a list or header, or is the object of a verb ('Anjali Bhosale, 47 - ...', 'Tanvir and Fatima', 'Claim form: X', 'ask X at the front desk'). Out of scope by design (cue-based names only, to protect drug and clinical words). (B2P021, B2P024, B2P031, B2P032, B2P039, B2P040, B2P041, B2P065, B2P070, B2P098, B2P099, B2P126, B2P138, B2P142, B2P144) | 15 | 17 | 0 |
| 2 | NAME after an English / Indian-English cue the detector does not know: 'X here', 'X this side', 'I X am', 'From: X', 'Me (X)', 'Sent by X', '- X' closing the message, 'name X' without a colon, 'Pt: X', and 'this is X,' with neither a greeting nor a 'calling/speaking' trailer. (B2P002, B2P008, B2P009, B2P017, B2P023, B2P025, B2P033, B2P061, B2P066, B2P084, B2P147, B2P148) | 12 | 12 | 0 |
| 3 | NAME in SMS style: lower-case name after 'im' / 'this is', or a misspelt cue ('husbnd', 'doctr'). Capitalised-only cues miss them. (B2P007, B2P028, B2P029) | 3 | 4 | 0 |
| 4 | NAME after a Hinglish cue ('mera naam X hai', 'unka naam X', 'Beta X'). (B2P011, B2P012, B2P015) | 3 | 3 | 0 |
| 5 | UK formats not supported (deliberately not added: they appear only in the blind set's metadata, not in the development data): NHS numbers, +44 phone numbers. (B2P052, B2P083, B2P088, B2P126) | 4 | 4 | 0 |
| 6 | ADDRESS with a UK postcode: the locality extension stops inside the postcode ('Leicester LE' + '3 9QT'), leaving part of it visible (bug). (B2P143, B2P144) | 2 | 2 | 2 |
| 7 | ADDRESS boundary errors on Indian / US formats: house numbers with a letter prefix or a slash ('B-12', '12/4', 'Quarter No. C-45', 'Gali No. 3'), the US abbreviation 'Dr' (Drive) and a directional prefix ('2205 W Belmont Ave'). Part of the address stays visible; in B2P130 the PIN too. (B2P130, B2P132, B2P135, B2P137, B2P139, B2P140) | 6 | 7 | 5 |
| 8 | Other formats: scheme-less URL ('bpbuddy.app/...'), the DOB cue 'b.', and an ESIC 'IP number' redacted as MRN instead of INSURANCE_ID (the value is still masked; only the category is wrong). (B2P093, B2P106, B2P120) | 3 | 3 | 1 |
| 9 | False positive on a negative: the title cue 'Dr.' followed by a non-person word ('Dr. Google'). No clinical value was touched. (B2N072) | 1 | 0 | 1 |

Unclassified errors: 0.

### Errors (strict)

- **B2P002** — FN NAME `Lalitha`
- **B2P007** — FN NAME `rohit` — _all-lowercase name with typo'd cue_
- **B2P008** — FN NAME `Kishore Nambiar` — _Indian-English cue 'X this side'_
- **B2P009** — FN NAME `Ramesh Kumar Yadav`
- **B2P011** — FN NAME `Awadhesh Pandey` — _Hinglish cue 'naam'_
- **B2P012** — FN NAME `Imran` — _Hinglish_
- **B2P015** — FN NAME `Arjun`
- **B2P017** — FN NAME `Margaret Holloway`
- **B2P021** — FN NAME `Tanvir`; FN NAME `Fatima`
- **B2P023** — FN NAME `Rohan Batra`
- **B2P024** — FN NAME `Anjali Bhosale` — _name at message start, no verbal cue_
- **B2P025** — FN NAME `Hemant Ghosh`
- **B2P028** — FN NAME `meena` — _lowercase name, typos_
- **B2P029** — FN NAME `sneha`; FN NAME `ashok` — _lowercase names, SMS style_
- **B2P031** — FN NAME `Mridula Sen`
- **B2P032** — FN NAME `Sarita Bose` — _name directly followed by a BP value_
- **B2P033** — FN NAME `Bhavna` — _sign-off name_
- **B2P039** — FN NAME `Carol Dempsey`
- **B2P040** — FN NAME `Prakash` — _name with no introduction cue_
- **B2P041** — FN NAME `Savita`; FN NAME `Mohan` — _names with no introduction cue_
- **B2P052** — FN PHONE `+44 7700 900872`
- **B2P061** — FN NAME `Tejas Wagh`
- **B2P065** — FN NAME `Lakshmi Narayanan`
- **B2P066** — FN NAME `Gurpreet Kaur Sandhu`
- **B2P070** — FN NAME `Suresh Kulkarni`
- **B2P083** — FN MRN `999 802 9252` — _NHS number labelled MRN (999 test range)_
- **B2P084** — FN NAME `Deepa Menon`
- **B2P088** — FN MRN `999 979 7385` — _NHS number labelled MRN (999 test range)_
- **B2P093** — FN INSURANCE_ID `3115678942`; FP MRN `3115678942` — _10-digit insurance number that looks like a phone_
- **B2P098** — FN NAME `Chitra Venkataraman`
- **B2P099** — FN NAME `Daniel Okafor`
- **B2P106** — FN URL `bpbuddy.app/share/k7Fq92` — _scheme-less URL_
- **B2P120** — FN AGE `9/23/1962` — _US month-first date_
- **B2P126** — FN NAME `Gwen Pritchard`; FN MRN `999 984 8109`
- **B2P130** — FN ADDRESS `B-12, Lajpat Nagar II, New Delhi`; FN PINCODE `110024`; FP ADDRESS `12, Lajpat Nagar`
- **B2P132** — FN ADDRESS `12/4 Karve Road, Erandwane, Pune`; FP ADDRESS `4 Karve Road, Erandwane, Pune` — _house number 12/4 looks like a BP/date fraction_
- **B2P135** — FN ADDRESS `Quarter No. C-45, Railway Colony, Sector 4, Bokaro Steel City, Jharkhand`; FP ADDRESS `45, Railway Colony, Sector 4, Bokaro Steel City, Jharkhand`
- **B2P137** — FN ADDRESS `Gali No. 3, Krishna Nagar, Shahdara, Delhi`; FP ADDRESS `No. 3, Krishna Nagar, Shahdara, Delhi` — _Hinglish label_
- **B2P138** — FN NAME `Rekha Pillai`
- **B2P139** — FN ADDRESS `1427 Maple Ridge Dr, Apt 3B, Columbus, OH`; FP ADDRESS `Apt 3B, Columbus, OH`
- **B2P140** — FN ADDRESS `2205 W Belmont Ave, Chicago, IL`
- **B2P142** — FN NAME `Raymond Ortiz`
- **B2P143** — FN ADDRESS `14 Hawthorn Close, Leicester LE3 9QT`; FP ADDRESS `14 Hawthorn Close, Leicester LE` — _UK postcode is not a PIN/ZIP, so it stays inside ADDRESS_
- **B2P144** — FN NAME `Siobhan Gallagher`; FN ADDRESS `22 Kingsley Road, Didsbury, Manchester M20 2RS`; FP ADDRESS `22 Kingsley Road, Didsbury, Manchester M` — _UK postcode is not a PIN/ZIP, so it stays inside ADDRESS_
- **B2P147** — FN NAME `Abhay Joglekar`
- **B2P148** — FN NAME `Linda Castellano`
- **B2N072** — FP NAME `Google` — _title followed by a non-person word_

Environment: Python 3.11.15, arm64.
