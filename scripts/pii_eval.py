#!/usr/bin/env python3
"""
Evaluate the PII/PHI redactor on the development set and the blind held-out set.

    python scripts/pii_eval.py                                  # both sets (default)
    python scripts/pii_eval.py --dataset development            # one named set
    python scripts/pii_eval.py --dataset blind_v2 --print-only  # no report files written
    python scripts/pii_eval.py --dataset mine=path/to/set.json  # any set, as label=path
    python scripts/pii_eval.py --out-dir reports

Named datasets:

* ``development`` — ``tests/privacy/pii_testset.json`` (v1.1). Both of its splits
  (``dev`` and the earlier ``heldout``) were used to develop the detector, so its
  numbers are in-sample.
* ``blind_v2`` — ``tests/privacy/pii_testset_blind_v2.json``, written by a separate
  author who never saw the detector, its evaluation script or its metrics. It was
  evaluated once, after the detector was frozen; the detector's SHA-256 and commit are
  recorded so the run can be tied to the exact code. Its primary numbers always come
  from that frozen detector (read back from git and hash-checked). When the current
  detector differs (post-hoc fixes made after the blind run), the blind set is also
  evaluated with the current detector and reported in a separate, clearly labelled
  "post-hoc" section: those numbers are optimistic, not an independent estimate.

Writes ``reports/privacy_redaction_metrics.json`` and ``.md`` with span-level
precision / recall / F1 per category (strict = exact offsets + category,
overlap = any overlap + category) for every dataset and split, plus the
clinical-value corruption check (negatives altered, protected values corrupted).
Gold spans may be given as substrings (``text`` + optional ``occurrence``) or with
explicit ``start``/``end`` offsets (v2 format); offsets are validated and honoured.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import subprocess
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from medbridge.privacy import evaluation as evaluation_module  # noqa: E402
from medbridge.privacy import redaction as current_detector  # noqa: E402
from medbridge.privacy.evaluation import EVAL_TODAY, evaluate, resolve_gold  # noqa: E402

DETECTOR = REPO_ROOT / "medbridge" / "privacy" / "redaction.py"
DATASETS: dict[str, Path] = {
    "development": REPO_ROOT / "tests" / "privacy" / "pii_testset.json",
    "blind_v2": REPO_ROOT / "tests" / "privacy" / "pii_testset_blind_v2.json",
}
ROLES: dict[str, str] = {"development": "development", "blind_v2": "blind_heldout"}

PROTOCOL: list[str] = [
    "The detector (`medbridge/privacy/redaction.py`) is rule based; it was developed "
    "only against the development set `tests/privacy/pii_testset.json` (v1.1, both its "
    "`dev` split and its earlier `heldout` split, which became development data once its "
    "misses were studied) plus self-authored probe sentences in "
    "`tests/privacy/test_redaction.py`.",
    "The v1 held-out misses (names after 'its me X', 'my husband's name is X', 'Sign: X'; a "
    "+91 landline written with spaces; a PIN after a comma-separated locality) were fixed "
    "by general rules, not string-specific ones, and clinical values were re-checked.",
    "The detector was then frozen; its SHA-256 is recorded below.",
    "The blind held-out set `tests/privacy/pii_testset_blind_v2.json` was written by "
    "another author who did not read the detector, this script or its metrics. Only its "
    "header metadata (format, conventions, category counts) was read before the run, to "
    "make the loader accept its start/end offsets; no detector rule was added for anything "
    "mentioned only there.",
    "The blind set was evaluated once with the frozen detector; problems found were "
    "listed as open issues. The report was regenerated afterwards with the same detector "
    "hash only to add the category-agnostic masking metrics and the post-hoc error "
    "analysis; every span metric is identical to the first run. The blind section always "
    "shows this frozen detector's numbers.",
    "Development-set numbers are in-sample and optimistic; the blind numbers of the "
    "frozen detector are the estimate of generalisation.",
    "After the blind run, a code review found an ADDRESS boundary bug (the locality "
    "extension could stop inside a token such as '15A', 'C3' or 'LE3', leaving the rest "
    "of the address and the PIN visible) and the blind run showed the 'Dr. <non-name>' "
    "false positive. These bug classes were fixed post-hoc with general rules (listed in "
    "the post-hoc section) and re-checked on the development set. The blind section keeps "
    "the frozen detector's numbers (re-computed from the frozen source at its recorded "
    "commit and hash); the post-hoc numbers on the blind set are optimistic and are not "
    "an independent estimate.",
]

# Changes made to the detector after the single blind run (shown in the post-hoc section).
POST_HOC_FIXES: list[str] = [
    "ADDRESS continuation tokens end on a word boundary and may mix letters and digits "
    "('Sector 15A', 'Phase 2B', 'Tower C3', 'Road No. 12A', 'Tower 4567', UK postcodes "
    "'LE3 9QT', 'M20 2RS'); bare numbers keep at most 4 digits so 5- and 6-digit PIN/ZIP "
    "codes still go to the PINCODE pass. The span fails closed (it never ends inside a "
    "word) and starts at the beginning of its token ('B-12', '12/4'). Code-review finding "
    "privacy-security-1; blind error groups 6 and 7 (in part).",
    "A street core continues over the rest of its own locality before the next comma "
    "('Lajpat Nagar II, New Delhi'), so the remainder and the PIN are no longer visible.",
    "Over-redaction guard on the same extension: it stops at function words, lab and "
    "measurement words, drug names from the lexicon, a number followed by a unit "
    "('40 mg', '25%'), decimals and times, and a sentence-ending full stop "
    "('Mumbai, Atorvastatin 40mg daily' and 'Bangalore. Sugar 7.2' keep their clinical "
    "text; abbreviations such as 'St.', 'No.' and 'J.P.' do not end the address).",
    "NAME: a title or name cue skips information sources and services ('Dr. Google', "
    "'Doctor YouTube', 'Dr. Internet'); blind error group 9.",
]

# Post-hoc error analysis of the single blind run, keyed by the blind file's SHA-256 so it
# is only rendered for that exact file. Members are "ID" (all errors of the example) or
# "ID:CATEGORY"; counts are recomputed from the run's errors, nothing is typed in by hand.
POST_HOC_ANALYSIS: dict[str, dict[str, Any]] = {
    "eefa4b15d601daa8fad7e1466f1c37fc9945317d40d60ae1676711262f524d2c": {
        "first_run_at": "2026-10-09T16:02:05+00:00",
        "detector_sha256": "e75b67bc708d05227fb4e9338919d06b5150530aa2f6a877dc252dc90c9c1015",
        # Commit holding the frozen detector; its source is read back with `git show`
        # and checked against detector_sha256 before the blind set is re-evaluated.
        "detector_commit": "9717a2a",
        # Overall results recorded by the single blind run. They are used to verify that
        # the re-computation reproduces the run, and as the fallback when the frozen
        # source cannot be read (no git history).
        "recorded": {
            "overall_strict": {"tp": 151, "fp": 9, "fn": 52},
            "overall_overlap": {"tp": 158, "fp": 2, "fn": 45},
            "fully_masked": 152,
            "negatives_altered": 1,
            "protected_values_corrupted": 0,
        },
        "groups": [
            (
                "NAME with no supported cue: the name opens the message, sits in a list or "
                "header, or is the object of a verb ('Anjali Bhosale, 47 - ...', 'Tanvir and "
                "Fatima', 'Claim form: X', 'ask X at the front desk'). Out of scope by "
                "design (cue-based names only, to protect drug and clinical words).",
                ["B2P021", "B2P024", "B2P031", "B2P032", "B2P039", "B2P040", "B2P041",
                 "B2P065", "B2P070", "B2P098", "B2P099", "B2P126:NAME", "B2P138:NAME",
                 "B2P142", "B2P144:NAME"],
            ),
            (
                "NAME after an English / Indian-English cue the detector does not know: "
                "'X here', 'X this side', 'I X am', 'From: X', 'Me (X)', 'Sent by X', "
                "'- X' closing the message, 'name X' without a colon, 'Pt: X', and "
                "'this is X,' with neither a greeting nor a 'calling/speaking' trailer.",
                ["B2P002", "B2P008", "B2P009", "B2P017", "B2P023", "B2P025", "B2P033",
                 "B2P061", "B2P066", "B2P084", "B2P147", "B2P148"],
            ),
            (
                "NAME in SMS style: lower-case name after 'im' / 'this is', or a misspelt "
                "cue ('husbnd', 'doctr'). Capitalised-only cues miss them.",
                ["B2P007", "B2P028", "B2P029"],
            ),
            (
                "NAME after a Hinglish cue ('mera naam X hai', 'unka naam X', 'Beta X').",
                ["B2P011", "B2P012", "B2P015"],
            ),
            (
                "UK formats not supported (deliberately not added: they appear only in the "
                "blind set's metadata, not in the development data): NHS numbers, +44 "
                "phone numbers.",
                ["B2P083", "B2P088", "B2P126:MRN", "B2P052"],
            ),
            (
                "ADDRESS with a UK postcode: the locality extension stops inside the "
                "postcode ('Leicester LE' + '3 9QT'), leaving part of it visible (bug).",
                ["B2P143", "B2P144:ADDRESS"],
            ),
            (
                "ADDRESS boundary errors on Indian / US formats: house numbers with a "
                "letter prefix or a slash ('B-12', '12/4', 'Quarter No. C-45', 'Gali No. "
                "3'), the US abbreviation 'Dr' (Drive) and a directional prefix ('2205 W "
                "Belmont Ave'). Part of the address stays visible; in B2P130 the PIN too.",
                ["B2P130", "B2P132", "B2P135", "B2P137", "B2P139", "B2P140"],
            ),
            (
                "Other formats: scheme-less URL ('bpbuddy.app/...'), the DOB cue 'b.', and "
                "an ESIC 'IP number' redacted as MRN instead of INSURANCE_ID (the value is "
                "still masked; only the category is wrong).",
                ["B2P106", "B2P120", "B2P093"],
            ),
            (
                "False positive on a negative: the title cue 'Dr.' followed by a non-person "
                "word ('Dr. Google'). No clinical value was touched.",
                ["B2N072"],
            ),
        ],
    },
}



@dataclass(frozen=True)
class Dataset:
    label: str
    path: Path
    role: str


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalise_offsets(example: dict[str, Any]) -> dict[str, Any]:
    """Turn ``start``/``end`` gold offsets into the ``occurrence`` form used by the evaluator.

    Raises:
        ValueError: when ``text[start:end]`` does not equal the span's ``text``.
    """
    text: str = example["text"]
    spans: list[dict[str, Any]] = []
    for raw in example.get("spans", []):
        span = dict(raw)
        if "start" in span and "end" in span:
            start, end = int(span["start"]), int(span["end"])
            needle: str = span["text"]
            if text[start:end] != needle:
                raise ValueError(f"{example['id']}: offsets {start}:{end} do not match {needle!r}")
            occurrence = 0
            idx = text.find(needle)
            while idx != start:
                idx = text.find(needle, idx + 1)
                occurrence += 1
            span["occurrence"] = occurrence
        spans.append(span)
    return {**example, "spans": spans}


def load_examples(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a test-set file; return (header without examples, normalised examples)."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    examples = [normalise_offsets(e) for e in data["examples"]]
    header = {k: v for k, v in data.items() if k != "examples"}
    return header, examples


def load_frozen_detector(sha256: str, commit: str) -> types.ModuleType | None:
    """Load the detector source frozen for a blind run from git, checked by its SHA-256.

    Returns None when git, the commit or the file is unavailable, or when the bytes do not
    hash to ``sha256`` (then the frozen run cannot be re-computed).
    """
    rel = _relative(DETECTOR)
    try:
        out = subprocess.run(
            ["git", "show", f"{commit}:{rel}"], cwd=REPO_ROOT, capture_output=True,
            timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or hashlib.sha256(out.stdout).hexdigest() != sha256:
        return None
    name = f"medbridge_redaction_frozen_{sha256[:12]}"
    module = types.ModuleType(name)
    module.__file__ = f"{commit}:{rel}"
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    exec(compile(out.stdout, module.__file__, "exec"), module.__dict__)
    return module


@contextlib.contextmanager
def use_detector(detector: types.ModuleType) -> Iterator[None]:
    """Make ``medbridge.privacy.evaluation`` score ``detector`` instead of the current one."""
    saved = {name: getattr(evaluation_module, name) for name in ("detect", "redact")}
    for name in saved:
        setattr(evaluation_module, name, getattr(detector, name))
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(evaluation_module, name, value)


def masking_metrics(
    examples: list[dict[str, Any]], detector: types.ModuleType = current_detector
) -> dict[str, Any]:
    """Category-agnostic privacy view: is each gold identifier hidden, whatever the label?

    ``touched``: some predicted span (any category) overlaps the gold span.
    ``fully_masked``: every letter/digit of the gold span lies inside predicted spans, so
    no character of the identifier survives redaction.
    """
    per_cat: dict[str, dict[str, int]] = {}
    for ex in examples:
        text: str = ex["text"]
        preds = detector.detect(text, today=EVAL_TODAY)
        for g in resolve_gold(ex):
            row = per_cat.setdefault(g.category, {"gold": 0, "touched": 0, "fully_masked": 0})
            row["gold"] += 1
            if any(p.start < g.end and g.start < p.end for p in preds):
                row["touched"] += 1
            visible = [
                i for i in range(g.start, g.end)
                if text[i].isalnum() and not any(p.start <= i < p.end for p in preds)
            ]
            if not visible:
                row["fully_masked"] += 1
    total = {k: sum(r[k] for r in per_cat.values()) for k in ("gold", "touched", "fully_masked")}

    def rates(row: dict[str, int]) -> dict[str, float | int]:
        gold = row["gold"]
        return {
            **row,
            "touched_rate": round(row["touched"] / gold, 4) if gold else 1.0,
            "fully_masked_rate": round(row["fully_masked"] / gold, 4) if gold else 1.0,
        }

    return {
        "overall": rates(total),
        "per_category": {k: rates(v) for k, v in sorted(per_cat.items())},
    }


def _attach_notes(result: dict[str, Any], examples: list[dict[str, Any]]) -> dict[str, Any]:
    notes = {e["id"]: e["note"] for e in examples if e.get("note")}
    for err in result["errors"]:
        if err["id"] in notes:
            err["note"] = notes[err["id"]]
    return result


def evaluate_examples(
    examples: list[dict[str, Any]], detector: types.ModuleType = current_detector
) -> dict[str, dict[str, Any]]:
    """Score ``detector`` on all examples plus one block per split."""
    subsets: dict[str, list[dict[str, Any]]] = {"all": examples}
    splits = sorted({str(e.get("split", "dev")) for e in examples})
    if len(splits) > 1:
        for split in splits:
            subsets[split] = [e for e in examples if str(e.get("split", "dev")) == split]
    results: dict[str, dict[str, Any]] = {}
    with use_detector(detector):
        for name, subset in subsets.items():
            result = _attach_notes(evaluate(subset).to_dict(), subset)
            result["masking"] = masking_metrics(subset, detector)
            results[name] = result
    return results


def check_recorded(result: dict[str, Any], recorded: dict[str, Any]) -> dict[str, Any]:
    """Compare a re-computed frozen run with the numbers recorded by the original run."""
    recomputed = {
        "overall_strict": {k: result["overall_strict"][k] for k in ("tp", "fp", "fn")},
        "overall_overlap": {k: result["overall_overlap"][k] for k in ("tp", "fp", "fn")},
        "fully_masked": result["masking"]["overall"]["fully_masked"],
        "negatives_altered": len(result["negatives_altered"]),
        "protected_values_corrupted": len(result["protected_values_corrupted"]),
    }
    return {"reproduced": recomputed == recorded, "recorded": recorded, "recomputed": recomputed}


def evaluate_dataset(dataset: Dataset) -> dict[str, Any]:
    """Evaluate one dataset.

    A blind set with a recorded frozen run is always scored with that frozen detector
    (``results``). When the current detector differs, it is scored again with the current
    detector under ``post_hoc`` — optimistic numbers, never presented as blind.
    """
    header, examples = load_examples(dataset.path)
    sha = sha256_file(dataset.path)
    current_sha = sha256_file(DETECTOR)
    out: dict[str, Any] = {
        "role": dataset.role,
        "testset": _relative(dataset.path),
        "name": header.get("name", ""),
        "version": header.get("version", "?"),
        "sha256": sha,
        "splits": header.get("splits", {}),
        "n_examples": len(examples),
    }
    frozen = POST_HOC_ANALYSIS.get(sha)
    if frozen is None or frozen["detector_sha256"] == current_sha:
        out["detector"] = {"sha256": current_sha, "source": "working tree"}
        out["results"] = evaluate_examples(examples)
        if frozen is not None:
            out["post_hoc_analysis"] = classify_errors(out["results"]["all"], frozen, current_sha)
        return out

    frozen_detector = load_frozen_detector(frozen["detector_sha256"], frozen["detector_commit"])
    out["detector"] = {
        "sha256": frozen["detector_sha256"],
        "source": f"git commit {frozen['detector_commit']}, frozen for the blind run",
    }
    if frozen_detector is None:
        out["results"] = {}
        out["frozen_unavailable"] = frozen["recorded"]
    else:
        out["results"] = evaluate_examples(examples, frozen_detector)
        out["frozen_check"] = check_recorded(out["results"]["all"], frozen["recorded"])
        out["post_hoc_analysis"] = classify_errors(
            out["results"]["all"], frozen, frozen["detector_sha256"]
        )
    post_hoc = evaluate_examples(examples)
    out["post_hoc"] = {
        "label": "Post-hoc fixes (after blind evaluation; optimistic, not an independent "
                 "estimate)",
        "detector_sha256": current_sha,
        "fixes": POST_HOC_FIXES,
        "results": post_hoc,
        "remaining_by_group": classify_errors(post_hoc["all"], frozen, current_sha),
    }
    return out


def classify_errors(
    result: dict[str, Any], analysis: dict[str, Any], detector_sha256: str
) -> dict[str, Any]:
    """Count the run's FN/FP spans per analysis group (members: "ID" or "ID:CATEGORY")."""
    groups: list[dict[str, Any]] = []
    claimed: set[tuple[str, int]] = set()
    for description, members in analysis["groups"]:
        whole = {m for m in members if ":" not in m}
        partial = {m for m in members if ":" in m}
        fn = fp = 0
        ids: list[str] = []
        for err in result["errors"]:
            for idx, e in enumerate(err["errors"]):
                category = e["span"].split("@")[0]
                if err["id"] in whole or f"{err['id']}:{category}" in partial:
                    claimed.add((err["id"], idx))
                    fn += e["kind"] == "FN"
                    fp += e["kind"] == "FP"
                    if err["id"] not in ids:
                        ids.append(err["id"])
        groups.append({"description": description, "examples": ids, "fn": fn, "fp": fp})
    unclassified = [
        {"id": err["id"], **e}
        for err in result["errors"]
        for idx, e in enumerate(err["errors"])
        if (err["id"], idx) not in claimed
    ]
    return {
        "first_run_at": analysis["first_run_at"],
        "detector_sha256": analysis["detector_sha256"],
        "detector_matches": analysis["detector_sha256"] == detector_sha256,
        "groups": groups,
        "unclassified": unclassified,
    }


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def parse_dataset(spec: str) -> Dataset:
    """``development`` | ``blind_v2`` | ``label=path`` | ``path``."""
    if spec in DATASETS:
        return Dataset(spec, DATASETS[spec], ROLES[spec])
    label, sep, raw_path = spec.partition("=")
    path = Path(raw_path if sep else spec)
    if not sep:
        label = path.stem
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"dataset file not found: {path}")
    return Dataset(label, path, "custom")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _summary_row(label: str, role: str, result: dict[str, Any]) -> str:
    s, o = result["overall_strict"], result["overall_overlap"]
    return (
        f"| {label} | {role} | {result['n_examples']} | {result['n_negative_examples']} | "
        f"{s['support']} | {_fmt(s['precision'])} | {_fmt(s['recall'])} | {_fmt(s['f1'])} | "
        f"{_fmt(o['precision'])} | {_fmt(o['recall'])} | {_fmt(o['f1'])} | "
        f"{len(result['negatives_altered'])} | {len(result['protected_values_corrupted'])} |"
    )


def _category_table(result: dict[str, Any]) -> list[str]:
    rows = [
        "| Category | Support | TP | FP | FN | P (strict) | R (strict) | F1 (strict) "
        "| P (overlap) | R (overlap) | F1 (overlap) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    strict, overlap = result["per_category_strict"], result["per_category_overlap"]
    for cat, m in strict.items():
        o = overlap[cat]
        if m["support"] == 0 and m["fp"] == 0 and o["fp"] == 0:
            continue
        rows.append(
            f"| {cat} | {m['support']} | {m['tp']} | {m['fp']} | {m['fn']} | "
            f"{_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | "
            f"{_fmt(o['precision'])} | {_fmt(o['recall'])} | {_fmt(o['f1'])} |"
        )
    s, o = result["overall_strict"], result["overall_overlap"]
    rows.append(
        f"| **overall** | {s['support']} | {s['tp']} | {s['fp']} | {s['fn']} | "
        f"{_fmt(s['precision'])} | {_fmt(s['recall'])} | {_fmt(s['f1'])} | "
        f"{_fmt(o['precision'])} | {_fmt(o['recall'])} | {_fmt(o['f1'])} |"
    )
    return rows


def _error_lines(result: dict[str, Any]) -> list[str]:
    if not result["errors"]:
        return ["None."]
    lines: list[str] = []
    for err in result["errors"]:
        kinds = "; ".join(
            f"{e['kind']} {e['span'].split('@')[0]} `{e['value']}`" for e in err["errors"]
        )
        note = f" — _{err['note']}_" if err.get("note") else ""
        lines.append(f"- **{err['id']}** — {kinds}{note}")
    return lines


def _corruption_lines(result: dict[str, Any]) -> list[str]:
    altered = result["negatives_altered"]
    corrupted = result["protected_values_corrupted"]
    errors = {err["id"]: err for err in result["errors"]}
    details = [
        f"{i}: " + ", ".join(f"{e['kind']} {e['span'].split('@')[0]} `{e['value']}`"
                             for e in errors[i]["errors"]) if i in errors else i
        for i in altered
    ]
    lines = [
        f"- Negative (PII-free, clinically dense) messages altered: **{len(altered)}** of "
        f"{result['n_negative_examples']}" + (f" ({'; '.join(details)})" if altered else ""),
        f"- Protected clinical values (BP, labs, doses, drug names, ages, reading dates) "
        f"corrupted across all {result['n_examples']} examples: **{len(corrupted)}**",
    ]
    lines += [f"  - {c['id']}: `{c['value']}`" for c in corrupted]
    return lines


def _analysis_lines(analysis: dict[str, Any]) -> list[str]:
    lines = [
        "### Post-hoc error analysis",
        "",
        f"Written after the single blind run of {analysis['first_run_at']} (detector SHA-256 "
        f"`{analysis['detector_sha256'][:16]}…`). Counts are recomputed from this run's "
        "errors. Bug classes fixed after this analysis are listed in the post-hoc section "
        "below; the rest are open issues.",
        "",
    ]
    if not analysis["detector_matches"]:
        lines += [
            "> **Warning:** the detector scored here differs from the one this analysis was "
            "written for; the groups may no longer describe the errors listed below.",
            "",
        ]
    lines += ["| # | Error group | Examples | FN spans | FP spans |", "|---:|---|---:|---:|---:|"]
    for i, group in enumerate(analysis["groups"], start=1):
        ids = ", ".join(group["examples"])
        lines.append(
            f"| {i} | {group['description']} ({ids}) | {len(group['examples'])} | "
            f"{group['fn']} | {group['fp']} |"
        )
    lines += ["", f"Unclassified errors: {len(analysis['unclassified'])}.", ""]
    return lines


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _recorded_row(label: str, n_examples: int, recorded: dict[str, Any]) -> str:
    """Summary row from the recorded frozen run (when its source cannot be re-run)."""
    st, ov = recorded["overall_strict"], recorded["overall_overlap"]
    ps, rs, fs = _prf(st["tp"], st["fp"], st["fn"])
    po, ro, fo = _prf(ov["tp"], ov["fp"], ov["fn"])
    return (
        f"| {label} | blind_heldout (recorded) | {n_examples} | — | {st['tp'] + st['fn']} | "
        f"{_fmt(ps)} | {_fmt(rs)} | {_fmt(fs)} | {_fmt(po)} | {_fmt(ro)} | {_fmt(fo)} | "
        f"{recorded['negatives_altered']} | {recorded['protected_values_corrupted']} |"
    )


def _rows(label: str, ds: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """(name, role, result) for every block of a dataset, post-hoc blocks last."""
    rows = [
        (label if split == "all" else f"{label} / {split}", ds["role"], result)
        for split, result in ds["results"].items()
    ]
    if "post_hoc" in ds:
        rows += [
            (f"{label} — post-hoc" if split == "all" else f"{label} / {split} — post-hoc",
             "post-hoc (optimistic)", result)
            for split, result in ds["post_hoc"]["results"].items()
        ]
    return rows


def _frozen_lines(ds: dict[str, Any]) -> list[str]:
    detector = ds.get("detector")
    if not detector:
        return []
    lines = [f"Detector scored: SHA-256 `{detector['sha256'][:16]}…` ({detector['source']})."]
    check = ds.get("frozen_check")
    if check is not None:
        verdict = "**yes**" if check["reproduced"] else (
            f"**no** (recorded {check['recorded']}, re-computed {check['recomputed']})"
        )
        lines.append(
            "The frozen detector's source was read back from git and its SHA-256 verified; "
            f"the re-computation reproduces the recorded blind run: {verdict}."
        )
    return [*lines, ""]


def _post_hoc_lines(label: str, ds: dict[str, Any]) -> list[str]:
    post = ds["post_hoc"]
    after = post["results"]["all"]
    lines = [
        f"## {post['label']}: `{label}`",
        "",
        f"Detector SHA-256 `{post['detector_sha256'][:16]}…` (working tree). These changes "
        "were made **after** the single blind run, partly in response to errors seen in it, "
        "so the blind set is no longer unseen for this detector: the numbers below are "
        "optimistic and are **not** an independent estimate of generalisation. The frozen "
        "numbers above remain the primary blind result. The development set was re-checked "
        "with this detector (development section).",
        "",
        "### Fixes",
        "",
        *[f"{i}. {fix}" for i, fix in enumerate(post["fixes"], start=1)],
        "",
        "### Blind set: frozen vs post-hoc detector",
        "",
        "| Detector | P (strict) | R (strict) | F1 (strict) | P (overlap) | R (overlap) "
        "| F1 (overlap) | Fully masked | Negatives altered | Protected values corrupted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    compare: list[tuple[str, dict[str, Any]]] = []
    if ds["results"]:
        compare.append(("frozen (blind estimate)", ds["results"]["all"]))
    compare.append(("post-hoc (optimistic)", after))
    for name, result in compare:
        s, o, m = result["overall_strict"], result["overall_overlap"], result["masking"]
        lines.append(
            f"| {name} | {_fmt(s['precision'])} | {_fmt(s['recall'])} | {_fmt(s['f1'])} | "
            f"{_fmt(o['precision'])} | {_fmt(o['recall'])} | {_fmt(o['f1'])} | "
            f"{m['overall']['fully_masked']}/{m['overall']['gold']} | "
            f"{len(result['negatives_altered'])} | {len(result['protected_values_corrupted'])} |"
        )
    lines += [
        "",
        "### Clinical-value protection (post-hoc detector)",
        "",
        *_corruption_lines(after),
        "",
        "### Per category (post-hoc detector)",
        "",
        *_category_table(after),
        "",
    ]
    before = ds.get("post_hoc_analysis")
    remaining = post["remaining_by_group"]
    lines += [
        "### Remaining errors by blind error group",
        "",
        "| # | Frozen FN | Frozen FP | Post-hoc FN | Post-hoc FP | Examples still affected |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for i, group in enumerate(remaining["groups"], start=1):
        old = before["groups"][i - 1] if before else {"fn": "—", "fp": "—"}
        ids = ", ".join(group["examples"]) or "none"
        lines.append(
            f"| {i} | {old['fn']} | {old['fp']} | {group['fn']} | {group['fp']} | {ids} |"
        )
    lines += [
        "",
        f"Errors outside the original groups: {len(remaining['unclassified'])}"
        + (" (" + ", ".join(f"{u['id']} {u['kind']} {u['span'].split('@')[0]} `{u['value']}`"
                            for u in remaining["unclassified"]) + ")"
           if remaining["unclassified"] else "")
        + ".",
        "",
        "### Errors (strict, post-hoc detector)",
        "",
        *_error_lines(after),
        "",
    ]
    return lines


def render_markdown(meta: dict[str, Any], datasets: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# PII/PHI redaction — evaluation",
        "",
        f"Generated {meta['generated_at']} by `scripts/pii_eval.py`; reference date "
        f"{meta['reference_date']} (DOB → age); detector `{meta['detector']['path']}` "
        f"SHA-256 `{meta['detector']['sha256'][:16]}…` at commit `{meta['git_commit']}` "
        "(plus working-tree changes when the hash differs from the committed file).",
        "",
        "## Protocol",
        "",
        *[f"{i}. {step}" for i, step in enumerate(meta["protocol"], start=1)],
        "",
        "Span-level metrics. *Strict*: exact character offsets and category; *overlap*: any "
        "overlap with a gold span of the same category. Precision is 1.0 by convention when "
        "nothing is predicted. Rows marked *post-hoc* score the blind set with the detector "
        "fixed after the blind run: optimistic, not an independent estimate.",
        "",
        "## Summary",
        "",
        "| Dataset / split | Role | Examples | Negatives | Gold spans | P (strict) | R (strict) "
        "| F1 (strict) | P (overlap) | R (overlap) | F1 (overlap) | Negatives altered "
        "| Protected values corrupted |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, ds in datasets.items():
        if "frozen_unavailable" in ds:
            lines.append(_recorded_row(label, ds["n_examples"], ds["frozen_unavailable"]))
        lines += [_summary_row(name, role, result) for name, role, result in _rows(label, ds)]
    lines += [
        "",
        "### Privacy masking (category-agnostic)",
        "",
        "Whatever the predicted label: *touched* = some redaction overlaps the identifier; "
        "*fully masked* = no letter or digit of the identifier survives redaction.",
        "",
        "| Dataset / split | Gold spans | Touched | Touched rate | Fully masked "
        "| Fully masked rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, ds in datasets.items():
        for name, _role, result in _rows(label, ds):
            m = result["masking"]["overall"]
            lines.append(
                f"| {name} | {m['gold']} | {m['touched']} | {_fmt(m['touched_rate'])} | "
                f"{m['fully_masked']} | {_fmt(m['fully_masked_rate'])} |"
            )
    lines.append("")
    for label, ds in datasets.items():
        title = {
            "blind_heldout": "Blind held-out set (evaluated once, frozen detector)",
            "development": "Development set (in-sample)",
        }.get(ds["role"], "Dataset")
        if not ds["results"]:
            lines += [
                f"## {title}: `{label}`",
                "",
                "The frozen detector's source could not be read from git, so only the "
                f"recorded overall results are shown: {ds['frozen_unavailable']}.",
                "",
            ]
        else:
            overall = ds["results"]["all"]
            lines += [
                f"## {title}: `{label}`",
                "",
                f"`{ds['testset']}` — {ds['name']} v{ds['version']}, SHA-256 "
                f"`{ds['sha256'][:16]}…`, {overall['n_examples']} examples "
                f"({overall['n_positive_examples']} positive, "
                f"{overall['n_negative_examples']} negative), "
                f"{overall['overall_strict']['support']} gold spans. Mean detection "
                f"latency {overall['mean_latency_ms']:.3f} ms per message.",
                "",
                *_frozen_lines(ds),
                "### Clinical-value protection",
                "",
                *_corruption_lines(overall),
                "",
                "### Per category",
                "",
                *_category_table(overall),
                "",
            ]
            for split, result in ds["results"].items():
                if split == "all":
                    continue
                lines += [f"#### Split `{split}`", "", *_category_table(result), ""]
            if "post_hoc_analysis" in ds:
                lines += _analysis_lines(ds["post_hoc_analysis"])
            lines += ["### Errors (strict)", "", *_error_lines(overall), ""]
        if "post_hoc" in ds:
            lines += _post_hoc_lines(label, ds)
    lines += [
        f"Environment: Python {meta['python']}, {meta['machine']}.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the PII redactor.")
    parser.add_argument(
        "--dataset", action="append", type=parse_dataset, default=None,
        help="development | blind_v2 | label=path | path (repeatable; default: both named sets)",
    )
    parser.add_argument("--testset", type=parse_dataset, default=None,
                        help="deprecated alias of a single --dataset")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports")
    parser.add_argument("--print-only", action="store_true",
                        help="print the summary without writing report files")
    args = parser.parse_args(argv)

    selected: list[Dataset] = list(args.dataset or [])
    if args.testset is not None:
        selected.append(args.testset)
    if not selected:
        selected = [Dataset(name, path, ROLES[name]) for name, path in DATASETS.items()]

    datasets = {ds.label: evaluate_dataset(ds) for ds in selected}
    meta = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reference_date": EVAL_TODAY.isoformat(),
        "detector": {"path": _relative(DETECTOR), "sha256": sha256_file(DETECTOR)},
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "protocol": PROTOCOL,
    }

    for label, ds in datasets.items():
        if "frozen_unavailable" in ds:
            print(f"{label}: frozen detector source unavailable; recorded "
                  f"{ds['frozen_unavailable']}")
        for name, _role, result in _rows(label, ds):
            s = result["overall_strict"]
            print(f"{name:32s} n={result['n_examples']:3d}  spans={s['support']:3d}  "
                  f"P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}  "
                  f"negatives altered={len(result['negatives_altered'])}  "
                  f"protected corrupted={len(result['protected_values_corrupted'])}")
        check = ds.get("frozen_check")
        if check is not None and not check["reproduced"]:
            print(f"WARNING {label}: the frozen re-run does not reproduce the recorded blind "
                  f"run: {check}")
    if args.print_only:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "privacy_redaction_metrics.json"
    md_path = args.out_dir / "privacy_redaction_metrics.md"
    json_path.write_text(json.dumps({"meta": meta, "datasets": datasets}, indent=2) + "\n",
                         encoding="utf-8")
    md_path.write_text(render_markdown(meta, datasets), encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
