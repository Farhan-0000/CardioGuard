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
  author who never saw the detector, its evaluation script or its metrics. It is
  evaluated once, after the detector was frozen; the detector's SHA-256 is recorded
  so the run can be tied to the exact code.

Writes ``reports/privacy_redaction_metrics.json`` and ``.md`` with span-level
precision / recall / F1 per category (strict = exact offsets + category,
overlap = any overlap + category) for every dataset and split, plus the
clinical-value corruption check (negatives altered, protected values corrupted).
Gold spans may be given as substrings (``text`` + optional ``occurrence``) or with
explicit ``start``/``end`` offsets (v2 format); offsets are validated and honoured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from medbridge.privacy.evaluation import EVAL_TODAY, evaluate, resolve_gold  # noqa: E402
from medbridge.privacy.redaction import detect  # noqa: E402

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
    "The blind set was evaluated once with the frozen detector. The detector was not "
    "changed after seeing the results; problems found are listed as open issues instead. "
    "The report was regenerated afterwards with the same detector hash only to add the "
    "category-agnostic masking metrics and the post-hoc error analysis; every span metric "
    "is identical to the first run.",
    "Development-set numbers are in-sample and optimistic; the blind numbers are the "
    "estimate of generalisation.",
]

# Post-hoc error analysis of the single blind run, keyed by the blind file's SHA-256 so it
# is only rendered for that exact file. Members are "ID" (all errors of the example) or
# "ID:CATEGORY"; counts are recomputed from the run's errors, nothing is typed in by hand.
POST_HOC_ANALYSIS: dict[str, dict[str, Any]] = {
    "eefa4b15d601daa8fad7e1466f1c37fc9945317d40d60ae1676711262f524d2c": {
        "first_run_at": "2026-10-09T16:02:05+00:00",
        "detector_sha256": "e75b67bc708d05227fb4e9338919d06b5150530aa2f6a877dc252dc90c9c1015",
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


def masking_metrics(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Category-agnostic privacy view: is each gold identifier hidden, whatever the label?

    ``touched``: some predicted span (any category) overlaps the gold span.
    ``fully_masked``: every letter/digit of the gold span lies inside predicted spans, so
    no character of the identifier survives redaction.
    """
    per_cat: dict[str, dict[str, int]] = {}
    for ex in examples:
        text: str = ex["text"]
        preds = detect(text, today=EVAL_TODAY)
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


def evaluate_dataset(dataset: Dataset) -> dict[str, Any]:
    """Evaluate one dataset: all examples plus one block per split."""
    header, examples = load_examples(dataset.path)
    subsets: dict[str, list[dict[str, Any]]] = {"all": examples}
    splits = sorted({str(e.get("split", "dev")) for e in examples})
    if len(splits) > 1:
        for split in splits:
            subsets[split] = [e for e in examples if str(e.get("split", "dev")) == split]
    results: dict[str, dict[str, Any]] = {}
    for name, subset in subsets.items():
        result = _attach_notes(evaluate(subset).to_dict(), subset)
        result["masking"] = masking_metrics(subset)
        results[name] = result
    sha = sha256_file(dataset.path)
    out: dict[str, Any] = {
        "role": dataset.role,
        "testset": _relative(dataset.path),
        "name": header.get("name", ""),
        "version": header.get("version", "?"),
        "sha256": sha,
        "splits": header.get("splits", {}),
        "results": results,
    }
    if sha in POST_HOC_ANALYSIS:
        out["post_hoc_analysis"] = classify_errors(results["all"], POST_HOC_ANALYSIS[sha])
    return out


def classify_errors(result: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
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
        "detector_matches": analysis["detector_sha256"] == sha256_file(DETECTOR),
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
        f"`{analysis['detector_sha256'][:16]}…`). The detector was **not** changed in "
        "response; the groups below are the open issues. Counts are recomputed from this "
        "run's errors.",
        "",
    ]
    if not analysis["detector_matches"]:
        lines += [
            "> **Warning:** the current detector differs from the one this analysis was "
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
        "nothing is predicted.",
        "",
        "## Summary",
        "",
        "| Dataset / split | Role | Examples | Negatives | Gold spans | P (strict) | R (strict) "
        "| F1 (strict) | P (overlap) | R (overlap) | F1 (overlap) | Negatives altered "
        "| Protected values corrupted |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, ds in datasets.items():
        for split, result in ds["results"].items():
            name = label if split == "all" else f"{label} / {split}"
            lines.append(_summary_row(name, ds["role"], result))
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
        for split, result in ds["results"].items():
            name = label if split == "all" else f"{label} / {split}"
            m = result["masking"]["overall"]
            lines.append(
                f"| {name} | {m['gold']} | {m['touched']} | {_fmt(m['touched_rate'])} | "
                f"{m['fully_masked']} | {_fmt(m['fully_masked_rate'])} |"
            )
    lines.append("")
    for label, ds in datasets.items():
        overall = ds["results"]["all"]
        title = {
            "blind_heldout": "Blind held-out set (evaluated once, frozen detector)",
            "development": "Development set (in-sample)",
        }.get(ds["role"], "Dataset")
        lines += [
            f"## {title}: `{label}`",
            "",
            f"`{ds['testset']}` — {ds['name']} v{ds['version']}, SHA-256 "
            f"`{ds['sha256'][:16]}…`, {overall['n_examples']} examples "
            f"({overall['n_positive_examples']} positive, {overall['n_negative_examples']} "
            f"negative), {overall['overall_strict']['support']} gold spans. Mean detection "
            f"latency {overall['mean_latency_ms']:.3f} ms per message.",
            "",
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
        for split, result in ds["results"].items():
            s = result["overall_strict"]
            name = label if split == "all" else f"{label}/{split}"
            print(f"{name:22s} n={result['n_examples']:3d}  spans={s['support']:3d}  "
                  f"P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}  "
                  f"negatives altered={len(result['negatives_altered'])}  "
                  f"protected corrupted={len(result['protected_values_corrupted'])}")
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
