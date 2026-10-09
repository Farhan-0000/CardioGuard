#!/usr/bin/env python3
"""
Evaluate the PII/PHI redactor on the synthetic test set.

    python scripts/pii_eval.py
    python scripts/pii_eval.py --testset tests/privacy/pii_testset.json --out-dir reports

Writes ``reports/privacy_redaction_metrics.json`` and ``.md`` with span-level
precision / recall / F1 per category (strict = exact offsets + category,
overlap = any overlap + category), for the whole set and for each split
(``dev`` was authored with the detector, ``heldout`` after it was frozen),
plus the clinical-value corruption check on negatives.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from medbridge.privacy.evaluation import EVAL_TODAY, evaluate, load_testset  # noqa: E402


def _table(per_category: dict[str, dict[str, Any]]) -> list[str]:
    rows = [
        "| Category | Support | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cat, m in per_category.items():
        if m["support"] == 0 and m["fp"] == 0:
            continue
        rows.append(
            f"| {cat} | {m['support']} | {m['tp']} | {m['fp']} | {m['fn']} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
        )
    return rows


def _overall_row(name: str, result: dict[str, Any]) -> str:
    s, o = result["overall_strict"], result["overall_overlap"]
    return (
        f"| {name} | {result['n_examples']} | {result['n_negative_examples']} | {s['support']} | "
        f"{s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | "
        f"{o['precision']:.3f} | {o['recall']:.3f} | {len(result['negatives_altered'])} |"
    )


def render_markdown(results: dict[str, dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = [
        "# PII/PHI redaction — evaluation",
        "",
        f"Generated {meta['generated_at']} by `scripts/pii_eval.py` on "
        f"`{meta['testset']}` (v{meta['testset_version']}), reference date {meta['reference_date']}.",
        "",
        "Span-level metrics. *Strict*: exact character offsets and category; "
        "*overlap*: any overlap with a gold span of the same category. "
        "The `dev` split was written together with the detector (in-sample); the `heldout` "
        "split was written after the detector was frozen and never used for tuning.",
        "",
        "| Split | Examples | Negatives | Gold spans | P (strict) | R (strict) | F1 (strict) "
        "| P (overlap) | R (overlap) | Negatives altered |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        lines.append(_overall_row(name, result))
    overall = results["all"]
    lines += [
        "",
        f"Clinical-value protection: {len(overall['negatives_altered'])} of "
        f"{overall['n_negative_examples']} clinically dense negative messages altered; "
        f"{len(overall['protected_values_corrupted'])} protected clinical values "
        "(BP readings, doses, drug names, lab values, ages) corrupted across all examples.",
        "",
        f"Mean detection latency: {overall['mean_latency_ms']:.3f} ms per message "
        f"({meta['python']}, {meta['machine']}).",
        "",
        "## Per category (all examples, strict)",
        "",
        *_table(overall["per_category_strict"]),
        "",
        "## Per category (held-out split, strict)",
        "",
        *(_table(results["heldout"]["per_category_strict"]) if "heldout" in results else []),
        "",
        "## Errors (strict)",
        "",
    ]
    errors = overall["errors"]
    if not errors:
        lines.append("None.")
    for err in errors:
        kinds = ", ".join(f"{e['kind']} {e['span'].split('@')[0]} `{e['value']}`"
                          for e in err["errors"])
        lines.append(f"- **{err['id']}** — {kinds}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the PII redactor.")
    parser.add_argument("--testset", type=Path,
                        default=REPO_ROOT / "tests" / "privacy" / "pii_testset.json")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports")
    args = parser.parse_args(argv)

    data = json.loads(args.testset.read_text(encoding="utf-8"))
    examples = load_testset(args.testset)
    results: dict[str, dict[str, Any]] = {"all": evaluate(examples).to_dict()}
    for split in sorted({str(e.get("split", "dev")) for e in examples}):
        subset = [e for e in examples if str(e.get("split", "dev")) == split]
        results[split] = evaluate(subset).to_dict()

    try:
        testset_label = str(args.testset.resolve().relative_to(REPO_ROOT))
    except ValueError:
        testset_label = str(args.testset)
    meta = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "testset": testset_label,
        "testset_version": data.get("version", "?"),
        "reference_date": EVAL_TODAY.isoformat(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "privacy_redaction_metrics.json"
    md_path = args.out_dir / "privacy_redaction_metrics.md"
    json_path.write_text(json.dumps({"meta": meta, "results": results}, indent=2) + "\n",
                         encoding="utf-8")
    md_path.write_text(render_markdown(results, meta), encoding="utf-8")

    for name, result in results.items():
        s = result["overall_strict"]
        print(f"{name:8s} n={result['n_examples']:3d}  P={s['precision']:.3f}  "
              f"R={s['recall']:.3f}  F1={s['f1']:.3f}  "
              f"negatives altered={len(result['negatives_altered'])}")
    print(f"Wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
