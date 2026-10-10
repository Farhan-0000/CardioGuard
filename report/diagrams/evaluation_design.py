"""
Evaluation design — the calibration-gated hallucination study protocol.

Sources (protocol enforced in code, see ``evaluation/README.md``):

* dataset ``evaluation/datasets/hallubench_v1.json`` (counts computed here);
  split = ``stratified_split(dev_fraction=0.2, seed=42)`` (``datasets/schema.py``);
* conditions = ``AblationConfig.all_names()`` in ``medbridge/core/pipeline_types.py``
  (plus the optional ``C1G`` reference);
* runner ``evaluation/run.py`` (V1/V2/V12 refuse the test split without
  ``results/calibration.json``), judges ``evaluation/judges/llm_judge.py``
  (default ``gemma3:12b``) and ``nli_judge.py`` (``cross-encoder/nli-deberta-v3-base``),
  ``evaluation/calibrate.py``, ``evaluation/analyze.py`` (statistics, > 5 % degraded
  turns blocks headline results), ``evaluation/human_annotation.py`` (50 items).

No results are drawn — this is the design only.

Run: ``python report/diagrams/evaluation_design.py`` →
``report/figures/evaluation_design.{png,svg}``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import PALETTE, REPO_ROOT, save_matplotlib
from _mpl import Canvas

sys.path.insert(0, str(REPO_ROOT))
from medbridge.core.pipeline_types import AblationConfig

NAME = "evaluation_design"
DATASET = REPO_ROOT / "evaluation" / "datasets" / "hallubench_v1.json"
AMBER = PALETTE["gate"].line
BLUE = PALETTE["llm"].line
PURPLE = PALETTE["storage"].line

CONDITION_TEXT = {
    "C0": "C0  direct LLM (no retrieval, no gates)",
    "C1": "C1  vanilla RAG",
    "C2": "C2  RAG + prompt-only safety",
    "C3": "C3  Context Gate only",
    "C4": "C4  Evidence Gate only",
    "C5": "C5  MedBridge: both gates",
    "V1": "V1  C5 + calibrated Evidence Gate",
    "V2": "V2  C5 + Gate 3 grounding verifier",
    "V12": "V12  V1 + V2",
}
CATEGORY_LABELS = [
    ("answerable_complete", "answerable"),
    ("answerable_missing_context", "missing context"),
    ("general_topic", "general topic"),
    ("unanswerable_in_domain", "unanswerable"),
    ("out_of_domain", "out of domain"),
    ("emergency", "emergency"),
]


def dataset_facts() -> dict[str, object]:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    items = data["items"]
    cats = Counter(i["category"] for i in items)
    splits = Counter(i.get("split") for i in items)
    return {
        "version": data.get("version", "1"),
        "items": len(items),
        "turns": sum(len(i["turns"]) for i in items),
        "multi": sum(1 for i in items if len(i["turns"]) > 1),
        "cats": cats,
        "dev": splits.get("dev", 0),
        "test": splits.get("test", 0),
    }


def build() -> list[Path]:
    f = dataset_facts()
    names = AblationConfig.all_names()
    missing = [n for n in names if n not in CONDITION_TEXT]
    if missing:
        raise SystemExit(f"New ablation presets need a label: {missing}")
    cats = f["cats"]
    assert isinstance(cats, Counter)
    cat_txt = [f"{label} {cats.get(key, 0)}" for key, label in CATEGORY_LABELS]

    c = Canvas(7.5, 8.75)
    lab = c.label
    XL, XR, WL, WR = 1.85, 5.5, 3.0, 3.3

    hb = c.box(
        3.7,
        0.55,
        5.4,
        0.78,
        f"MedBridge-HalluBench v{f['version']} · synthetic queries, no patient data",
        [
            f"{f['items']} items · {f['turns']} turns ({f['multi']} multi-turn items) · 6 categories:",
            " · ".join(cat_txt[:3]),
            " · ".join(cat_txt[3:]),
        ],
        kind="storage",
        sub_size=6.8,
    )
    dev_note = f"{f['dev']} items · model selection and calibration only"
    dev = c.box(XL, 1.62, WL, 0.48, "DEV split", [dev_note])
    test = c.box(XR, 1.62, WR, 0.48, "TEST split", [f"{f['test']} items · every reported result"])
    c.arrow([(2.6, hb.bottom), (2.6, 1.17), (XL, 1.17), (XL, dev.top)])
    c.arrow([(4.8, hb.bottom), (4.8, 1.17), (XR, 1.17), (XR, test.top)])
    lab(3.7, 1.17, "stratified by category · seed 42 · 20 % dev", size=6.6)

    # ---- dev / calibration column ------------------------------------------------------
    d1 = c.box(XL, 2.45, WL, 0.5, "Run C5 on DEV", ["python -m evaluation.run --split dev"])
    d2 = c.box(
        XL,
        3.27,
        WL,
        0.5,
        "Judge DEV responses",
        ["LLM judge, then NLI judge on the same claims"],
        kind="llm",
    )
    cal = c.box(
        XL,
        4.27,
        WL,
        0.86,
        "Calibrate on DEV only",
        [
            "python -m evaluation.calibrate → calibration.json",
            "NLI judge θ, θc (agreement with LLM judge)",
            "V1 evidence thresholds · V2 τ + entailment",
        ],
        kind="det",
    )
    for a, b in ((dev, d1), (d1, d2), (d2, cal)):
        c.arrow([(XL, a.bottom), (XL, b.top)])

    # ---- test column -------------------------------------------------------------------------
    cond = c.box(
        XR,
        2.95,
        WR,
        1.78,
        f"{len(names)} conditions (AblationConfig presets)",
        [CONDITION_TEXT[n] for n in names]
        + ["(optional C1G: RAG + production prompt)", "fast-path on in all except C0"],
        kind="gate",
        sub_size=6.6,
    )
    run = c.box(
        XR,
        4.55,
        WR,
        0.92,
        "python -m evaluation.run --split test",
        [
            "same app model, retrieval index and gate prompts",
            "LLM cache keyed by model digest · resumable",
            "degraded turns flagged and retried",
            "→ runs/<COND>.jsonl + manifest",
        ],
        sub_size=6.6,
    )
    jd = c.box(
        XR,
        5.82,
        WR,
        0.98,
        "Two judges of different kinds",
        [
            "LLM judge (gemma3:12b, T 0): atomic claims →",
            "SUPPORTED · UNSUPPORTED · CONTRADICTED",
            "+ behaviour flags; own + judge-side evidence",
            "NLI judge (nli-deberta-v3-base): same claims",
        ],
        kind="llm",
        sub_size=6.6,
    )
    an = c.box(
        XR,
        7.12,
        WR,
        1.0,
        "python -m evaluation.analyze",
        [
            "Wilson 95 % CI · cluster bootstrap (10k)",
            "exact McNemar + Holm · Cochran's Q",
            "risk difference, odds ratio, RRR · Cohen's κ",
            "degraded turns excluded; > 5 % blocks headline",
        ],
        sub_size=6.6,
    )
    out = c.box(
        XR,
        8.25,
        WR,
        0.6,
        "Outputs (every reported number)",
        [
            "summary.json · tables · error_analysis.md",
            "figures → reports/figures (PNG 300 dpi + SVG)",
        ],
        kind="storage",
        sub_size=6.5,
    )
    for a, b in ((test, cond), (cond, run), (run, jd), (jd, an), (an, out)):
        c.arrow([(XR, a.bottom), (XR, b.top)])

    # calibration feeds the test run (V1/V2/V12) and the analysis (NLI thresholds)
    c.arrow([(cal.right, 4.45), (run.left, 4.45)], color=AMBER, lw=1.4)
    lab((cal.right + run.left) / 2, 4.33, "V1 / V2", size=6.3, color=AMBER, bg=False)
    lab((cal.right + run.left) / 2, 4.58, "required", size=6.0, color=AMBER, bg=False)
    c.arrow([(0.24, cal.cy), (0.24, 7.32), (an.left, 7.32)], color=AMBER, lw=1.4)
    c.arrow([(cal.left, cal.cy), (0.24, cal.cy)], color=AMBER, lw=1.4, head=False)
    lab(0.24, 5.9, "NLI θ, θc for the analysis", size=6.3, color=AMBER, rotation=90)

    # human subset
    hs = c.box(
        XL,
        5.72,
        WL - 0.25,
        0.62,
        "Human subset (blinded)",
        ["50 test items, C5 + C1 responses", "human_annotation --export → sheet"],
        kind="actor",
    )
    hk = c.box(
        XL,
        6.88,
        WL - 0.25,
        0.56,
        "Team annotates (A1, A2)",
        ["--score → human–human, human–judge κ"],
        kind="actor",
    )
    c.arrow([(run.left, 4.86), (2.3, 4.86), (2.3, hs.top)])
    lab(3.05, 4.97, "responses", size=6.3, bg=False)
    c.arrow([(XL, hs.bottom), (XL, hk.top)])
    c.arrow([(hk.right, hk.cy), (an.left, hk.cy)])
    lab((hk.right + an.left) / 2, hk.cy - 0.1, "κ", size=6.6)

    c.legend(
        8.72,
        [
            ("storage", "Data / result files"),
            ("gate", "Gate ablation"),
            ("llm", "LLM-based step"),
            ("det", "Deterministic step"),
            ("actor", "Human step"),
        ],
        per_row=5,
        x0=0.15,
        col_w=1.47,
        size=6.5,
    )
    c.check()
    c.resize(8.95)
    return save_matplotlib(c.fig, NAME)


if __name__ == "__main__":
    for p in build():
        print(p.relative_to(REPO_ROOT))
