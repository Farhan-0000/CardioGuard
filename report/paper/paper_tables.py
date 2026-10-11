"""
Paper-specific generated tables for Section V of the IEEE paper.

The report's study tables (``report/lib/tables.py``) are built for a full page: they carry
every column of ``evaluation/results/summary.json`` and show the NLI judge's partial C2
subset (76 of 239 responses) in the same column as the complete conditions. A conference
column is 3.5 in wide and the paper may not present that subset as a condition result, so
the paper uses two compact tables generated from the same ``summary.json`` here:

``paper_outcomes``
    the primary outcomes per condition (LLM judge, the only judge that scored all ten
    conditions): response-level hallucination and unsafe-answer rates with Wilson 95%
    intervals, each with the Holm-adjusted exact McNemar p-value of that condition against
    C5 (every test in ``tests.mcnemar`` pairs one condition with C5), coverage and
    appropriate abstention. This is Table I of the paper; it replaced the separate test
    table to meet the 6-page limit.
``paper_tests``
    all exact McNemar tests in ``tests.mcnemar`` (the analysis computes them for the LLM
    judge only), with b / c, the paired risk difference in percentage points and the
    Holm-adjusted p-value in scientific notation. No longer used by the 6-page paper; kept
    for a supplement or a longer version.

Use them like the report's generators: ``[[table:paper_outcomes|Caption {#tab:outcomes}]]``.
:func:`generate` tries these keys first and falls back to ``report/lib/tables.py``, so every
report table key keeps working in the paper. Nothing here is typed by hand: a missing or
malformed ``summary.json`` gives ``None`` and the builders show ``[TABLE PENDING: ...]``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from lib.content import Table
from lib.results_context import ResultsContext
from lib.tables import DEFAULT_CAPTIONS as REPORT_CAPTIONS
from lib.tables import generate as report_generate

MINUS = "−"
SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

#: Order of the conditions in the paper (the analysis' condition order).
CONDITIONS = ["C0", "C1", "C1G", "C2", "C3", "C4", "C5", "V1", "V2", "V12"]

#: Rows of the test table: (treatment, control) in reading order; each gets two outcomes.
TEST_ROWS: list[tuple[str, str]] = [
    ("C5", "C0"),
    ("C5", "C1"),
    ("C5", "C2"),
    ("C5", "C3"),
    ("C5", "C4"),
    ("C5", "C1G"),
    ("V1", "C5"),
    ("V2", "C5"),
    ("V12", "C5"),
]
OUTCOMES = [("hallucination", "Halluc."), ("unsafe", "Unsafe")]

PAPER_CAPTIONS: dict[str, str] = {
    "paper_outcomes": (
        "Primary outcomes on the test split (LLM judge) and paired exact McNemar tests "
        "against C5"
    ),
    "paper_tests": "Paired exact McNemar tests (LLM judge, Holm-adjusted)",
}


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _neg(text: str) -> str:
    return text.replace("-", MINUS)


def rate_ci(p: Any, digits: int = 1) -> str:
    """Wilson proportion ``{rate, low, high}`` -> ``37.2 [31.4, 43.5]`` (percent)."""
    if not isinstance(p, dict) or not _finite(p.get("rate")):
        return "—"
    text = f"{100 * p['rate']:.{digits}f}"
    if _finite(p.get("low")) and _finite(p.get("high")):
        text += f" [{100 * p['low']:.{digits}f}, {100 * p['high']:.{digits}f}]"
    return text


def rate(p: Any, digits: int = 1) -> str:
    if not isinstance(p, dict) or not _finite(p.get("rate")):
        return "—"
    return f"{100 * p['rate']:.{digits}f}"


def risk_difference_pp(d: Any) -> str:
    """Paired risk difference ``{estimate, low, high}`` (fraction) -> ``−35.6 [−42.7, −27.7]``."""
    if not isinstance(d, dict) or not _finite(d.get("estimate")):
        return "—"
    text = f"{100 * d['estimate']:.1f}"
    if _finite(d.get("low")) and _finite(d.get("high")):
        text += f" [{100 * d['low']:.1f}, {100 * d['high']:.1f}]"
    return _neg(text)


def pvalue_sci(p: Any) -> str:
    """Two significant figures; below 0.001 as ``5.4×10⁻¹⁶`` (Unicode superscripts)."""
    if not _finite(p):
        return "—"
    if p >= 0.01:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.4f}"
    if p <= 0:
        return "0"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"{mantissa}×10{str(int(exponent)).translate(SUPERSCRIPT)}"


# ----------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------


def _table(
    key: str,
    header: list[str],
    rows: list[list[str]],
    note: str,
    align: list[str],
    widths: list[float],
) -> Table | None:
    if not rows:
        return None
    return Table(
        caption=PAPER_CAPTIONS[key],
        header=header,
        rows=rows,
        align=align,
        markup=False,
        note=note,
        key=key,
        col_widths=widths,
    )


def _llm_tests(ctx: ResultsContext) -> dict[tuple[Any, Any, Any], dict[str, Any]]:
    """LLM-judge McNemar tests of ``tests.mcnemar`` by (treatment, control, outcome)."""
    tests = ctx.get("study.tests.mcnemar")
    if not isinstance(tests, list):
        return {}
    return {
        (t.get("treatment"), t.get("control"), t.get("outcome")): t
        for t in tests
        if isinstance(t, dict) and str(t.get("judge", "")).lower() == "llm"
    }


def _vs_c5(index: dict[tuple[Any, Any, Any], dict[str, Any]], cond: str, outcome: str) -> str:
    """Holm-adjusted p of the test pairing ``cond`` with C5 (either direction)."""
    if cond == "C5":
        return "—"
    t = index.get(("C5", cond, outcome)) or index.get((cond, "C5", outcome))
    return pvalue_sci(t.get("pvalue_holm")) if t else "—"


def _odd_counts(counts: dict[str, Any]) -> tuple[Any, str]:
    """Most common count and the conditions that differ from it (``"C0: 63"``)."""
    values = [v for v in counts.values() if v]
    if not values:
        return "—", ""
    typical = max(set(values), key=values.count)
    odd = ", ".join(f"{c}: {v}" for c, v in counts.items() if v and v != typical)
    return typical, odd


def paper_outcomes(ctx: ResultsContext) -> Table | None:
    per = ctx.get("study.per_condition")
    if not isinstance(per, dict):
        return None
    index = _llm_tests(ctx)
    rows: list[list[str]] = []
    for cond in CONDITIONS:
        m = per.get(cond)
        if not isinstance(m, dict):
            continue
        llm = m.get("llm") or {}
        rows.append(
            [
                cond,
                rate_ci(llm.get("hallucination")),
                _vs_c5(index, cond, "hallucination"),
                rate_ci(llm.get("unsafe_answer")),
                _vs_c5(index, cond, "unsafe"),
                rate(m.get("coverage_answerable")),
                rate(m.get("appropriate_abstention")),
            ]
        )

    def counts(get: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        return {c: get(per.get(c) or {}) for c in CONDITIONS}

    typical_h, odd_h = _odd_counts(
        counts(lambda m: ((m.get("llm") or {}).get("hallucination") or {}).get("n"))
    )
    typical_u, odd_u = _odd_counts(
        counts(lambda m: ((m.get("llm") or {}).get("unsafe_answer") or {}).get("n"))
    )
    typical_c, odd_c = _odd_counts(
        counts(lambda m: (m.get("coverage_answerable") or {}).get("n"))
    )
    note = (
        f"Source: evaluation/results/summary.json. Halluc.: {typical_h} judged turns"
        f"{f' ({odd_h})' if odd_h else ''}; Unsafe and Abst. (appropriate abstention): "
        f"{typical_u} unanswerable turns{f' ({odd_u})' if odd_u else ''}; Cov. (coverage): "
        f"{typical_c} answerable turns{f' ({odd_c})' if odd_c else ''}. p: exact McNemar "
        "test against C5, Holm-adjusted within each family and outcome (C5 vs C0 to C4; "
        "V1, V2, V12 vs C5); C5 vs C1G unadjusted."
    )
    table = _table(
        "paper_outcomes",
        ["Cond.", "Halluc. % [95% CI]", "p", "Unsafe % [95% CI]", "p", "Cov. %", "Abst. %"],
        rows,
        note,
        ["l", "c", "c", "c", "c", "c", "c"],
        [0.30, 0.88, 0.40, 0.88, 0.40, 0.29, 0.32],
    )
    if table is not None:
        table.font_size = 7.0
    return table


def paper_tests(ctx: ResultsContext) -> Table | None:
    tests = ctx.get("study.tests.mcnemar")
    if not isinstance(tests, list) or not tests:
        return None
    index = {
        (t.get("treatment"), t.get("control"), t.get("outcome")): t
        for t in tests
        if isinstance(t, dict) and str(t.get("judge", "")).lower() == "llm"
    }
    rows: list[list[str]] = []
    for treatment, control in TEST_ROWS:
        first = True
        for outcome, label in OUTCOMES:
            t = index.get((treatment, control, outcome))
            if t is None:
                continue
            rows.append(
                [
                    f"{treatment} vs {control}" if first else "",
                    label,
                    f"{t.get('b_treatment_only')} / {t.get('c_control_only')}",
                    risk_difference_pp(t.get("risk_difference")),
                    pvalue_sci(t.get("pvalue_holm")),
                ]
            )
            first = False

    def span(outcome: str) -> str:
        ns = sorted({int(t["n_pairs"]) for k, t in index.items() if k[2] == outcome})
        if not ns:
            return "—"
        return str(ns[0]) if ns[0] == ns[-1] else f"{ns[0]}–{ns[-1]}"

    note = (
        f"Source: evaluation/results/summary.json. Pairs: {span('hallucination')} "
        f"(halluc.), {span('unsafe')} (unsafe). b / c: events only under the first / second "
        "condition. Holm correction within each family and outcome: C5 vs C0, C1, C2, C3, C4; "
        "V1, V2, V12 vs C5; C5 vs C1G (one test per outcome, unadjusted)."
    )
    return _table(
        "paper_tests",
        ["Comparison", "Outcome", "b / c", "RD, pp [95% CI]", "p (Holm)"],
        rows,
        note,
        ["l", "l", "c", "c", "c"],
        [0.72, 0.5, 0.45, 1.15, 0.6],
    )


PAPER_GENERATORS: dict[str, Callable[[ResultsContext], Table | None]] = {
    "paper_outcomes": paper_outcomes,
    "paper_tests": paper_tests,
}
DEFAULT_CAPTIONS: dict[str, str] = {**REPORT_CAPTIONS, **PAPER_CAPTIONS}


def generate(key: str, ctx: ResultsContext, errors: list[str] | None = None) -> Table | None:
    """Paper generators first, then the report's; errors yield ``None`` (pending)."""
    fn = PAPER_GENERATORS.get(key)
    if fn is None:
        return report_generate(key, ctx, errors)
    try:
        return fn(ctx)
    except Exception as exc:  # a malformed results file must never break the paper
        if errors is not None:
            errors.append(f"table '{key}': {type(exc).__name__}: {exc}")
        return None
