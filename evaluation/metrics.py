"""
Statistics for the hallucination study (CLAUDE_AGENT_PROMPT §6.4–6.5).

All functions are pure (numpy/scipy only) and documented with the exact
formula used, so the report can cite them:

* ``wilson_ci`` — Wilson score interval for a binomial proportion.
* ``bootstrap_ci`` — percentile bootstrap (default 10 000 resamples, seed 42);
  rows of a 2-D array are resampled together (cluster bootstrap), which is how
  ratio-of-sums statistics such as the unsupported-claim ratio are handled.
* ``mcnemar_exact`` — two-sided exact McNemar test (binomial on discordant pairs).
* ``cochrans_q`` — Cochran's Q for k related binary samples.
* ``holm_bonferroni`` — Holm step-down adjusted p-values.
* ``paired_risk_difference`` — difference of paired proportions with the
  Newcombe (1998, method 10) hybrid score interval.
* ``conditional_odds_ratio`` — McNemar odds ratio b/c with an exact
  Clopper–Pearson-based interval; Haldane +0.5 point estimate when b or c is 0.
* ``relative_risk_reduction`` — 1 − p_treat / p_control.
* ``cohen_kappa`` — Cohen's κ for 2+ classes with the Fleiss–Cohen–Everitt
  (1969) large-sample standard error.
* ``routing_metrics`` — strict/lenient accuracy, per-class P/R/F1, macro-F1,
  confusion matrix.
* ``risk_coverage_curve`` — selective-prediction risk–coverage curve and AURC.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

NAN = float("nan")


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided confidence interval."""

    estimate: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Proportion:
    """k successes out of n with a Wilson interval."""

    k: int
    n: int
    rate: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class McNemarResult:
    """Exact McNemar test on discordant pairs ``b`` (first only) and ``c`` (second only)."""

    b: int
    c: int
    n_discordant: int
    statistic: float  # uncorrected chi-square (b - c)^2 / (b + c), for reference
    pvalue: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class CochranQResult:
    """Cochran's Q statistic, degrees of freedom, p-value and number of complete subjects."""

    q: float
    df: int
    pvalue: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class OddsRatio:
    """Conditional (McNemar) odds ratio with an exact interval."""

    estimate: float
    low: float
    high: float
    haldane: bool

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class KappaResult:
    """Cohen's kappa with its large-sample interval."""

    kappa: float
    low: float
    high: float
    se: float
    n: int
    observed_agreement: float
    expected_agreement: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class RiskCoverage:
    """Risk–coverage curve: answering the ``k`` most confident cases, k = 1..n."""

    coverage: list[float]
    risk: list[float]
    thresholds: list[float]
    aurc: float


def _z(alpha: float) -> float:
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


# ----------------------------------------------------------------------
# Proportions
# ----------------------------------------------------------------------


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials (no continuity correction).

    centre = (p + z²/2n) / (1 + z²/n);
    half   = z·sqrt(p(1−p)/n + z²/4n²) / (1 + z²/n).
    Returns ``(nan, nan)`` when ``n == 0``.
    """
    if n <= 0:
        return (NAN, NAN)
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, n], got k={k}, n={n}")
    z = _z(alpha)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return (low, high)


def proportion(k: int, n: int, alpha: float = 0.05) -> Proportion:
    """``k/n`` with its Wilson interval (rate is NaN when ``n == 0``)."""
    low, high = wilson_ci(k, n, alpha)
    return Proportion(k=int(k), n=int(n), rate=(k / n) if n else NAN, low=low, high=high)


def bootstrap_ci(
    values: ArrayLike,
    stat: Callable[[NDArray[np.float64]], float] = np.mean,
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Interval:
    """Percentile bootstrap interval for ``stat(values)``.

    ``values`` may be 1-D (resample elements) or 2-D (resample rows, i.e. a
    cluster bootstrap — use this for ratio-of-sums statistics, e.g.
    ``stat=lambda a: a[:, 0].sum() / a[:, 1].sum()``). Resamples whose statistic
    is NaN (e.g. a zero denominator) are dropped before taking percentiles.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.shape[0] if arr.ndim else 0
    if n == 0:
        return Interval(NAN, NAN, NAN)
    estimate = float(stat(arr))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    fast = stat is np.mean and arr.ndim == 1
    chunk = max(1, min(n_boot, 2_000_000 // max(n, 1)))
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(size, n))
        if fast:
            boots[done : done + size] = arr[idx].mean(axis=1)
        else:
            boots[done : done + size] = [float(stat(arr[row])) for row in idx]
        done += size
    finite = boots[np.isfinite(boots)]
    if finite.size == 0:
        return Interval(estimate, NAN, NAN)
    low, high = np.percentile(finite, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return Interval(estimate, float(low), float(high))


# ----------------------------------------------------------------------
# Paired tests
# ----------------------------------------------------------------------


def paired_table(first: Sequence[int], second: Sequence[int]) -> tuple[int, int, int, int]:
    """2×2 paired table ``(a, b, c, d)``: a = both 1, b = first only, c = second only, d = both 0."""
    if len(first) != len(second):
        raise ValueError("paired samples must have equal length")
    x = np.asarray(first, dtype=int)
    y = np.asarray(second, dtype=int)
    a = int(np.sum((x == 1) & (y == 1)))
    b = int(np.sum((x == 1) & (y == 0)))
    c = int(np.sum((x == 0) & (y == 1)))
    d = int(np.sum((x == 0) & (y == 0)))
    return a, b, c, d


def mcnemar_exact(b: int, c: int) -> McNemarResult:
    """Two-sided exact McNemar test.

    Under H0 the ``b`` discordant pairs among ``n = b + c`` follow
    Binomial(n, 0.5); p = min(1, 2·P[X ≤ min(b, c)]). ``n == 0`` gives p = 1.
    """
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return McNemarResult(b=b, c=c, n_discordant=0, statistic=0.0, pvalue=1.0)
    p = min(1.0, 2.0 * float(stats.binom.cdf(min(b, c), n, 0.5)))
    statistic = (b - c) ** 2 / n
    return McNemarResult(b=b, c=c, n_discordant=n, statistic=float(statistic), pvalue=p)


def cochrans_q(matrix: ArrayLike) -> CochranQResult:
    """Cochran's Q test for ``k`` related binary samples.

    ``matrix`` is ``n_subjects × k_conditions`` of 0/1 values (complete cases).
    Q = (k−1)·[k·ΣC_j² − N²] / [k·N − ΣR_i²], df = k−1, p = P[χ²_df ≥ Q],
    with C_j column totals, R_i row totals and N the grand total. When every
    subject is constant across conditions the denominator is 0 and Q = 0, p = 1.
    """
    x = np.asarray(matrix, dtype=int)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("matrix must be 2-D with at least two conditions (columns)")
    if not np.isin(x, (0, 1)).all():
        raise ValueError("matrix must contain only 0/1")
    n, k = x.shape
    col = x.sum(axis=0)
    row = x.sum(axis=1)
    total = int(x.sum())
    denom = k * total - int(np.sum(row**2))
    if denom == 0:
        return CochranQResult(q=0.0, df=k - 1, pvalue=1.0, n=n)
    q = (k - 1) * (k * int(np.sum(col**2)) - total**2) / denom
    p = float(stats.chi2.sf(q, k - 1))
    return CochranQResult(q=float(q), df=k - 1, pvalue=p, n=n)


def holm_bonferroni(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as the input).

    For sorted p_(1) ≤ … ≤ p_(m): adj_(i) = max_{j ≤ i} min(1, (m − j + 1)·p_(j)).
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * float(pvalues[idx]))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


# ----------------------------------------------------------------------
# Effect sizes
# ----------------------------------------------------------------------


def paired_risk_difference(
    treatment: Sequence[int], control: Sequence[int], alpha: float = 0.05
) -> Interval:
    """Paired risk difference p_treatment − p_control with Newcombe's method-10 interval.

    With the paired table (a, b, c, d) of ``treatment`` × ``control``,
    p1 = (a+b)/n, p2 = (a+c)/n, D = p1 − p2, Wilson limits (l1, u1), (l2, u2),
    and φ = (ad − bc)/√((a+b)(c+d)(a+c)(b+d)) (0 when undefined):

        L = D − √((p1−l1)² − 2φ(p1−l1)(u2−p2) + (u2−p2)²)
        U = D + √((u1−p1)² − 2φ(u1−p1)(p2−l2) + (p2−l2)²)

    Newcombe RG (1998) Stat Med 17:2635–2650.
    """
    a, b, c, d = paired_table(treatment, control)
    n = a + b + c + d
    if n == 0:
        return Interval(NAN, NAN, NAN)
    p1 = (a + b) / n
    p2 = (a + c) / n
    l1, u1 = wilson_ci(a + b, n, alpha)
    l2, u2 = wilson_ci(a + c, n, alpha)
    denom = math.sqrt(float((a + b) * (c + d) * (a + c) * (b + d)))
    phi = (a * d - b * c) / denom if denom > 0 else 0.0
    diff = p1 - p2
    lower_term = (p1 - l1) ** 2 - 2.0 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2
    upper_term = (u1 - p1) ** 2 - 2.0 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2
    low = diff - math.sqrt(max(lower_term, 0.0))
    high = diff + math.sqrt(max(upper_term, 0.0))
    return Interval(diff, max(-1.0, low), min(1.0, high))


def conditional_odds_ratio(b: int, c: int, alpha: float = 0.05) -> OddsRatio:
    """McNemar conditional odds ratio ``b / c`` with an exact interval.

    The interval transforms the Clopper–Pearson interval of π = b/(b+c) with
    OR = π/(1−π) (upper limit +inf when c = 0, lower limit 0 when b = 0). When
    b or c is zero the point estimate uses the Haldane correction
    (b+0.5)/(c+0.5). Returns NaNs when there are no discordant pairs.
    """
    n = b + c
    if n == 0:
        return OddsRatio(NAN, NAN, NAN, haldane=False)
    haldane = b == 0 or c == 0
    estimate = (b + 0.5) / (c + 0.5) if haldane else b / c
    pi_low = float(stats.beta.ppf(alpha / 2.0, b, c + 1)) if b > 0 else 0.0
    pi_high = float(stats.beta.ppf(1.0 - alpha / 2.0, b + 1, c)) if c > 0 else 1.0
    low = pi_low / (1.0 - pi_low)
    high = math.inf if pi_high >= 1.0 else pi_high / (1.0 - pi_high)
    return OddsRatio(estimate=float(estimate), low=float(low), high=float(high), haldane=haldane)


def relative_risk_reduction(p_treatment: float, p_control: float) -> float:
    """RRR = 1 − p_treatment / p_control (NaN when p_control is 0 or undefined)."""
    if not math.isfinite(p_control) or p_control <= 0 or not math.isfinite(p_treatment):
        return NAN
    return 1.0 - p_treatment / p_control


# ----------------------------------------------------------------------
# Agreement
# ----------------------------------------------------------------------


def cohen_kappa(
    rater_a: Sequence[str],
    rater_b: Sequence[str],
    labels: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> KappaResult:
    """Cohen's κ = (p_o − p_e)/(1 − p_e) for two raters and 2+ categories.

    Large-sample SE (Fleiss, Cohen & Everitt 1969), with p_ij cell proportions
    and marginals p_i. (rater A) / p_.j (rater B)::

        A = Σ_i p_ii [1 − (p_i. + p_.i)(1 − κ)]²
        B = (1 − κ)² Σ_{i≠j} p_ij (p_.i + p_j.)²
        C = [κ − p_e (1 − κ)]²
        Var(κ) = (A + B − C) / (n (1 − p_e)²)

    The interval is κ ± z·SE clipped to [−1, 1]. κ is NaN when p_e = 1.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must label the same number of items")
    n = len(rater_a)
    cats = list(labels) if labels is not None else sorted(set(rater_a) | set(rater_b))
    if n == 0 or not cats:
        return KappaResult(NAN, NAN, NAN, NAN, n, NAN, NAN)
    index = {lab: i for i, lab in enumerate(cats)}
    table = np.zeros((len(cats), len(cats)), dtype=float)
    for x, y in zip(rater_a, rater_b, strict=True):
        table[index[x], index[y]] += 1.0
    p = table / n
    row = p.sum(axis=1)
    col = p.sum(axis=0)
    p_o = float(np.trace(p))
    p_e = float(np.sum(row * col))
    if math.isclose(p_e, 1.0):
        return KappaResult(NAN, NAN, NAN, NAN, n, p_o, p_e)
    kappa = (p_o - p_e) / (1.0 - p_e)
    diag = np.diag(p)
    term_a = float(np.sum(diag * (1.0 - (row + col) * (1.0 - kappa)) ** 2))
    off = p.copy()
    np.fill_diagonal(off, 0.0)
    weights = (col[:, None] + row[None, :]) ** 2  # (p_.i + p_j.)² for cell (i, j)
    term_b = (1.0 - kappa) ** 2 * float(np.sum(off * weights))
    term_c = (kappa - p_e * (1.0 - kappa)) ** 2
    var = max(term_a + term_b - term_c, 0.0) / (n * (1.0 - p_e) ** 2)
    se = math.sqrt(var)
    z = _z(alpha)
    return KappaResult(
        kappa=float(kappa),
        low=max(-1.0, kappa - z * se),
        high=min(1.0, kappa + z * se),
        se=se,
        n=n,
        observed_agreement=p_o,
        expected_agreement=p_e,
    )


# ----------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------


def confusion_matrix(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]
) -> list[list[int]]:
    """Rows = gold label, columns = predicted label, in ``labels`` order (others ignored)."""
    index = {lab: i for i, lab in enumerate(labels)}
    mat = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred, strict=True):
        if g in index and p in index:
            mat[index[g]][index[p]] += 1
    return mat


def routing_metrics(
    gold: Sequence[str],
    pred: Sequence[str],
    acceptable: Sequence[Sequence[str]] | None = None,
    labels: Sequence[str] = ("ANSWER", "SOFT-ASK", "GENERALIZE", "ABSTAIN", "ESCALATE"),
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Routing accuracy and per-class metrics.

    * strict accuracy — ``pred == gold`` (primary gold action);
    * lenient accuracy — ``pred in acceptable`` (defaults to strict);
    * per class — precision tp/(tp+fp) (0 when nothing predicted), recall
      tp/support, F1 harmonic mean (0 when P + R = 0);
    * macro-F1 — unweighted mean F1 over classes with gold support > 0;
    * confusion — rows gold, columns predicted.
    """
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have equal length")
    n = len(gold)
    strict_k = sum(1 for g, p in zip(gold, pred, strict=True) if g == p)
    if acceptable is None:
        lenient_k = strict_k
    else:
        lenient_k = sum(1 for p, acc in zip(pred, acceptable, strict=True) if p in acc)
    per_class: dict[str, dict[str, float | int]] = {}
    f1s: list[float] = []
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == lab and p == lab)
        predicted = sum(1 for p in pred if p == lab)
        support = sum(1 for g in gold if g == lab)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else NAN
        if support and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if support else NAN
        per_class[lab] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
        }
        if support:
            f1s.append(f1)
    strict = proportion(strict_k, n, alpha)
    lenient = proportion(lenient_k, n, alpha)
    return {
        "n": n,
        "accuracy_strict": strict.as_dict(),
        "accuracy_lenient": lenient.as_dict(),
        "per_class": per_class,
        "macro_f1": float(np.mean(f1s)) if f1s else NAN,
        "labels": list(labels),
        "confusion": confusion_matrix(gold, pred, labels),
    }


# ----------------------------------------------------------------------
# Selective prediction
# ----------------------------------------------------------------------


def risk_coverage_curve(confidence: Sequence[float], loss: Sequence[float]) -> RiskCoverage:
    """Risk–coverage curve for selective prediction.

    Cases are sorted by decreasing ``confidence``; answering the top ``k``
    gives coverage k/n and risk = mean loss of those k. AURC is the mean of
    the risk over the n operating points.
    """
    if len(confidence) != len(loss):
        raise ValueError("confidence and loss must have equal length")
    n = len(confidence)
    if n == 0:
        return RiskCoverage([], [], [], NAN)
    conf = np.asarray(confidence, dtype=float)
    los = np.asarray(loss, dtype=float)
    order = np.argsort(-conf, kind="stable")
    cum = np.cumsum(los[order])
    k = np.arange(1, n + 1)
    risk = cum / k
    return RiskCoverage(
        coverage=(k / n).tolist(),
        risk=risk.tolist(),
        thresholds=conf[order].tolist(),
        aurc=float(np.mean(risk)),
    )


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (NaN for empty input)."""
    if not values:
        return NAN
    return float(np.percentile(np.asarray(values, dtype=float), q))
