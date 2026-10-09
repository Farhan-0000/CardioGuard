"""Statistics for the hallucination study — known values and scipy/sklearn cross-checks."""

import math

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from evaluation import metrics as m


class TestWilson:
    @pytest.mark.parametrize("k,n", [(0, 10), (3, 10), (10, 10), (17, 240), (1, 1), (120, 240)])
    def test_matches_scipy(self, k, n):
        ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        low, high = m.wilson_ci(k, n)
        assert low == pytest.approx(ref.low, abs=1e-9)
        assert high == pytest.approx(ref.high, abs=1e-9)

    def test_known_value(self):
        # Classic textbook example: 81/263 → (0.2553, 0.3662)
        low, high = m.wilson_ci(81, 263)
        assert low == pytest.approx(0.2553, abs=1e-4)
        assert high == pytest.approx(0.3662, abs=1e-4)

    def test_empty_and_invalid(self):
        low, high = m.wilson_ci(0, 0)
        assert math.isnan(low) and math.isnan(high)
        with pytest.raises(ValueError):
            m.wilson_ci(5, 3)

    def test_proportion(self):
        p = m.proportion(3, 12)
        assert p.rate == pytest.approx(0.25)
        assert p.low < 0.25 < p.high


class TestBootstrap:
    def test_mean_interval_close_to_scipy(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5.0, 2.0, size=200)
        ours = m.bootstrap_ci(x, n_boot=4000, seed=1)
        ref = stats.bootstrap((x,), np.mean, n_resamples=4000, method="percentile", random_state=1)
        assert ours.estimate == pytest.approx(float(np.mean(x)))
        assert ours.low == pytest.approx(ref.confidence_interval.low, abs=0.08)
        assert ours.high == pytest.approx(ref.confidence_interval.high, abs=0.08)

    def test_deterministic_for_seed(self):
        x = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1]
        assert m.bootstrap_ci(x, seed=7, n_boot=500) == m.bootstrap_ci(x, seed=7, n_boot=500)

    def test_ratio_of_sums_cluster(self):
        rows = np.array([[1, 4], [0, 3], [2, 2], [0, 5], [1, 1]], dtype=float)
        res = m.bootstrap_ci(rows, stat=lambda a: a[:, 0].sum() / a[:, 1].sum(), n_boot=2000)
        assert res.estimate == pytest.approx(4 / 15)
        assert 0.0 <= res.low <= res.estimate <= res.high <= 1.0

    def test_empty(self):
        res = m.bootstrap_ci([])
        assert math.isnan(res.estimate)


class TestMcNemar:
    @pytest.mark.parametrize("b,c", [(10, 2), (0, 7), (5, 5), (1, 0), (30, 12)])
    def test_matches_binomtest(self, b, c):
        res = m.mcnemar_exact(b, c)
        ref = stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
        assert res.pvalue == pytest.approx(ref, rel=1e-9)
        assert (res.b, res.c, res.n_discordant) == (b, c, b + c)

    def test_known_value(self):
        # b=10, c=2: p = 2 * P[X <= 2 | n=12] = 2 * 79/4096
        assert m.mcnemar_exact(10, 2).pvalue == pytest.approx(2 * 79 / 4096)

    def test_no_discordant_pairs(self):
        assert m.mcnemar_exact(0, 0).pvalue == 1.0


class TestCochranQ:
    def test_two_conditions_equals_mcnemar_chi2(self):
        rng = np.random.default_rng(3)
        x = rng.integers(0, 2, size=(80, 2))
        res = m.cochrans_q(x)
        _, b, c, _ = m.paired_table(x[:, 0], x[:, 1])
        assert res.q == pytest.approx((b - c) ** 2 / (b + c))
        assert res.df == 1
        assert res.pvalue == pytest.approx(stats.chi2.sf(res.q, 1))

    def test_known_example(self):
        # Hand-computed: k=3, C=(4,3,1), R=(1,1,1,2,1,2,0), N=8
        x = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 0, 0]])
        res = m.cochrans_q(x)
        expected = 2 * (3 * (16 + 9 + 1) - 64) / (3 * 8 - (1 + 1 + 1 + 4 + 1 + 4 + 0))
        assert res.q == pytest.approx(expected)
        assert res.pvalue == pytest.approx(stats.chi2.sf(expected, 2))

    def test_constant_rows(self):
        res = m.cochrans_q([[1, 1, 1], [0, 0, 0]])
        assert res.q == 0.0 and res.pvalue == 1.0

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            m.cochrans_q([[1], [0]])
        with pytest.raises(ValueError):
            m.cochrans_q([[2, 0], [0, 1]])


class TestHolm:
    def test_known_values(self):
        adj = m.holm_bonferroni([0.01, 0.04, 0.03, 0.005])
        assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])

    def test_monotone_and_capped(self):
        adj = m.holm_bonferroni([0.5, 0.6, 0.9])
        assert adj == pytest.approx([1.0, 1.0, 1.0])

    def test_empty(self):
        assert m.holm_bonferroni([]) == []


class TestEffectSizes:
    def test_paired_risk_difference_estimate_and_bounds(self):
        treat = [0] * 40 + [1] * 5 + [0] * 15 + [1] * 10
        ctrl = [0] * 40 + [1] * 5 + [1] * 15 + [0] * 10
        res = m.paired_risk_difference(treat, ctrl)
        assert res.estimate == pytest.approx((15 - 20) / 70)
        assert res.low < res.estimate < res.high
        assert -1 <= res.low and res.high <= 1

    def test_paired_rd_close_to_wald_for_large_n(self):
        rng = np.random.default_rng(11)
        ctrl = rng.integers(0, 2, size=4000)
        treat = np.where(rng.random(4000) < 0.8, ctrl, 1 - ctrl)
        res = m.paired_risk_difference(treat.tolist(), ctrl.tolist())
        _, b, c, _ = m.paired_table(treat, ctrl)
        n = len(treat)
        diff = (b - c) / n
        se = math.sqrt(b + c - (b - c) ** 2 / n) / n
        assert res.low == pytest.approx(diff - 1.96 * se, abs=0.004)
        assert res.high == pytest.approx(diff + 1.96 * se, abs=0.004)

    def test_conditional_or_exact_ci_matches_clopper_pearson(self):
        res = m.conditional_odds_ratio(12, 4)
        ci = stats.binomtest(12, 16).proportion_ci(confidence_level=0.95, method="exact")
        assert res.estimate == pytest.approx(3.0)
        assert res.low == pytest.approx(ci.low / (1 - ci.low))
        assert res.high == pytest.approx(ci.high / (1 - ci.high))
        assert not res.haldane

    def test_conditional_or_haldane(self):
        res = m.conditional_odds_ratio(0, 6)
        assert res.haldane
        assert res.estimate == pytest.approx(0.5 / 6.5)
        assert res.low == 0.0
        res2 = m.conditional_odds_ratio(5, 0)
        assert math.isinf(res2.high)
        assert math.isnan(m.conditional_odds_ratio(0, 0).estimate)

    def test_rrr(self):
        assert m.relative_risk_reduction(0.1, 0.4) == pytest.approx(0.75)
        assert math.isnan(m.relative_risk_reduction(0.1, 0.0))


class TestKappa:
    def test_matches_sklearn_binary_and_multiclass(self):
        rng = np.random.default_rng(5)
        labels = ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
        a = rng.choice(labels, size=300, p=[0.6, 0.3, 0.1])
        b = np.where(rng.random(300) < 0.75, a, rng.choice(labels, size=300))
        res = m.cohen_kappa(a.tolist(), b.tolist(), labels=labels)
        assert res.kappa == pytest.approx(cohen_kappa_score(a, b, labels=labels))
        binary_a = [x == "SUPPORTED" for x in a]
        binary_b = [x == "SUPPORTED" for x in b]
        res2 = m.cohen_kappa([str(x) for x in binary_a], [str(x) for x in binary_b])
        assert res2.kappa == pytest.approx(cohen_kappa_score(binary_a, binary_b))

    def test_se_close_to_bootstrap(self):
        rng = np.random.default_rng(9)
        a = rng.choice(["A", "B", "C"], size=400)
        b = np.where(rng.random(400) < 0.6, a, rng.choice(["A", "B", "C"], size=400))
        res = m.cohen_kappa(a.tolist(), b.tolist())
        idx = rng.integers(0, 400, size=(1500, 400))
        boots = [cohen_kappa_score(a[i], b[i]) for i in idx]
        assert res.se == pytest.approx(float(np.std(boots)), rel=0.2)
        assert res.low < res.kappa < res.high

    def test_perfect_and_degenerate(self):
        assert m.cohen_kappa(["A", "B", "A"], ["A", "B", "A"]).kappa == pytest.approx(1.0)
        assert math.isnan(m.cohen_kappa(["A", "A"], ["A", "A"]).kappa)


class TestRouting:
    def test_against_sklearn(self):
        labels = ["ANSWER", "SOFT-ASK", "GENERALIZE", "ABSTAIN", "ESCALATE"]
        gold = ["ANSWER", "ANSWER", "SOFT-ASK", "ABSTAIN", "ESCALATE", "GENERALIZE", "ABSTAIN"]
        pred = ["ANSWER", "GENERALIZE", "SOFT-ASK", "ANSWER", "ESCALATE", "GENERALIZE", "ABSTAIN"]
        acceptable = [
            ["ANSWER"],
            ["ANSWER", "GENERALIZE"],
            ["SOFT-ASK"],
            ["ABSTAIN"],
            ["ESCALATE"],
            ["GENERALIZE"],
            ["ABSTAIN", "GENERALIZE"],
        ]
        res = m.routing_metrics(gold, pred, acceptable)
        p, r, f, s = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
        for i, lab in enumerate(labels):
            assert res["per_class"][lab]["precision"] == pytest.approx(p[i])
            assert res["per_class"][lab]["recall"] == pytest.approx(r[i])
            assert res["per_class"][lab]["f1"] == pytest.approx(f[i])
        assert res["accuracy_strict"]["rate"] == pytest.approx(5 / 7)
        assert res["accuracy_lenient"]["rate"] == pytest.approx(6 / 7)
        assert res["macro_f1"] == pytest.approx(float(np.mean(f)))
        assert res["confusion"][0] == [1, 0, 1, 0, 0]
        assert sum(map(sum, res["confusion"])) == 7

    def test_macro_f1_ignores_absent_classes(self):
        res = m.routing_metrics(["ANSWER", "ABSTAIN"], ["ANSWER", "ABSTAIN"])
        assert res["macro_f1"] == pytest.approx(1.0)
        assert math.isnan(res["per_class"]["ESCALATE"]["recall"])


class TestRiskCoverage:
    def test_curve(self):
        rc = m.risk_coverage_curve([0.9, 0.1, 0.5, 0.7], [0, 1, 1, 0])
        assert rc.coverage == pytest.approx([0.25, 0.5, 0.75, 1.0])
        assert rc.risk == pytest.approx([0.0, 0.0, 1 / 3, 0.5])
        assert rc.aurc == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)

    def test_percentile(self):
        assert m.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
        assert math.isnan(m.percentile([], 50))


def test_wilson_extremes_are_exact():
    for n in (1, 3, 7, 240):
        assert m.wilson_ci(0, n)[0] == 0.0
        assert m.wilson_ci(n, n)[1] == 1.0
