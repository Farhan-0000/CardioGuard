"""A judge's per-condition rate is reported only when it covered (almost) every valid turn."""
from types import SimpleNamespace

from evaluation.analyze import MIN_JUDGE_COVERAGE, judge_coverage


def _turn(judged: bool):
    return SimpleNamespace(claims={"nli": [] if judged else None})


def test_partial_judge_subset_is_not_reported():
    rows = [_turn(True)] * 76 + [_turn(False)] * 163
    cov = judge_coverage(rows, "nli")
    assert cov["judged"] == 76 and cov["valid"] == 239
    assert cov["reported"] is False


def test_complete_judge_is_reported():
    rows = [_turn(True)] * 237 + [_turn(False)] * 2
    cov = judge_coverage(rows, "nli")
    assert cov["fraction"] >= MIN_JUDGE_COVERAGE
    assert cov["reported"] is True


def test_no_turns_is_not_reported():
    assert judge_coverage([], "nli")["reported"] is False
