"""Pure-function tests for confusion matrix / agreement stats. No LLM."""

from __future__ import annotations

from reflect.calibrate import CalibrationSample, agreement_stats, confusion_matrix
from reflect.hard_checks import HardCheckResult
from reflect.judge import CriterionScore, JudgeResult


def _sample(task_id: str, tests_passed: bool, judge_says_pass: bool) -> CalibrationSample:
    return CalibrationSample(
        task_id=task_id,
        output="out",
        tests_passed=tests_passed,
        judge_says_pass=judge_says_pass,
        judge_mean=5.0 if judge_says_pass else 2.0,
        hard=HardCheckResult(passed=tests_passed, checks=[]),
        judge_result=JudgeResult(scores=[CriterionScore(name="correctness", score=5 if judge_says_pass else 2, justification="j")]),
    )


def test_confusion_matrix_counts_all_four_cells():
    samples = [
        _sample("a", tests_passed=True, judge_says_pass=True),   # tp
        _sample("b", tests_passed=False, judge_says_pass=True),  # fp -- the dangerous case
        _sample("c", tests_passed=True, judge_says_pass=False),  # fn
        _sample("d", tests_passed=False, judge_says_pass=False),  # tn
    ]
    cm = confusion_matrix(samples)
    assert (cm.tp, cm.fp, cm.fn, cm.tn) == (1, 1, 1, 1)
    assert cm.total == 4
    assert cm.accuracy == 0.5
    assert cm.false_positive_rate == 0.5  # fp / (fp + tn) = 1/2


def test_false_positive_rate_is_undefined_with_no_failing_tests():
    samples = [_sample("a", tests_passed=True, judge_says_pass=True)]
    cm = confusion_matrix(samples)
    assert cm.false_positive_rate != cm.false_positive_rate  # NaN


def test_agreement_stats_perfect_agreement():
    pairs = [(True, True), (False, False), (True, True)]
    stats = agreement_stats(pairs)
    assert stats["percent_agreement"] == 1.0
    assert stats["cohens_kappa"] == 1.0


def test_agreement_stats_systematic_disagreement_gives_negative_kappa():
    # both raters split 50/50 but always land on opposite sides -> worse
    # than chance agreement, kappa should go negative (not just low).
    pairs = [(True, False), (False, True), (True, False), (False, True)]
    stats = agreement_stats(pairs)
    assert stats["percent_agreement"] == 0.0
    assert stats["cohens_kappa"] == -1.0


def test_agreement_stats_empty():
    stats = agreement_stats([])
    assert stats["n"] == 0
