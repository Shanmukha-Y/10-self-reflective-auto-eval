"""Pure-function tests for confusion matrix / agreement stats, plus a
scripted regression test for per-instance failure isolation. No LLM."""

from __future__ import annotations

import json

import reflect.calibrate as calibrate
from reflect.calibrate import CalibrationSample, agreement_stats, confusion_matrix
from reflect.hard_checks import HardCheckResult
from reflect.judge import CriterionScore, JudgeResult
from reflect.llm import LLMError
from reflect.sandbox import SandboxResult, TestCaseResult
from reflect.tasks import CodeSpec, Task


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


def _codegen_task(task_id: str) -> Task:
    return Task(
        id=task_id, domain="codegen", title="t", instructions="i", rubric_name="codegen",
        code=CodeSpec(function_name="f", starter_signature="def f():", test_asserts=["assert True"]),
    )


def test_run_codegen_calibration_isolates_per_instance_failures(monkeypatch, tmp_path):
    """Regression test for the live crash where one instance's LLMError
    (a real 240s judge timeout) killed the entire 8-instance batch instead
    of being recorded and skipped. Scripts the judge to raise LLMError on
    instance 2 of 3; the batch must complete with 2 samples + 1 recorded
    failure, and the checkpoint file must already reflect instance 1's
    success by the time instance 2 is attempted -- proving checkpointing
    happens per instance, not only once at the end."""
    tasks = [_codegen_task("c1"), _codegen_task("c2"), _codegen_task("c3")]
    checkpoint_path = tmp_path / "checkpoint.json"
    db_path = tmp_path / "cal.sqlite3"

    def fake_generate(task, constraints=None):
        return "code"

    def fake_hard_checks(task, output):
        return HardCheckResult(
            passed=True, checks=[],
            sandbox=SandboxResult(ran=True, results=[TestCaseResult(case=0, passed=True, error=None)]),
        )

    def fake_judge(task, output):
        if task.id == "c2":
            assert checkpoint_path.exists(), "checkpoint must exist before instance 2 runs"
            data = json.loads(checkpoint_path.read_text())
            assert data["completed"] == ["c1"], "checkpoint must already reflect instance 1's success"
            raise LLMError("simulated timeout on instance 2")
        return JudgeResult(scores=[
            CriterionScore(name="correctness", score=5, justification="j"),
            CriterionScore(name="readability", score=5, justification="j"),
            CriterionScore(name="robustness", score=5, justification="j"),
        ])

    monkeypatch.setattr(calibrate, "all_bench_tasks", lambda: tasks)
    monkeypatch.setattr(calibrate, "generate", fake_generate)
    monkeypatch.setattr(calibrate, "run_hard_checks", fake_hard_checks)
    monkeypatch.setattr(calibrate, "judge", fake_judge)

    cm, samples, failures = calibrate.run_codegen_calibration(n=3, db_path=db_path, checkpoint_path=checkpoint_path)

    assert {s.task_id for s in samples} == {"c1", "c3"}
    assert len(failures) == 1
    assert failures[0].task_id == "c2"
    assert "simulated timeout" in failures[0].error
    assert cm.total == 2  # only the 2 completed instances enter the confusion matrix

    final = json.loads(checkpoint_path.read_text())
    assert final["completed"] == ["c1", "c3"]
    assert final["failed"] == [{"task_id": "c2", "error": failures[0].error}]
    assert final["total_planned"] == 3

    # the 2 successful instances landed in the DB even though instance 2 failed
    rows = calibrate.fetch_calibration(db_path=db_path)
    assert {r["task_id"] for r in rows} == {"c1", "c3"}
