"""Scripted judge/critic pipeline test: no LLM involved anywhere.

This is the test that proves the design thesis end to end: attempt 1 fails
on a judged criterion, the critic's constraint is compiled and actually
shows up in the *next* generate() call's arguments (not the raw failed
output), and attempt 2 passing ends the loop. A second test proves the
exhaustion path returns the best attempt, flagged, rather than silently
accepting a below-threshold output.
"""

from __future__ import annotations

import reflect.pipeline as pipeline
from reflect.critic import Constraint, Critique
from reflect.judge import CriterionScore, JudgeResult
from reflect.tasks import Task

RUBRIC_NAME = "constrained_write"  # pass_threshold=4.0, criterion_floor=3


def _task() -> Task:
    return Task(
        id="cw1",
        domain="constrained_write",
        title="test",
        instructions="Write a short apology that mentions the refund window.",
        rubric_name=RUBRIC_NAME,
    )


def _scores(adherence: int, coherence: int = 5, tone: int = 5) -> JudgeResult:
    return JudgeResult(scores=[
        CriterionScore(name="constraint_adherence", score=adherence, justification="j"),
        CriterionScore(name="coherence", score=coherence, justification="j"),
        CriterionScore(name="tone_fit", score=tone, justification="j"),
    ])


def test_fail_then_constrained_retry_passes(monkeypatch, tmp_path):
    task = _task()
    db_path = tmp_path / "metrics.sqlite3"

    generate_calls: list[list[str] | None] = []

    def fake_generate(t: Task, constraints=None):
        generate_calls.append(constraints)
        if constraints is None:
            return "Sorry about the issue."  # attempt 1: doesn't mention refund window
        return "Sorry about the issue. Your refund window is 30 days."  # attempt 2: fixed

    judge_calls: list[str] = []

    def fake_judge(t: Task, output: str) -> JudgeResult:
        judge_calls.append(output)
        if "refund window" in output:
            return _scores(adherence=5)
        return _scores(adherence=2)  # below floor of 3 -> fails

    critique_calls: list[JudgeResult] = []

    def fake_critique(t: Task, output: str, judge_result: JudgeResult) -> Critique:
        critique_calls.append(judge_result)
        return Critique(constraints=[
            Constraint(constraint="Mention the refund window explicitly.", reason="constraint_adherence scored 2/5, missing required fact")
        ])

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    monkeypatch.setattr(pipeline, "judge", fake_judge)
    monkeypatch.setattr(pipeline, "critique", fake_critique)

    result = pipeline.run_cycle(task, max_attempts=3, db_path=db_path)

    assert len(result.attempts) == 2
    assert result.final_status == "passed"
    assert result.final_output == "Sorry about the issue. Your refund window is 30 days."

    # attempt 1 got no constraints; attempt 2 got exactly the critic's constraint text
    assert generate_calls[0] is None
    assert generate_calls[1] == ["Mention the refund window explicitly."]

    assert len(judge_calls) == 2
    assert len(critique_calls) == 1  # critique only runs on the failing attempt

    # metrics store recorded both attempts, with attempt 2 marked accepted
    import reflect.metrics as metrics
    rows = metrics.fetch_all(db_path)
    assert len(rows) == 2
    assert rows[0]["accepted"] == 0
    assert rows[1]["accepted"] == 1
    assert rows[1]["final_status"] == "passed"
    assert rows[0]["final_status"] == "passed"  # both rows of the run share the final status


def test_exhaustion_returns_best_attempt_flagged(monkeypatch, tmp_path):
    task = _task()
    db_path = tmp_path / "metrics.sqlite3"

    def fake_generate(t: Task, constraints=None):
        return f"draft (constraints={constraints})"

    # every attempt fails (mean stays well below the 4.0 pass_threshold), but
    # adherence strictly improves attempt over attempt so there's a single
    # unambiguous best attempt to accept.
    scores_sequence = [1, 2, 3]

    def fake_judge(t: Task, output: str) -> JudgeResult:
        return _scores(adherence=scores_sequence.pop(0), coherence=2, tone=2)

    def fake_critique(t: Task, output: str, judge_result: JudgeResult) -> Critique:
        return Critique(constraints=[Constraint(constraint="improve attempt", reason="still failing")])

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    monkeypatch.setattr(pipeline, "judge", fake_judge)
    monkeypatch.setattr(pipeline, "critique", fake_critique)

    result = pipeline.run_cycle(task, max_attempts=3, db_path=db_path)

    assert len(result.attempts) == 3
    assert result.final_status == "exhausted"
    # best attempt is the last one: adherence 1,2,3 with coherence/tone fixed
    # at 2 means the judge mean strictly rises (1.67 -> 2.0 -> 2.33), all well
    # under the 4.0 pass_threshold, so this is purely a "best of a bad batch" pick.
    assert result.accepted_index == 2
    assert not result.attempts[2].passed  # still flagged as below threshold, not silently accepted
