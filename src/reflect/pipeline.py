"""Ties generator, hard checks, judge, critic, and constraint compilation
into the generate -> judge -> critique -> regenerate loop, logging every
attempt to the metrics store.

Not in the spec's original file listing -- factored out of what would
otherwise have been duplicated between the CLI's `run` command and the bench
runner, and it's the one function `test_pipeline_mocked.py` exercises with a
scripted judge/critic to prove the constraint-carrying behavior end to end.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from reflect import constraints as constraints_mod
from reflect import metrics
from reflect.config import CONFIG
from reflect.critic import Constraint, critique
from reflect.generator import generate
from reflect.hard_checks import HardCheckResult, run_hard_checks
from reflect.judge import JudgeResult, judge
from reflect.tasks import Task


@dataclass
class AttemptOutcome:
    attempt_number: int
    output: str
    hard: HardCheckResult
    judge_result: JudgeResult | None
    passed: bool
    constraints_used: list[Constraint]
    new_constraints: list[Constraint]
    latency_s: float


@dataclass
class RunResult:
    run_id: str
    task: Task
    attempts: list[AttemptOutcome] = field(default_factory=list)
    accepted_index: int = 0

    @property
    def final_output(self) -> str:
        return self.attempts[self.accepted_index].output

    @property
    def final_status(self) -> str:
        return "passed" if self.attempts[self.accepted_index].passed else "exhausted"


def _pick_best(attempts: list[AttemptOutcome]) -> int:
    """First passing attempt if any; otherwise the highest judge mean among
    attempts that at least cleared hard checks (a judged-but-failed attempt
    beats one that never got judged); ties broken by earliest attempt."""
    for i, a in enumerate(attempts):
        if a.passed:
            return i

    def score(a: AttemptOutcome) -> float:
        if a.judge_result is None:
            return float("-inf")
        return a.judge_result.mean_score()

    return max(range(len(attempts)), key=lambda i: score(attempts[i]))


def run_cycle(
    task: Task,
    max_attempts: int = CONFIG.max_attempts,
    show_cycles: bool = False,
    db_path: Path | None = None,
) -> RunResult:
    run_id = uuid.uuid4().hex[:12]
    accumulated: list[Constraint] = []
    attempts: list[AttemptOutcome] = []

    for k in range(1, max_attempts + 1):
        t0 = time.monotonic()
        constraint_strs = [c.constraint for c in accumulated] or None
        output = generate(task, constraints=constraint_strs)
        hard = run_hard_checks(task, output)

        judge_result: JudgeResult | None = None
        passed = False
        new_constraints: list[Constraint] = []

        if not hard.passed:
            new_constraints = constraints_mod.constraints_from_hard_checks(hard)
        else:
            judge_result = judge(task, output)
            passed = judge_result.passed(task.rubric())
            if not passed:
                new_constraints = critique(task, output, judge_result).constraints

        latency = time.monotonic() - t0
        outcome = AttemptOutcome(
            attempt_number=k,
            output=output,
            hard=hard,
            judge_result=judge_result,
            passed=passed,
            constraints_used=list(accumulated),
            new_constraints=new_constraints,
            latency_s=latency,
        )
        attempts.append(outcome)

        if show_cycles:
            _print_cycle(task, outcome)

        metrics.log_attempt(
            metrics.AttemptRecord(
                run_id=run_id,
                task_id=task.id,
                domain=task.domain,
                attempt_number=k,
                output=output,
                hard_passed=hard.passed,
                hard_detail="; ".join(f"{c.name}={'ok' if c.passed else 'FAIL'} ({c.detail})" for c in hard.checks) or "no hard constraints",
                judge_ran=judge_result is not None,
                judge_scores={s.name: s.score for s in judge_result.scores} if judge_result else None,
                judge_mean=judge_result.mean_score() if judge_result else None,
                judge_min=judge_result.min_score() if judge_result else None,
                judge_passed=passed if judge_result else None,
                code_tests_passed=hard.sandbox.pass_count if hard.sandbox else None,
                code_tests_total=len(hard.sandbox.results) if hard.sandbox else None,
                latency_s=latency,
            ),
            db_path=db_path,
        )

        if passed:
            break

        accumulated = constraints_mod.compile_constraints(accumulated, new_constraints)

    result = RunResult(run_id=run_id, task=task, attempts=attempts)
    result.accepted_index = _pick_best(attempts)
    metrics.mark_accepted(
        run_id=run_id,
        attempt_number=attempts[result.accepted_index].attempt_number,
        final_status=result.final_status,
        db_path=db_path,
    )
    return result


def _print_cycle(task: Task, outcome: AttemptOutcome) -> None:
    print(f"\n--- attempt {outcome.attempt_number} ({task.id}) ---")
    if outcome.constraints_used:
        print("constraints carried in:")
        for c in outcome.constraints_used:
            print(f"  - {c.constraint}")
    print(f"\noutput:\n{outcome.output}\n")
    print("hard checks: " + ("PASS" if outcome.hard.passed else "FAIL"))
    for c in outcome.hard.checks:
        print(f"  [{'ok' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    if outcome.judge_result:
        print(f"judge: mean={outcome.judge_result.mean_score():.2f} min={outcome.judge_result.min_score()} "
              f"-> {'PASS' if outcome.passed else 'FAIL'}")
        for s in outcome.judge_result.scores:
            print(f"  {s.name}: {s.score}/5 -- {s.justification}")
    if outcome.new_constraints:
        print("new constraints for next attempt:")
        for c in outcome.new_constraints:
            print(f"  - {c.constraint}  (reason: {c.reason})")
