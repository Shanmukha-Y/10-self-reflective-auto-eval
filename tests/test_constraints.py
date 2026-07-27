"""Constraint compilation + dedup. No LLM."""

from __future__ import annotations

from reflect.constraints import compile_constraints, constraints_from_hard_checks, dedupe
from reflect.critic import Constraint
from reflect.hard_checks import CheckDetail, run_hard_checks
from reflect.sandbox import SandboxResult, TestCaseResult
from reflect.tasks import HardConstraints, Task


def _task(**hard_kwargs) -> Task:
    return Task(id="t1", domain="summarize", title="t", instructions="i", rubric_name="summarize", hard=HardConstraints(**hard_kwargs))


def test_hard_check_failures_compile_to_constraints():
    task = _task(max_words=3, must_include_all=["refund"])
    hard = run_hard_checks(task, "way too many words here for sure")  # over limit, missing phrase
    constraints = constraints_from_hard_checks(hard)
    texts = [c.constraint for c in constraints]
    assert any("Stay within the required length" in t for t in texts)
    assert any("refund" in t for t in texts)


def test_passing_checks_produce_no_constraints():
    task = _task(max_words=10)
    hard = run_hard_checks(task, "short and fine")
    assert constraints_from_hard_checks(hard) == []


def test_code_test_failures_compile_one_constraint_per_failing_case():
    hard = run_hard_checks(_task(), "n/a")  # baseline, no checks
    hard = hard.model_copy(update={
        "passed": False,
        "checks": [CheckDetail(name="code_tests", passed=False, detail="1/2 tests passed")],
        "sandbox": SandboxResult(ran=True, results=[
            TestCaseResult(case=0, passed=True, error=None),
            TestCaseResult(case=1, passed=False, error="AssertionError: assertion failed"),
        ]),
    })
    constraints = constraints_from_hard_checks(hard)
    assert len(constraints) == 1
    assert "test case 1" in constraints[0].constraint


def test_dedupe_removes_case_and_whitespace_variants():
    constraints = [
        Constraint(constraint="Mention the refund window.", reason="r1"),
        Constraint(constraint="mention the refund window.", reason="r2"),
        Constraint(constraint="  Mention   the refund window.  ", reason="r3"),
        Constraint(constraint="Mention the shipping cost.", reason="r4"),
    ]
    out = dedupe(constraints)
    assert len(out) == 2
    assert out[0].reason == "r1"  # first occurrence wins


def test_compile_constraints_merges_groups_preserving_order():
    group1 = [Constraint(constraint="A", reason="a")]
    group2 = [Constraint(constraint="B", reason="b"), Constraint(constraint="A", reason="a-dup")]
    merged = compile_constraints(group1, group2)
    assert [c.constraint for c in merged] == ["A", "B"]
