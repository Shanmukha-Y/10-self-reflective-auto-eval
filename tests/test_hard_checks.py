"""Deterministic validators. No LLM."""

from __future__ import annotations

from reflect.hard_checks import extract_code, run_hard_checks
from reflect.tasks import CodeSpec, HardConstraints, Task


def _task(domain="summarize", **hard_kwargs) -> Task:
    return Task(
        id="t1",
        domain=domain,
        title="test",
        instructions="do the thing",
        rubric_name="summarize",
        hard=HardConstraints(**hard_kwargs),
    )


def test_no_constraints_always_passes():
    task = _task()
    result = run_hard_checks(task, "any output at all")
    assert result.passed
    assert result.checks == []


def test_min_words():
    task = _task(min_words=5)
    assert run_hard_checks(task, "one two three").passed is False
    assert run_hard_checks(task, "one two three four five").passed is True


def test_max_words():
    task = _task(max_words=3)
    assert run_hard_checks(task, "one two three four").passed is False
    assert run_hard_checks(task, "one two three").passed is True


def test_must_include_all():
    task = _task(must_include_all=["refund", "30 days"])
    assert run_hard_checks(task, "you get a refund").passed is False
    result = run_hard_checks(task, "you get a refund within 30 days")
    assert result.passed is True


def test_must_include_all_is_case_insensitive():
    task = _task(must_include_all=["Refund"])
    assert run_hard_checks(task, "your refund is processed").passed is True


def test_must_include_any():
    task = _task(must_include_any=["cat", "dog"])
    assert run_hard_checks(task, "I have a fish").passed is False
    assert run_hard_checks(task, "I have a dog").passed is True


def test_forbidden():
    task = _task(forbidden=["guarantee"])
    assert run_hard_checks(task, "we guarantee results").passed is False
    assert run_hard_checks(task, "we expect good results").passed is True


def test_multiple_checks_all_must_pass():
    task = _task(min_words=3, must_include_all=["refund"], forbidden=["guarantee"])
    result = run_hard_checks(task, "we guarantee a refund")  # has forbidden word
    assert result.passed is False
    failing = [c.name for c in result.checks if not c.passed]
    assert "forbidden:guarantee" in failing


def test_extract_code_strips_markdown_fence():
    text = "Here you go:\n```python\ndef f():\n    return 1\n```\n"
    assert extract_code(text).strip() == "def f():\n    return 1"


def test_extract_code_passthrough_when_no_fence():
    text = "def f():\n    return 1\n"
    assert extract_code(text) == text


def test_codegen_hard_check_runs_sandbox():
    task = _task(
        domain="codegen",
        **{},
    )
    task = task.model_copy(update={
        "code": CodeSpec(
            function_name="add",
            starter_signature="def add(a: int, b: int) -> int:",
            test_asserts=["assert add(1, 2) == 3", "assert add(2, 2) == 5"],
        )
    })
    output = "```python\ndef add(a, b):\n    return a + b\n```"
    result = run_hard_checks(task, output)
    assert result.passed is False  # second assert is wrong on purpose
    assert result.sandbox is not None
    assert result.sandbox.pass_count == 1

    good_output = "```python\ndef add(a, b):\n    return a + b\n```"
    task2 = task.model_copy(update={
        "code": CodeSpec(
            function_name="add",
            starter_signature="def add(a: int, b: int) -> int:",
            test_asserts=["assert add(1, 2) == 3", "assert add(2, 2) == 4"],
        )
    })
    result2 = run_hard_checks(task2, good_output)
    assert result2.passed is True
