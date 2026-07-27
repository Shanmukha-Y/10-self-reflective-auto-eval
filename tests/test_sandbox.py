"""Sandbox isolation, timeout, and per-case error handling. No LLM."""

from __future__ import annotations

from reflect.sandbox import run_code


def test_all_pass():
    code = "def add(a, b):\n    return a + b\n"
    result = run_code(code, ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"])
    assert result.ran
    assert result.all_passed
    assert result.pass_count == 2


def test_partial_failure_does_not_stop_other_cases():
    code = "def add(a, b):\n    return a + b\n"
    result = run_code(code, ["assert add(1, 2) == 999", "assert add(2, 2) == 4"])
    assert result.ran
    assert not result.all_passed
    assert result.pass_count == 1
    assert result.results[0].passed is False
    assert "AssertionError" in result.results[0].error
    assert result.results[1].passed is True


def test_exception_in_one_case_is_isolated():
    code = "def divide(a, b):\n    return a / b\n"
    result = run_code(code, ["assert divide(10, 0) == 1", "assert divide(10, 2) == 5"])
    assert result.ran
    assert result.results[0].passed is False
    assert "ZeroDivisionError" in result.results[0].error
    assert result.results[1].passed is True


def test_syntax_error_reported_as_crash():
    code = "def broken(:\n    pass\n"
    result = run_code(code, ["assert True"])
    assert result.ran is False
    assert result.error is not None
    assert not result.all_passed


def test_timeout_is_enforced():
    code = "def spin():\n    while True:\n        pass\n"
    result = run_code(code, ["spin()"], timeout_s=1.0)
    assert result.ran is False
    assert "timed out" in result.error


def test_network_access_is_blocked():
    code = (
        "import socket\n"
        "def try_connect():\n"
        "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "    return s\n"
    )
    result = run_code(code, ["try_connect()"])
    assert result.ran
    assert result.results[0].passed is False
    assert "network access is disabled" in result.results[0].error


def test_code_with_braces_does_not_break_harness():
    """Regression: the harness must not use str.format on candidate code,
    since dict literals / f-strings / comprehensions contain braces that
    would corrupt a naive .format() template."""
    code = (
        "def make(d):\n"
        "    merged = {**d, 'extra': 1}\n"
        "    return f'{merged}'\n"
    )
    result = run_code(code, ["assert make({'a': 1}) == \"{'a': 1, 'extra': 1}\""])
    assert result.ran
    assert result.all_passed
