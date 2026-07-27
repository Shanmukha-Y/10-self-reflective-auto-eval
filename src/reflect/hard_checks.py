"""Deterministic validators that run before the judge is ever called.

Word counts, required/forbidden strings for any domain; for codegen, the
candidate's code is actually executed against test cases in the sandbox.
Failing a hard check means the judge never sees the output at all -- output
that fails a mechanical check isn't worth spending judge tokens scoring,
and for codegen the sandbox result also doubles as the calibration ground
truth judge scores get checked against later (see calibrate.py).
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from reflect.sandbox import SandboxResult, run_code
from reflect.tasks import Task

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


class CheckDetail(BaseModel):
    name: str
    passed: bool
    detail: str


class HardCheckResult(BaseModel):
    passed: bool
    checks: list[CheckDetail]
    sandbox: SandboxResult | None = None  # populated for codegen only


def _word_count(text: str) -> int:
    return len(text.split())


def extract_code(text: str) -> str:
    """Strip a markdown fence if the model wrapped code in ```python ... ```."""
    match = _CODE_FENCE_RE.search(text)
    return match.group(1) if match else text


def run_hard_checks(task: Task, output: str) -> HardCheckResult:
    checks: list[CheckDetail] = []
    sandbox_result: SandboxResult | None = None
    hard = task.hard

    if hard.min_words is not None:
        wc = _word_count(output)
        checks.append(CheckDetail(
            name="min_words", passed=wc >= hard.min_words,
            detail=f"{wc} words, need >= {hard.min_words}",
        ))
    if hard.max_words is not None:
        wc = _word_count(output)
        checks.append(CheckDetail(
            name="max_words", passed=wc <= hard.max_words,
            detail=f"{wc} words, need <= {hard.max_words}",
        ))
    for phrase in hard.must_include_all:
        ok = phrase.lower() in output.lower()
        checks.append(CheckDetail(
            name=f"must_include:{phrase}", passed=ok,
            detail="found" if ok else "required phrase is missing",
        ))
    if hard.must_include_any:
        ok = any(p.lower() in output.lower() for p in hard.must_include_any)
        checks.append(CheckDetail(
            name="must_include_any", passed=ok,
            detail="found at least one alternative" if ok else "none of the required alternatives were found",
        ))
    for phrase in hard.forbidden:
        ok = phrase.lower() not in output.lower()
        checks.append(CheckDetail(
            name=f"forbidden:{phrase}", passed=ok,
            detail="absent (good)" if ok else "forbidden phrase is present",
        ))

    if task.domain == "codegen" and task.code is not None:
        code = extract_code(output)
        sandbox_result = run_code(code, task.code.test_asserts)
        total = len(task.code.test_asserts)
        detail = (
            f"{sandbox_result.pass_count}/{total} tests passed"
            if sandbox_result.ran
            else f"execution failed: {sandbox_result.error}"
        )
        checks.append(CheckDetail(name="code_tests", passed=sandbox_result.all_passed, detail=detail))

    passed = all(c.passed for c in checks) if checks else True
    return HardCheckResult(passed=passed, checks=checks, sandbox=sandbox_result)
