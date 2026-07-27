"""Sandboxed subprocess execution for the code-task ground truth.

Runs generated code plus a set of assert-style test cases in a fresh
subprocess: timeout-bounded, network calls disabled inside the child, a
minimal environment. This is the actual ground truth for codegen tasks --
the judge never gets an opinion on whether tests pass, they just do or
don't, and this result also doubles as the calibration ground truth in
calibrate.py.

The harness script is assembled by string concatenation, not `str.format`
or an f-string wrapping the candidate code: generated Python routinely
contains `{`/`}` (dict literals, f-strings, comprehensions), which would
corrupt a `.format()` template. Only trusted, self-authored fragments
(the try/except wrapper) use f-strings.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel

from reflect.config import CONFIG

_MARKER = "===REFLECT_SANDBOX_RESULTS==="

_PRELUDE = """\
import socket as _socket
import json as _json


def _network_blocked(*args, **kwargs):
    raise RuntimeError("network access is disabled in the sandbox")


_socket.socket = _network_blocked
_socket.create_connection = _network_blocked

_results = []
"""


class TestCaseResult(BaseModel):
    __test__ = False  # not a pytest test class despite the name

    case: int
    passed: bool
    error: str | None = None


class SandboxResult(BaseModel):
    ran: bool  # False if the process crashed before producing results (syntax error, timeout, crash)
    error: str | None = None
    results: list[TestCaseResult] = []

    @property
    def all_passed(self) -> bool:
        return self.ran and self.error is None and len(self.results) > 0 and all(r.passed for r in self.results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)


def _build_harness(code: str, test_asserts: list[str]) -> str:
    lines = [_PRELUDE, code, ""]
    for idx, assertion in enumerate(test_asserts):
        lines.append("try:")
        lines.append(f"    {assertion}")
        lines.append(f"    _results.append({{'case': {idx}, 'passed': True, 'error': None}})")
        lines.append("except AssertionError:")
        lines.append(f"    _results.append({{'case': {idx}, 'passed': False, 'error': 'AssertionError: assertion failed'}})")
        lines.append("except Exception as _e:")
        lines.append(f"    _results.append({{'case': {idx}, 'passed': False, 'error': f'{{type(_e).__name__}}: {{_e}}'}})")
    lines.append(f"print({_MARKER!r})")
    lines.append("print(_json.dumps(_results))")
    return "\n".join(lines)


def run_code(code: str, test_asserts: list[str], timeout_s: float = CONFIG.sandbox_timeout_s) -> SandboxResult:
    """Execute `code` followed by each assert in `test_asserts`, each
    independently caught so one failing/raising case doesn't stop the rest
    from running."""
    script = _build_harness(code, test_asserts)

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "harness.py"
        script_path.write_text(script)
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tmpdir,
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(ran=False, error=f"execution timed out after {timeout_s}s")

    if _MARKER not in proc.stdout:
        stderr_tail = proc.stderr.strip()[-800:]
        return SandboxResult(ran=False, error=f"code crashed before producing results: {stderr_tail or 'no output'}")

    results_json = proc.stdout.split(_MARKER, 1)[1].strip()
    try:
        raw_results = json.loads(results_json)
    except json.JSONDecodeError as exc:
        return SandboxResult(ran=False, error=f"could not parse sandbox output: {exc}")

    return SandboxResult(ran=True, results=[TestCaseResult.model_validate(r) for r in raw_results])
