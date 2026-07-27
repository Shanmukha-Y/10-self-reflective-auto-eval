"""Live tests against the real Ollama server (qwen3.5:9b). Excluded by
default (`addopts = "-m 'not integration'"`); run explicitly with
`uv run pytest -m integration`.

Ollama here is shared with other builders under load, so per-call timeouts
in llm.py are generous (240s) and each test's own pytest-timeout is set
well above that to allow for a full 3-attempt cycle (up to ~3 generate +
2 judge + 2 critique calls). This is a smoke test that the live wiring
works end to end for one instance per domain -- not the bench, and not the
calibration subset (see bench/run_bench.py and calibrate.py for those).
"""

from __future__ import annotations

import pytest

from reflect.pipeline import run_cycle
from reflect.tasks import get_task

pytestmark = pytest.mark.integration


@pytest.mark.timeout(600)
def test_summarize_live_cycle():
    task = get_task("summarize_paper")
    result = run_cycle(task, max_attempts=3)
    assert result.final_status in {"passed", "exhausted"}
    assert len(result.attempts) >= 1
    assert result.final_output.strip() != ""
    first = result.attempts[0]
    # every attempt either failed hard checks (and so has new_constraints)
    # or passed them and got judged.
    assert first.hard.passed or first.new_constraints
    if first.hard.passed:
        assert first.judge_result is not None


@pytest.mark.timeout(600)
def test_constrained_write_live_cycle():
    task = get_task("constrained_write")
    result = run_cycle(task, max_attempts=3)
    assert result.final_status in {"passed", "exhausted"}
    assert len(result.attempts) >= 1
    assert result.final_output.strip() != ""


@pytest.mark.timeout(600)
def test_codegen_live_cycle():
    task = get_task("codegen")
    result = run_cycle(task, max_attempts=3)
    assert result.final_status in {"passed", "exhausted"}
    assert len(result.attempts) >= 1
    assert result.final_output.strip() != ""
    # codegen's ground truth is real: the sandbox actually ran, whether or
    # not hard checks passed.
    assert result.attempts[0].hard.sandbox is not None
