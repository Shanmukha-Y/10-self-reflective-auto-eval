# Self-Reflective Agent with Auto-Eval

Project 10: an LLM-as-judge auto-evaluation loop (generate → judge → critique → regenerate) that measures the judge's own reliability instead of assuming it.

## What it does

- Runs a generate → judge → critique → regenerate cycle (up to 3 attempts) where a generator, an LLM judge, and a critic all sit on the same local `qwen3.5:9b` model.
- Gates every candidate through deterministic hard checks first (word counts, required phrases; for code, actually running the tests in a sandboxed subprocess) — an LLM judge call is never spent on something a cheap check already caught.
- Retries with compiled, machine-usable constraints from a separate critic call — the generator never sees its own failed draft, only the specific things to fix.
- Judges blind (no attempt number, no history), against anchored 1/3/5 rubrics, at temperature 0, with a schema-enforced per-criterion justification.
- Ships `reflect calibrate`, which cross-tabulates judge verdicts against real test execution on codegen tasks to produce an actual confusion matrix, not an assumed one.
- Logs every attempt to a SQLite store and derives pass@1 vs. pass@3, score-by-attempt, and a criterion heatmap from it via `reflect report`.

## Quick start

```bash
uv sync

# one task, watch every attempt/critique/constraint
uv run reflect run --task constrained_write --show-cycles

# list all 30 bench task ids
uv run reflect bench --list

# charts + pass@k table from the metrics store
uv run reflect report

# judge-vs-tests calibration on a codegen subset (default n=8)
uv run reflect calibrate --n 8

# tests: fast, zero-network unit/mocked suite (default)
uv run pytest

# live tests against the real model (slow; requires a local Ollama instance running qwen3.5:9b)
uv run pytest -m integration
```

## Measured results (small sample, stated honestly)

5 live runs total against `qwen3.5:9b` (1 summarize, 2 constrained_write, 2 codegen) — enough to prove the loop and metrics pipeline end to end on real model output, not enough for a statistically meaningful pass@k estimate. The 30-instance bench is built and tested but was not run in full for these numbers.

| domain            | pass@1 | pass@3 | n (runs) |
|-------------------|--------|--------|----------|
| codegen           | 0%     | 100%   | 2        |
| constrained_write | 0%     | 100%   | 2        |
| summarize         | 100%   | 100%   | 1        |

Every codegen and constrained_write run in this sample failed attempt 1 and passed by attempt 2 or 3; with n=1-2 per domain, "0% pass@1" describes this sample, not a general claim about the model.

## Learnings

- **The judge's own false-positive rate came back undefined, not zero — and that distinction was the point.** The 8-instance codegen calibration subset (7 completed, 1 recorded timeout) produced 7/7 true positives, zero false positives, zero false negatives. That's 100% accuracy, but the false-positive rate is mathematically undefined (n/a), because the sample contained no instance where the tests actually failed — there was no bad code for the judge to potentially wave through. Reporting that as "0% false-positive rate" would have been a stronger claim than the data supports; the README states the undefined case explicitly instead of rounding up.
- **The score-by-attempt chart is flat at 5.0 across all attempts — and that flatness is itself the finding.** Across 5 live runs, every failure was caught by a deterministic hard check (a missing required phrase, a failing test) before the judge ever ran, so the judge only ever scored attempts that had already cleared hard checks, and scored every one of them 5/5. The judge's own before/after trajectory isn't what this sample demonstrates — the hard-check pass/fail transition is. Whether the judge shows a rising-score trend on failures it actually has to grade is an open question a larger run would need to answer.
- **A single failed instance took down an entire calibration batch.** `run_codegen_calibration()` originally built its sample list with an unguarded list comprehension; when instance 3's judge call hit the timeout, the whole run crashed and every already-completed result was lost. Fixed by wrapping each instance's generate+judge in try/except, recording failures as data (`CalibrationFailure`) instead of letting them propagate, and writing a checkpoint to disk after every instance so a mid-run crash no longer loses completed work.
- **The call timeout was moved from 240s to 300s based on two independent observed timeouts, not a guess.** One generator call timed out under multi-builder contention on the shared Ollama instance; one judge call timed out with exclusive server access on a larger, full-rubric prompt. Both were real, both at 240s, so one shared timeout constant was raised to cover both rather than inventing a judge-specific override from a sample of two.
- **`httpx.ReadTimeout` was propagating unwrapped past the code that was supposed to catch timeouts.** A live 3-domain run failed on `summarize` with an uncaught `httpx.ReadTimeout`; the Ollama client lets that exception through under load, and it is not a subclass of Python's builtin `TimeoutError`, which was the only timeout type the original code caught. Fixed by catching `httpx.TimeoutException` explicitly in both `chat()` and `generate_json()`, with a mocked regression test that reproduces the exact exception.
- **A deterministic retry that resent an identical temperature-0 prompt could never fix anything.** The judge's semantic-mismatch retry (criterion names not matching the rubric) originally resent the same prompt unchanged — deterministic generation means it would reproduce the same wrong output. Fixed by appending an explicit correction listing the expected names to the retry prompt.
- **No human-agreement numbers are claimed.** `calibrate.agreement_stats()` implements percent-agreement / Cohen's kappa and is unit-tested, but no independent human rater was available during this build. Self-labeling and calling it "human agreement" was treated as worse than not reporting the number at all.

See `readme.html` for the full write-up, including the architecture diagram, live-run demo transcripts, and measured-results charts.
