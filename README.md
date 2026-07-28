# Self-Reflective Agent with Auto-Eval

Project 10: an LLM-as-judge evaluation loop—generate → judge → critique → regenerate—that measures the judge's own reliability instead of assuming it.

## What it does

- Runs a generate → judge → critique → regenerate cycle for up to three attempts, with the generator, judge, and critic all using the same local `qwen3.5:9b` model.
- Gates every candidate through deterministic hard checks first: word counts, required phrases, and—for code—actual test execution in an isolated, timeout-bounded subprocess harness. The harness is **not** an operating-system security sandbox.
- Retries with machine-usable constraints compiled by a separate critic call. The generator sees the concrete conditions it must satisfy rather than its entire failed draft.
- Judges blindly—no attempt number or prior history—against anchored 1/3/5 rubrics at temperature 0, with schema-enforced criterion-level justifications.
- Ships `reflect calibrate`, which cross-tabulates judge verdicts against real code-test outcomes to produce a confusion matrix rather than treating the judge as ground truth.
- Logs every attempt to SQLite and derives pass@1 versus pass@3, score-by-attempt, and criterion heatmaps through `reflect report`.

## Quick start

```bash
uv sync

# Run one task and inspect every cycle
uv run reflect run --task constrained_write --show-cycles

# List all 30 benchmark task IDs
uv run reflect bench --list

# Generate charts and the pass@k table from the metrics store
uv run reflect report

# Calibrate judge verdicts against code-test outcomes (default n=8)
uv run reflect calibrate --n 8

# Fast unit and mocked-LLM suite
uv run pytest

# Slow live-model tests; requires Ollama with qwen3.5:9b
uv run pytest -m integration
```

## Code-execution safety note

The code-test runner uses a temporary working directory, a minimal environment, Python isolated mode (`-I`), a timeout, and Python-level socket blocking. These controls reduce accidental interference and make benchmark runs more repeatable; they do **not** safely contain hostile code. Generated code can still interact with host resources through paths that this Python-level harness does not mediate. Only run trusted benchmark fixtures. Evaluating untrusted third-party code requires a real container or VM boundary with filesystem, process, syscall, resource, and network controls.

## Measured results

Five live runs against `qwen3.5:9b`—one summarization, two constrained-writing, and two code-generation tasks—were enough to verify the loop and metrics pipeline end to end, but not enough for a statistically meaningful pass@k estimate. The 30-instance benchmark is implemented and tested but was not run in full for these numbers.

| Domain | pass@1 | pass@3 | n |
|---|---:|---:|---:|
| Code generation | 0% | 100% | 2 |
| Constrained writing | 0% | 100% | 2 |
| Summarization | 100% | 100% | 1 |

Every code-generation and constrained-writing run in this sample failed attempt 1 and passed by attempt 2 or 3. With only one or two observations per domain, these values describe the sample, not the model in general.

## Learnings

- **The judge's false-positive rate was undefined, not zero.** The eight-instance code-generation calibration subset completed seven cases and recorded one timeout. All seven completed cases were true positives, with no false positives or false negatives. Accuracy on those completed cases was 100%, but false-positive rate was undefined because the sample contained no ground-truth negatives.
- **The score-by-attempt chart was flat at 5.0 for a reason.** Every observed failure was caught by deterministic checks before the judge ran, so the judge only saw candidates that had already cleared the hard gate. This sample demonstrates the hard-check transition, not improving judge scores over attempts.
- **One failed item originally killed the entire calibration batch.** Per-item error handling and checkpointing now record failures as data and preserve completed work.
- **Timeout handling needed to include `httpx.TimeoutException`.** A real `httpx.ReadTimeout` escaped a handler that only caught Python's built-in `TimeoutError`; the transport now wraps both paths and has a regression test.
- **An identical deterministic retry cannot repair a semantic schema mismatch.** The judge's retry prompt now names the expected criterion labels instead of resending the same temperature-0 request.
- **No human-agreement statistic is claimed.** Agreement and Cohen's kappa utilities exist and are tested, but there was no independent human rater for this build.

See [readme.html](readme.html) for the full write-up, including the architecture diagram, live transcripts, and measured-result charts.
