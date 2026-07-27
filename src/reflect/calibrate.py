"""Judge calibration: how much a same-model judge can actually be trusted.

(a) Judge-vs-tests on codegen: the sandbox already gives ground truth (tests
    pass/fail) before the judge would normally even be called -- the main
    pipeline skips the judge entirely on a hard-check failure to save
    tokens. Calibration deliberately breaks that rule: it forces a judge
    call on every sampled codegen attempt *regardless* of whether the tests
    passed, purely so "did the judge think this passed" can be compared
    against "did the tests actually pass" on the same output. That
    comparison is the actual measurement of self-bias this project claims
    to take seriously, not just gesture at.

(b) A generic agreement-stats helper (percent agreement + Cohen's kappa)
    that works on any set of (judge_verdict, external_label) pairs -- used
    for the codegen confusion matrix, and reusable for a human-agreement
    spot check if/when independent labels are available. See readme.html
    for exactly how this build's human-agreement numbers (if any) were
    collected and their limitations -- this module doesn't fabricate labels.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel

from reflect import metrics
from reflect.config import CONFIG
from reflect.generator import generate
from reflect.hard_checks import HardCheckResult, run_hard_checks
from reflect.judge import JudgeResult, judge
from reflect.llm import LLMError
from reflect.tasks import Task, all_bench_tasks

_CALIBRATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch TEXT NOT NULL,
    task_id TEXT NOT NULL,
    output TEXT NOT NULL,
    tests_passed INTEGER,
    judge_says_pass INTEGER NOT NULL,
    judge_mean REAL NOT NULL,
    external_label INTEGER,
    ts REAL NOT NULL
);
"""


class CalibrationSample(BaseModel):
    task_id: str
    output: str
    tests_passed: bool
    judge_says_pass: bool
    judge_mean: float
    hard: HardCheckResult
    judge_result: JudgeResult


class CalibrationFailure(BaseModel):
    """One instance that could not be sampled -- a live-call failure (LLM
    timeout, malformed judge output after retries, etc.), recorded as data
    rather than allowed to kill the whole batch. A calibration run with
    7/8 completed and 1 recorded failure is valid; a run that returns
    nothing because instance #3 of 8 timed out is not."""

    task_id: str
    error: str


class ConfusionMatrix(BaseModel):
    tp: int  # judge says pass, tests pass
    fp: int  # judge says pass, tests FAIL  <- the dangerous case
    fn: int  # judge says fail, tests pass
    tn: int  # judge says fail, tests fail

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else float("nan")

    @property
    def false_positive_rate(self) -> float:
        """Of the attempts whose tests actually FAILED, what fraction did
        the judge wrongly call a pass? This is the number that matters:
        every point of this rate is a case where trusting the judge alone,
        with no sandbox, would have shipped broken code."""
        denom = self.fp + self.tn
        return self.fp / denom if denom else float("nan")


def confusion_matrix(samples: list[CalibrationSample]) -> ConfusionMatrix:
    tp = sum(1 for s in samples if s.judge_says_pass and s.tests_passed)
    fp = sum(1 for s in samples if s.judge_says_pass and not s.tests_passed)
    fn = sum(1 for s in samples if not s.judge_says_pass and s.tests_passed)
    tn = sum(1 for s in samples if not s.judge_says_pass and not s.tests_passed)
    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


def agreement_stats(pairs: list[tuple[bool, bool]]) -> dict:
    """Percent agreement + Cohen's kappa for a set of (verdict_a, verdict_b)
    boolean pairs -- domain-agnostic, works for judge-vs-tests or
    judge-vs-human."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "percent_agreement": float("nan"), "cohens_kappa": float("nan")}
    po = sum(1 for a, b in pairs if a == b) / n
    a_rate = sum(1 for a, _ in pairs if a) / n
    b_rate = sum(1 for _, b in pairs if b) / n
    pe = a_rate * b_rate + (1 - a_rate) * (1 - b_rate)
    kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    return {"n": n, "percent_agreement": po, "cohens_kappa": kappa}


def _calibration_sample(task: Task) -> CalibrationSample:
    """One attempt-1 candidate for `task`, judged unconditionally (bypassing
    the normal hard-check gate) so both ground-truth and judge signals exist
    on the same output."""
    output = generate(task)
    hard = run_hard_checks(task, output)
    judge_result = judge(task, output)
    tests_passed = hard.sandbox.all_passed if hard.sandbox is not None else hard.passed
    return CalibrationSample(
        task_id=task.id,
        output=output,
        tests_passed=tests_passed,
        judge_says_pass=judge_result.passed(task.rubric()),
        judge_mean=judge_result.mean_score(),
        hard=hard,
        judge_result=judge_result,
    )


def _write_checkpoint(
    path: Path, batch: str, samples: list[CalibrationSample], failures: list[CalibrationFailure], total_planned: int
) -> None:
    """Overwrite the checkpoint file with current progress. Called after
    every instance (success or failure), not just at the end -- a crash or
    a killed process mid-batch should still leave whatever completed on
    disk, the same discipline the live-cycle metrics store already has via
    per-attempt logging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch": batch,
        "total_planned": total_planned,
        "completed": [s.task_id for s in samples],
        "failed": [{"task_id": f.task_id, "error": f.error} for f in failures],
        "updated_at": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2))


def run_codegen_calibration(
    n: int = CONFIG.calibration_subset_size,
    db_path: Path | None = None,
    batch: str = "codegen_judge_vs_tests",
    checkpoint_path: Path | None = None,
) -> tuple[ConfusionMatrix, list[CalibrationSample], list[CalibrationFailure]]:
    """Run the judge-vs-tests calibration on the first `n` codegen bench
    instances. This is the ~8-instance *subset* referenced throughout the
    README -- deliberately not the full 10 codegen instances or the whole
    30-instance bench, to keep the shared Ollama load bounded for a single
    live verification pass.

    Each instance's generate+judge call is isolated: an `LLMError` (a live
    timeout, or a judge that never returned schema-valid scores even after
    its own retry) is recorded as a `CalibrationFailure` and the batch
    continues, rather than one bad instance discarding everything already
    completed. Progress is checkpointed to `checkpoint_path` after every
    instance, and each successful sample is written to the metrics DB
    immediately rather than batched at the end, for the same reason.
    """
    checkpoint_path = checkpoint_path or (CONFIG.data_dir / "calibration_checkpoint.json")
    tasks = [t for t in all_bench_tasks() if t.domain == "codegen"][:n]
    samples: list[CalibrationSample] = []
    failures: list[CalibrationFailure] = []

    with metrics.get_connection(db_path) as conn:
        conn.execute(_CALIBRATION_SCHEMA)

    for task in tasks:
        try:
            sample = _calibration_sample(task)
        except LLMError as exc:
            failures.append(CalibrationFailure(task_id=task.id, error=str(exc)))
            _write_checkpoint(checkpoint_path, batch, samples, failures, len(tasks))
            continue

        samples.append(sample)
        with metrics.get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO calibration (batch, task_id, output, tests_passed, judge_says_pass, "
                "judge_mean, external_label, ts) VALUES (?,?,?,?,?,?,?,?)",
                (
                    batch, sample.task_id, sample.output, int(sample.tests_passed),
                    int(sample.judge_says_pass), sample.judge_mean, None, time.time(),
                ),
            )
        _write_checkpoint(checkpoint_path, batch, samples, failures, len(tasks))

    cm = confusion_matrix(samples)
    return cm, samples, failures


def fetch_calibration(batch: str | None = None, db_path: Path | None = None) -> list[dict]:
    with metrics.get_connection(db_path) as conn:
        conn.execute(_CALIBRATION_SCHEMA)
        if batch:
            rows = conn.execute("SELECT * FROM calibration WHERE batch = ? ORDER BY id", (batch,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM calibration ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def chart_confusion_matrix(cm: ConfusionMatrix, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = [[cm.tp, cm.fn], [cm.fp, cm.tn]]  # rows: tests pass/fail; cols: judge pass/fail
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["judge: pass", "judge: fail"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["tests: pass", "tests: fail"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=14,
                     color="white" if matrix[i][j] > max(cm.total, 1) / 4 else "black")
    fpr = cm.false_positive_rate
    fpr_str = f"{fpr:.0%}" if fpr == fpr else "n/a"
    ax.set_title(f"Judge vs. ground truth (codegen)\nfalse-positive rate = {fpr_str}, n={cm.total}")
    fig.colorbar(im, ax=ax, label="count")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
