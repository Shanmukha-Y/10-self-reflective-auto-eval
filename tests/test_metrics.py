"""Metrics store queries and chart generation. No LLM -- attempts are
inserted directly."""

from __future__ import annotations

from reflect import metrics


def _rec(run_id, domain, attempt, mean, passed, task_id="t"):
    return metrics.AttemptRecord(
        run_id=run_id, task_id=task_id, domain=domain, attempt_number=attempt,
        output="out", hard_passed=True, hard_detail="ok", judge_ran=True,
        judge_scores={"a": mean, "b": mean}, judge_mean=float(mean), judge_min=mean,
        judge_passed=passed, code_tests_passed=None, code_tests_total=None,
        latency_s=1.0,
    )


def test_pass_at_k_counts_first_attempt_and_any_attempt(tmp_path):
    db = tmp_path / "m.sqlite3"
    # run 1: passes on attempt 1
    metrics.log_attempt(_rec("r1", "summarize", 1, 5, True), db)
    metrics.mark_accepted("r1", 1, "passed", db)
    # run 2: fails attempt 1, passes attempt 2
    metrics.log_attempt(_rec("r2", "summarize", 1, 2, False), db)
    metrics.log_attempt(_rec("r2", "summarize", 2, 5, True), db)
    metrics.mark_accepted("r2", 2, "passed", db)
    # run 3: never passes (exhausted)
    metrics.log_attempt(_rec("r3", "summarize", 1, 2, False), db)
    metrics.mark_accepted("r3", 1, "exhausted", db)

    pak = metrics.pass_at_k(db)
    assert pak["summarize"][1] == 1 / 3  # only r1 passed on attempt 1
    assert pak["summarize"][3] == 2 / 3  # r1 and r2 eventually passed


def test_pending_runs_excluded_from_pass_at_k(tmp_path):
    db = tmp_path / "m.sqlite3"
    metrics.log_attempt(_rec("r1", "summarize", 1, 5, True), db)
    # never call mark_accepted -> final_status stays 'pending'
    pak = metrics.pass_at_k(db)
    assert pak == {}


def test_mean_score_by_attempt(tmp_path):
    db = tmp_path / "m.sqlite3"
    metrics.log_attempt(_rec("r1", "summarize", 1, 2, False), db)
    metrics.log_attempt(_rec("r2", "summarize", 1, 4, False), db)
    metrics.log_attempt(_rec("r1", "summarize", 2, 5, True), db)
    by_attempt = metrics.mean_score_by_attempt(db)
    assert by_attempt[1] == 3.0  # mean of 2 and 4
    assert by_attempt[2] == 5.0


def test_criterion_scores_filters_by_domain(tmp_path):
    db = tmp_path / "m.sqlite3"
    metrics.log_attempt(_rec("r1", "summarize", 1, 4, True), db)
    metrics.log_attempt(_rec("r2", "codegen", 1, 2, False), db)
    scores = metrics.criterion_scores(domain="summarize", db_path=db)
    assert scores["a"] == [4]
    assert "codegen" not in scores


def test_generate_charts_writes_files(tmp_path):
    db = tmp_path / "m.sqlite3"
    out_dir = tmp_path / "reports"
    metrics.log_attempt(_rec("r1", "summarize", 1, 5, True), db)
    metrics.mark_accepted("r1", 1, "passed", db)
    metrics.log_attempt(_rec("r2", "codegen", 1, 3, False), db)
    metrics.log_attempt(_rec("r2", "codegen", 2, 5, True), db)
    metrics.mark_accepted("r2", 2, "passed", db)

    written = metrics.generate_charts(out_dir=out_dir, db_path=db)
    assert len(written) == 3
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0
