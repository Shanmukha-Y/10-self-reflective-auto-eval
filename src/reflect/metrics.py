"""SQLite metrics store: every attempt's hard-check result, judge scores,
timing, and outcome land here so improvement claims are read off a table,
not asserted. Also builds the charts referenced in the README: score vs
attempt, pass@1 vs pass@3, and the per-criterion weakness heatmap.
(Judge-vs-ground-truth calibration charts live in calibrate.py, which reuses
`_connect`/the schema but has its own confusion-matrix logic.)
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from reflect.config import CONFIG

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    ts REAL NOT NULL,
    output TEXT NOT NULL,
    hard_passed INTEGER NOT NULL,
    hard_detail TEXT NOT NULL,
    judge_ran INTEGER NOT NULL,
    judge_scores_json TEXT,
    judge_mean REAL,
    judge_min INTEGER,
    judge_passed INTEGER,
    code_tests_passed INTEGER,
    code_tests_total INTEGER,
    accepted INTEGER NOT NULL DEFAULT 0,
    final_status TEXT NOT NULL DEFAULT 'pending',
    latency_s REAL NOT NULL
);
"""


@contextmanager
def _connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or CONFIG.sqlite_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_connection(db_path: Path | None = None):
    """Public entry point for other modules (calibrate.py) that need their
    own tables in the same SQLite file -- keeps one DB file per run without
    coupling calibrate.py to this module's private connection helper."""
    return _connect(db_path)


@dataclass
class AttemptRecord:
    run_id: str
    task_id: str
    domain: str
    attempt_number: int
    output: str
    hard_passed: bool
    hard_detail: str
    judge_ran: bool
    judge_scores: dict[str, int] | None
    judge_mean: float | None
    judge_min: int | None
    judge_passed: bool | None
    code_tests_passed: int | None
    code_tests_total: int | None
    latency_s: float
    ts: float = field(default_factory=time.time)


def log_attempt(rec: AttemptRecord, db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO attempts (run_id, task_id, domain, attempt_number, ts, output, "
            "hard_passed, hard_detail, judge_ran, judge_scores_json, judge_mean, judge_min, "
            "judge_passed, code_tests_passed, code_tests_total, latency_s) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rec.run_id, rec.task_id, rec.domain, rec.attempt_number, rec.ts, rec.output,
                int(rec.hard_passed), rec.hard_detail, int(rec.judge_ran),
                json.dumps(rec.judge_scores) if rec.judge_scores is not None else None,
                rec.judge_mean, rec.judge_min,
                None if rec.judge_passed is None else int(rec.judge_passed),
                rec.code_tests_passed, rec.code_tests_total, rec.latency_s,
            ),
        )


def mark_accepted(run_id: str, attempt_number: int, final_status: str, db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute("UPDATE attempts SET final_status = ? WHERE run_id = ?", (final_status, run_id))
        conn.execute(
            "UPDATE attempts SET accepted = 1 WHERE run_id = ? AND attempt_number = ?",
            (run_id, attempt_number),
        )


def fetch_all(db_path: Path | None = None) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM attempts ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def pass_at_k(db_path: Path | None = None) -> dict[str, dict[int, float]]:
    """domain -> {1: pass@1 rate, 3: pass@3 rate} over completed runs
    (runs where final_status is no longer 'pending')."""
    rows = fetch_all(db_path)
    runs: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        runs[r["run_id"]].append(r)

    by_domain: dict[str, list[list[dict]]] = defaultdict(list)
    for run_id, attempts in runs.items():
        attempts = sorted(attempts, key=lambda a: a["attempt_number"])
        if attempts[-1]["final_status"] == "pending":
            continue  # run never completed
        by_domain[attempts[0]["domain"]].append(attempts)

    out: dict[str, dict[int, float]] = {}
    for domain, run_list in by_domain.items():
        n = len(run_list)
        pass1 = sum(1 for a in run_list if a[0]["judge_passed"]) / n
        pass3 = sum(1 for a in run_list if any(x["judge_passed"] for x in a)) / n
        out[domain] = {1: pass1, 3: pass3}
    return out


def mean_score_by_attempt(db_path: Path | None = None) -> dict[int, float]:
    rows = [r for r in fetch_all(db_path) if r["judge_mean"] is not None]
    by_attempt: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        by_attempt[r["attempt_number"]].append(r["judge_mean"])
    return {k: sum(v) / len(v) for k, v in sorted(by_attempt.items())}


def criterion_scores(domain: str | None = None, db_path: Path | None = None) -> dict[str, list[int]]:
    """criterion name -> list of every score it ever received (optionally
    filtered to one domain), for the weakness heatmap."""
    rows = fetch_all(db_path)
    out: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if r["judge_scores_json"] is None:
            continue
        if domain is not None and r["domain"] != domain:
            continue
        for name, score in json.loads(r["judge_scores_json"]).items():
            out[name].append(score)
    return out


def generate_charts(out_dir: Path | None = None, db_path: Path | None = None) -> list[Path]:
    """Render score-by-attempt, pass@k, and criterion-heatmap PNGs. Returns
    the list of file paths written. Raises if there is no data yet."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or CONFIG.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # --- score by attempt ---
    by_attempt = mean_score_by_attempt(db_path)
    if by_attempt:
        fig, ax = plt.subplots(figsize=(6, 4))
        attempts = sorted(by_attempt)
        ax.plot(attempts, [by_attempt[a] for a in attempts], marker="o", color="#2563eb")
        ax.set_xlabel("Attempt number")
        ax.set_ylabel("Mean judge score (1-5)")
        ax.set_title("Score by attempt number")
        ax.set_xticks(attempts)
        ax.set_ylim(1, 5)
        ax.grid(alpha=0.3)
        path = out_dir / "score_by_attempt.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    # --- pass@k ---
    pak = pass_at_k(db_path)
    if pak:
        fig, ax = plt.subplots(figsize=(6, 4))
        domains = sorted(pak)
        x = range(len(domains))
        width = 0.35
        ax.bar([i - width / 2 for i in x], [pak[d][1] for d in domains], width, label="pass@1", color="#93c5fd")
        ax.bar([i + width / 2 for i in x], [pak[d][3] for d in domains], width, label="pass@3", color="#2563eb")
        ax.set_xticks(list(x))
        ax.set_xticklabels(domains)
        ax.set_ylabel("Pass rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("pass@1 vs pass@3 by domain")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        path = out_dir / "pass_at_k.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    # --- criterion heatmap (mean score per criterion per domain) ---
    domains_seen = sorted({r["domain"] for r in fetch_all(db_path)})
    if domains_seen:
        crit_names: list[str] = []
        matrix: list[list[float]] = []
        for domain in domains_seen:
            scores = criterion_scores(domain, db_path)
            for name in scores:
                if name not in crit_names:
                    crit_names.append(name)
        for domain in domains_seen:
            scores = criterion_scores(domain, db_path)
            row = [sum(scores[n]) / len(scores[n]) if scores.get(n) else float("nan") for n in crit_names]
            matrix.append(row)

        if crit_names:
            fig, ax = plt.subplots(figsize=(max(6, len(crit_names) * 1.2), max(3, len(domains_seen) * 0.8)))
            im = ax.imshow(matrix, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")
            ax.set_xticks(range(len(crit_names)))
            ax.set_xticklabels(crit_names, rotation=30, ha="right")
            ax.set_yticks(range(len(domains_seen)))
            ax.set_yticklabels(domains_seen)
            for i in range(len(domains_seen)):
                for j in range(len(crit_names)):
                    val = matrix[i][j]
                    if val == val:  # not NaN
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black", fontsize=9)
            ax.set_title("Mean criterion score by domain")
            fig.colorbar(im, ax=ax, label="mean score (1-5)")
            path = out_dir / "criterion_heatmap.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(path)

    return written
