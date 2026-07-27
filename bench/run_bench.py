"""Runs the full generate -> judge -> critique -> regenerate pipeline over
bench task instances, logging every attempt to the metrics store.

This is deliberately a thin script, not a package module: it's the one
place the "run the whole 30-instance bench" command lives, kept separate
from src/reflect so `reflect.*` stays importable without ever touching the
live model.

Usage:
    uv run python bench/run_bench.py                    # all 30 instances
    uv run python bench/run_bench.py --domain codegen    # one domain
    uv run python bench/run_bench.py --limit 3           # first 3 instances (smoke test)
    uv run python bench/run_bench.py --domain summarize --limit 1

The full 30-instance run is NOT part of this project's automated
verification -- it takes many minutes against a shared local Ollama
instance. See readme.html for what was actually run live and why.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reflect.config import CONFIG  # noqa: E402
from reflect.pipeline import run_cycle  # noqa: E402
from reflect.tasks import all_bench_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reflect pipeline over bench task instances.")
    parser.add_argument("--domain", choices=["summarize", "constrained_write", "codegen"], default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N matching instances.")
    parser.add_argument("--max-attempts", type=int, default=CONFIG.max_attempts)
    parser.add_argument("--show-cycles", action="store_true")
    args = parser.parse_args()

    tasks = all_bench_tasks()
    if args.domain:
        tasks = [t for t in tasks if t.domain == args.domain]
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"Running {len(tasks)} task instance(s): {[t.id for t in tasks]}")
    for i, task in enumerate(tasks, start=1):
        t0 = time.monotonic()
        result = run_cycle(task, max_attempts=args.max_attempts, show_cycles=args.show_cycles)
        dt = time.monotonic() - t0
        print(
            f"[{i}/{len(tasks)}] {task.id} ({task.domain}): {result.final_status} "
            f"in {len(result.attempts)} attempt(s), {dt:.1f}s"
        )

    print("\nDone. Run `uv run reflect report` to generate charts from the metrics store.")


if __name__ == "__main__":
    main()
