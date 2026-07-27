"""CLI entry point: `reflect run`, `reflect bench`, `reflect report`, `reflect calibrate`."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from reflect import calibrate as calibrate_mod
from reflect import metrics
from reflect.config import CONFIG
from reflect.pipeline import run_cycle
from reflect.tasks import all_bench_tasks, get_task

console = Console()


@click.group()
def main() -> None:
    """Self-reflective agent: generate -> judge -> critique -> regenerate."""


@main.command()
@click.option("--task", "task_id", required=True, help="Task id (see `reflect bench --list`).")
@click.option(
    "--input", "input_path", type=click.Path(exists=True, path_type=Path), default=None,
    help="Override the task's input_text with this file's contents (summarize tasks).",
)
@click.option("--max-attempts", default=CONFIG.max_attempts, show_default=True)
@click.option("--show-cycles", is_flag=True, help="Print every attempt: output, hard checks, judge scores, constraints.")
def run(task_id: str, input_path: Path | None, max_attempts: int, show_cycles: bool) -> None:
    """Run the full generate -> judge -> critique -> regenerate loop on one task."""
    task = get_task(task_id)
    if input_path is not None:
        task = task.with_input_text(input_path.read_text())

    result = run_cycle(task, max_attempts=max_attempts, show_cycles=show_cycles)

    status_color = "green" if result.final_status == "passed" else "yellow"
    console.print()
    console.print(
        f"[bold]{task.id}[/bold] ({task.domain}): [{status_color}]{result.final_status}[/{status_color}] "
        f"after {len(result.attempts)} attempt(s)"
    )
    if result.final_status != "passed":
        console.print("[yellow]Returned the best-scoring attempt -- flagged as below rubric threshold, not silently accepted.[/yellow]")
    console.print()
    console.print(result.final_output)


@main.command()
@click.option("--domain", type=click.Choice(["summarize", "constrained_write", "codegen"]), default=None)
@click.option("--limit", type=int, default=None, help="Only run the first N matching instances.")
@click.option("--max-attempts", default=CONFIG.max_attempts, show_default=True)
@click.option("--show-cycles", is_flag=True)
@click.option("--list", "list_only", is_flag=True, help="List available task ids and exit, running nothing.")
def bench(domain: str | None, limit: int | None, max_attempts: int, show_cycles: bool, list_only: bool) -> None:
    """Run the pipeline over bench task instances.

    Default is all 30 instances -- see readme.html before running that live;
    it takes several minutes against a shared local Ollama instance. Use
    --limit / --domain for a quick smoke run.
    """
    tasks = all_bench_tasks()
    if domain:
        tasks = [t for t in tasks if t.domain == domain]

    if list_only:
        for t in tasks:
            console.print(f"{t.id}  [{t.domain}]  {t.title}")
        return

    if limit:
        tasks = tasks[:limit]

    for i, task in enumerate(tasks, start=1):
        result = run_cycle(task, max_attempts=max_attempts, show_cycles=show_cycles)
        console.print(f"[{i}/{len(tasks)}] {task.id}: {result.final_status} in {len(result.attempts)} attempt(s)")


@main.command()
def report() -> None:
    """Print pass@k / score-by-attempt tables and regenerate charts in reports/."""
    pak = metrics.pass_at_k()
    if not pak:
        console.print("[yellow]No completed runs in the metrics store yet. Run `reflect run` or `reflect bench` first.[/yellow]")
        return

    table = Table(title="pass@1 vs pass@3 by domain")
    table.add_column("domain")
    table.add_column("pass@1")
    table.add_column("pass@3")
    table.add_column("lift")
    for domain, rates in sorted(pak.items()):
        table.add_row(domain, f"{rates[1]:.0%}", f"{rates[3]:.0%}", f"+{(rates[3] - rates[1]):.0%}")
    console.print(table)

    by_attempt = metrics.mean_score_by_attempt()
    if by_attempt:
        table2 = Table(title="mean judge score by attempt number")
        table2.add_column("attempt")
        table2.add_column("mean score")
        for k, v in sorted(by_attempt.items()):
            table2.add_row(str(k), f"{v:.2f}")
        console.print(table2)

    written = metrics.generate_charts()
    console.print(f"\nCharts written to {CONFIG.reports_dir}: {[p.name for p in written]}")


@main.command()
@click.option("--n", default=CONFIG.calibration_subset_size, show_default=True, help="Number of codegen instances to calibrate on.")
def calibrate(n: int) -> None:
    """Run the judge-vs-tests calibration subset on codegen tasks and report the confusion matrix."""
    console.print(f"Running judge-vs-tests calibration on {n} codegen instance(s)...")
    cm, _samples, failures = calibrate_mod.run_codegen_calibration(n=n)

    if failures:
        console.print(f"[yellow]{len(failures)}/{n} instance(s) failed to sample (recorded, not silently dropped):[/yellow]")
        for f in failures:
            console.print(f"  [yellow]- {f.task_id}: {f.error}[/yellow]")

    table = Table(title=f"Judge vs. ground truth (n={cm.total} completed of {n} planned)")
    table.add_column("")
    table.add_column("judge: pass")
    table.add_column("judge: fail")
    table.add_row("tests: pass", str(cm.tp), str(cm.fn))
    table.add_row("tests: fail", str(cm.fp), str(cm.tn))
    console.print(table)
    console.print(
        f"Accuracy: {cm.accuracy:.0%}   "
        f"False-positive rate (judge says pass, tests actually fail): {cm.false_positive_rate:.0%}"
    )

    path = calibrate_mod.chart_confusion_matrix(cm, CONFIG.reports_dir / "calibration_confusion_matrix.png")
    console.print(f"Chart written to {path}")


if __name__ == "__main__":
    main()
