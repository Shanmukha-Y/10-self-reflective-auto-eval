"""Task definitions: prompt, rubric reference, hard constraints, and
optional code ground-truth tests.

Rubrics are versioned as YAML in `rubrics/` -- each criterion carries
anchored descriptions for scores 1/3/5. Anchoring is what makes a 9B judge
usable at all: without concrete "a 1 looks like this, a 5 looks like this"
examples, a small model's 1-5 scores are barely more informative than a
coin flip.

Task *instances* (both the handful used for `reflect run --task <id>` demos
and the full bench set) are stored as YAML files under `bench/inputs/` and
loaded through the same `Task` model, so there is exactly one schema and one
loader for "a task" regardless of whether it's used interactively or in the
30-instance bench.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from reflect.config import CONFIG

Domain = Literal["summarize", "constrained_write", "codegen"]


class Criterion(BaseModel):
    name: str
    description: str
    anchors: dict[int, str]


class Rubric(BaseModel):
    name: str
    pass_threshold: float
    criterion_floor: int
    criteria: list[Criterion]

    @classmethod
    def load(cls, name: str) -> "Rubric":
        path = CONFIG.rubrics_dir / f"{name}.yaml"
        data = yaml.safe_load(path.read_text())
        return cls.model_validate(data)


class HardConstraints(BaseModel):
    min_words: int | None = None
    max_words: int | None = None
    must_include_all: list[str] = Field(default_factory=list)
    must_include_any: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)


class CodeSpec(BaseModel):
    function_name: str
    starter_signature: str
    test_asserts: list[str]


class Task(BaseModel):
    id: str
    domain: Domain
    title: str
    instructions: str
    input_text: str | None = None
    rubric_name: str
    hard: HardConstraints = Field(default_factory=HardConstraints)
    code: CodeSpec | None = None

    def rubric(self) -> Rubric:
        return Rubric.load(self.rubric_name)

    def with_input_text(self, text: str) -> "Task":
        """Return a copy of this task with `input_text` overridden -- used
        by `reflect run --task ... --input file.txt` to supply a document
        without needing a dedicated bench file per document."""
        return self.model_copy(update={"input_text": text})


def load_task(path: Path) -> Task:
    data = yaml.safe_load(path.read_text())
    return Task.model_validate(data)


def all_bench_tasks() -> list[Task]:
    return [load_task(p) for p in sorted(CONFIG.bench_inputs_dir.glob("*.yaml"))]


def get_task(task_id: str) -> Task:
    for t in all_bench_tasks():
        if t.id == task_id:
            return t
    known = ", ".join(t.id for t in all_bench_tasks())
    raise KeyError(f"Unknown task id {task_id!r}. Known task ids: {known}")
