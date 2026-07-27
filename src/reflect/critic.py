"""Critic: diagnoses *why* judged criteria fell short and emits
machine-usable constraints for the next attempt.

Separate role from the judge, and a separate call. The judge is evaluative
-- its scores are treated as final for that attempt. The critic is
generative/diagnostic -- it produces guidance for what to change. Folding
these into one call would let the model's own explanation of what's wrong
retroactively negotiate its score; keeping them separate means nothing the
critic says afterward can change the score attempt k actually received.

The critic is deliberately NOT blind the way the judge is: it needs the
judge's per-criterion scores and justifications to diagnose root causes.
What it produces is a list of `Constraint`, consumed by constraints.py and
generator.py -- never a revised score.

Note: this module only runs when hard checks passed but the judge failed.
Hard-check failures (word count, missing string, failing test case) are
already mechanically diagnosed and go straight to constraints.py without a
critic call -- there's nothing for an LLM to diagnose in "the output was
40 words over the limit."
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from reflect.config import CONFIG
from reflect.judge import JudgeResult
from reflect.llm import generate_json
from reflect.tasks import Task

_CRITIC_SYSTEM = (
    "You are a precise editor diagnosing why a piece of writing or code "
    "fell short of a rubric. For each criterion that scored below the "
    "rubric's pass threshold, write one concrete, machine-checkable "
    "constraint the next attempt must satisfy -- not vague advice. "
    "Constraints should be specific enough that a different writer could "
    "verify them by inspection (e.g. 'state the return window is 30 days', "
    "not 'be more complete'). Do not write constraints for criteria that "
    "already scored well."
)


class Constraint(BaseModel):
    constraint: str
    reason: str

    @field_validator("constraint")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("constraint must not be empty")
        return v


class Critique(BaseModel):
    constraints: list[Constraint]


def _build_prompt(task: Task, output: str, judge_result: JudgeResult) -> str:
    rubric = task.rubric()
    scores_block = "\n".join(f"- {s.name}: {s.score}/5 -- {s.justification}" for s in judge_result.scores)
    return (
        f"Task instructions:\n{task.instructions}\n\n"
        f"Output that was evaluated:\n---\n{output}\n---\n\n"
        f"Rubric pass threshold: mean score >= {rubric.pass_threshold}, "
        f"every criterion >= {rubric.criterion_floor}.\n\n"
        f"Judge scores:\n{scores_block}\n\n"
        "For every criterion that scored below the floor, or that is dragging "
        "the mean below the pass threshold, diagnose the specific reason it "
        "fell short and write one constraint the next attempt must satisfy to "
        'fix it. Return JSON: {"constraints": [{"constraint": <specific '
        'requirement>, "reason": <why this was needed, referencing the '
        "score/justification above>}, ...]}"
    )


def critique(task: Task, output: str, judge_result: JudgeResult) -> Critique:
    prompt = _build_prompt(task, output, judge_result)
    result = generate_json(
        prompt=prompt,
        schema=Critique,
        system=_CRITIC_SYSTEM,
        temperature=CONFIG.critic_temperature,
    )
    assert isinstance(result, Critique)
    return result
