"""LLM-as-judge: blind, rubric-anchored scoring.

Blindness is deliberate and specific: the judge sees the task, the rubric,
and the candidate output -- nothing else. No attempt number, no prior
critiques, no generator "reasoning," no indication this is a retry. That
matters because the generator and judge are the same underlying model
(qwen3.5:9b, distinct system prompts only). If the judge could see "this is
attempt 2, previously critiqued for missing fact X," a same-model judge has
every incentive, in the sense of what a next-token predictor learns to do,
to rubber-stamp visible effort rather than score the output on its own
merits. Anchored 1/3/5 descriptions per criterion and temperature 0 are the
other two mitigations here. None of this makes the judge unbiased --
calibrate.py measures how much to trust it, which is the honest version of
"the judge is not the same thing as ground truth."
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from reflect.config import CONFIG
from reflect.llm import LLMError, generate_json
from reflect.tasks import Rubric, Task

_JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator. You did not write the text you "
    "are scoring and have no stake in it being good or bad. Score only what "
    "is on the page against the rubric provided. Do not infer intent, do "
    "not give credit for effort, and do not assume anything not stated in "
    "the output itself."
)


class CriterionScore(BaseModel):
    name: str
    score: int = Field(ge=1, le=5)
    justification: str

    @field_validator("justification")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("justification must not be empty")
        return v


class JudgeResult(BaseModel):
    scores: list[CriterionScore]

    def mean_score(self) -> float:
        return sum(c.score for c in self.scores) / len(self.scores)

    def min_score(self) -> int:
        return min(c.score for c in self.scores)

    def passed(self, rubric: Rubric) -> bool:
        return self.mean_score() >= rubric.pass_threshold and self.min_score() >= rubric.criterion_floor


def _build_prompt(task: Task, rubric: Rubric, output: str) -> str:
    criteria_block = "\n\n".join(
        f"### {c.name}\n{c.description}\n"
        + "\n".join(f"  Score {k} looks like: {v}" for k, v in sorted(c.anchors.items()))
        for c in rubric.criteria
    )
    source_block = f"Source text the writer was given:\n---\n{task.input_text}\n---\n\n" if task.input_text else ""
    return (
        f"Task instructions given to the writer:\n{task.instructions}\n\n"
        f"{source_block}"
        f"Output to evaluate:\n---\n{output}\n---\n\n"
        f"Rubric ({len(rubric.criteria)} criteria, score each 1-5 using the anchors below):\n{criteria_block}\n\n"
        'Return JSON of the form {"scores": [{"name": <criterion name exactly '
        'as given above>, "score": <integer 1-5>, "justification": <one '
        "sentence citing something specific from the output>}, ...]} with "
        "exactly one entry per criterion listed above."
    )


def judge(task: Task, output: str) -> JudgeResult:
    """Blind rubric scoring. Retries once at the semantic level (not just
    JSON-validity, which llm.generate_json already retries) if the model
    returns scores for the wrong set of criteria."""
    rubric = task.rubric()
    expected_names = {c.name for c in rubric.criteria}
    prompt = _build_prompt(task, rubric, output)

    last_got: set[str] = set()
    for _ in range(2):
        result = generate_json(
            prompt=prompt,
            schema=JudgeResult,
            system=_JUDGE_SYSTEM,
            temperature=CONFIG.judge_temperature,
        )
        assert isinstance(result, JudgeResult)
        got_names = {s.name for s in result.scores}
        if got_names == expected_names:
            return result
        last_got = got_names

    raise LLMError(
        f"Judge did not return scores for exactly the rubric criteria {sorted(expected_names)}; "
        f"got {sorted(last_got)}"
    )
