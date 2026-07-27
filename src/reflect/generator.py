"""Produces a candidate output for a task.

Attempt 1 gets just the task prompt. Attempt k>1 receives the *compiled
constraints* from the prior failure -- never the raw failed output. This is
the design thesis of the project: regeneration is constraint-driven, not
"here's what you wrote, try again," which would just let the model anchor
on its own bad draft instead of addressing the specific, diagnosed gap.
"""

from __future__ import annotations

from reflect.config import CONFIG
from reflect.llm import chat
from reflect.tasks import Task

_SYSTEM_PROMPTS = {
    "summarize": (
        "You are a careful technical summarizer. Follow the instructions "
        "exactly and output only the summary text, no preamble or labels."
    ),
    "constrained_write": (
        "You are a precise writer who follows explicit constraints exactly. "
        "Output only the requested piece of writing, no preamble or explanation."
    ),
    "codegen": (
        "You are a careful Python programmer. Output a single Python code "
        "block implementing exactly what is asked, no explanation before or after."
    ),
}


def _build_prompt(task: Task, constraints: list[str] | None) -> str:
    parts = [task.instructions]
    if task.input_text:
        parts.append(f"\nSource text:\n---\n{task.input_text}\n---")
    if task.code is not None:
        parts.append(f"\nImplement a function with this signature:\n{task.code.starter_signature}")
    if constraints:
        bullet_list = "\n".join(f"- {c}" for c in constraints)
        parts.append(
            "\nYour previous attempt failed review. You MUST satisfy every one "
            f"of these constraints in this attempt:\n{bullet_list}"
        )
    return "\n".join(parts)


def generate(task: Task, constraints: list[str] | None = None) -> str:
    """Produce one candidate output. `constraints` is the compiled list from
    the prior attempt's failure (hard-check or critic derived); empty/None
    on attempt 1."""
    system = _SYSTEM_PROMPTS[task.domain]
    prompt = _build_prompt(task, constraints)
    return chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=CONFIG.generator_temperature,
    )
