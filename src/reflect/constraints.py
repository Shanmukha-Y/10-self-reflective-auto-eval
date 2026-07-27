"""Compiles hard-check failures and critic diagnoses into the constraint
list handed to the generator for the next attempt.

Two sources feed in:
- Hard-check failures are already mechanically diagnosed (a word count, a
  missing string, a failing test case) -- turned directly into constraints
  here with no LLM call, since there's nothing to diagnose.
- Judge failures (hard checks passed, rubric didn't) go through the critic
  first (critic.py); its `Constraint` objects are merged in here.

Constraints are deduped by normalized text: if the same requirement would
appear twice (e.g. carried over from an earlier attempt and re-diagnosed by
the critic again) only one copy survives, so the regeneration prompt never
repeats itself.
"""

from __future__ import annotations

from reflect.critic import Constraint
from reflect.hard_checks import HardCheckResult


def constraints_from_hard_checks(hard: HardCheckResult) -> list[Constraint]:
    out: list[Constraint] = []
    for check in hard.checks:
        if check.passed:
            continue
        if check.name == "min_words":
            out.append(Constraint(constraint=f"Write at least the required length ({check.detail}).", reason=check.detail))
        elif check.name == "max_words":
            out.append(Constraint(constraint=f"Stay within the required length ({check.detail}).", reason=check.detail))
        elif check.name.startswith("must_include:"):
            phrase = check.name.split(":", 1)[1]
            out.append(Constraint(constraint=f'You must explicitly mention: "{phrase}".', reason="required phrase was missing"))
        elif check.name == "must_include_any":
            out.append(Constraint(constraint="You must include at least one of the required alternative phrases.", reason=check.detail))
        elif check.name.startswith("forbidden:"):
            phrase = check.name.split(":", 1)[1]
            out.append(Constraint(constraint=f'Do not mention "{phrase}".', reason="forbidden phrase was present"))
        elif check.name == "code_tests" and hard.sandbox is not None:
            if hard.sandbox.error:
                out.append(Constraint(constraint=f"Fix the execution error: {hard.sandbox.error}", reason="code failed to execute"))
            for tc in hard.sandbox.results:
                if not tc.passed:
                    out.append(Constraint(
                        constraint=f"Fix test case {tc.case}: it currently fails with {tc.error}.",
                        reason=f"code test case {tc.case} failed",
                    ))
    return out


def dedupe(constraints: list[Constraint]) -> list[Constraint]:
    seen: set[str] = set()
    out: list[Constraint] = []
    for c in constraints:
        key = " ".join(c.constraint.lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def compile_constraints(*groups: list[Constraint]) -> list[Constraint]:
    """Merge constraint groups (e.g. constraints carried over from earlier
    attempts + newly diagnosed ones) into one deduped list, first-seen
    order preserved."""
    merged: list[Constraint] = []
    for group in groups:
        merged.extend(group)
    return dedupe(merged)
