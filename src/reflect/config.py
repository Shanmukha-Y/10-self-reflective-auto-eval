"""Central configuration: models, timeouts, thresholds, and paths.

This is the ONE place model names are defined. The generator, judge, and
critic all use the same Ollama model -- distinct roles are expressed through
distinct system prompts, not distinct models. That shared-model choice is
also exactly why the self-bias mitigations in judge.py (blind scoring,
anchored rubrics, temperature 0) and calibrate.py (measured judge-vs-tests
agreement) exist: nothing here pretends a same-model judge is unbiased.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # --- Ollama connection ---
    ollama_host: str = "http://localhost:11434"

    # --- Model (the ONE place this name lives; generator/judge/critic share it) ---
    model: str = "qwen3.5:9b"

    # --- Paths ---
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    rubrics_dir: Path = project_root / "rubrics"
    data_dir: Path = project_root / "data"
    sqlite_path: Path = data_dir / "reflect.sqlite3"
    reports_dir: Path = project_root / "reports"
    bench_inputs_dir: Path = project_root / "bench" / "inputs"

    # --- Loop control ---
    max_attempts: int = 3

    # --- LLM call tuning ---
    # Measured, not guessed: two independent live timeouts were observed at
    # the original 240s -- one on a generator call while three builders were
    # concurrently hitting the shared Ollama instance, and a second on a
    # judge call made with *exclusive* server access (calibrate.py), where
    # contention from other builders can't explain it. The judge call was a
    # generate_json() request for a full anchored 3-criterion rubric, which
    # is a larger prompt than a typical chat() call. Rather than guess at a
    # judge-specific override, 240s -> 300s across the board: cheap
    # (adds latency only on the rare call that actually needs it) and
    # addresses both observed failure modes without inventing a second knob
    # from a sample size of two.
    llm_timeout_s: float = 300.0
    llm_max_retries: int = 1

    # --- Role temperatures ---
    generator_temperature: float = 0.7
    judge_temperature: float = 0.0  # deterministic scoring
    critic_temperature: float = 0.2

    # --- Sandbox (code execution ground truth) ---
    sandbox_timeout_s: float = 10.0

    # --- Calibration ---
    calibration_subset_size: int = 8


CONFIG = Config()
