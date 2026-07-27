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
    # Ollama here is shared across several concurrent builders; under load a
    # single call has been observed to take up to ~240s, so timeouts are
    # generous and every caller also catches a bare socket TimeoutError
    # (Ollama's client doesn't always wrap that as its own exception type).
    llm_timeout_s: float = 240.0
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
