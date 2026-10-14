"""Configuration loader for the N-of-1 Causal Lab pipeline."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, assert_never

import yaml
from dotenv import load_dotenv
from pydantic import ConfigDict, TypeAdapter, with_config

from nof1_causal_lab.actions.errors import execution_failure_handler
from nof1_causal_lab.artifacts.sampling import SamplerSpec
from nof1_causal_lab.llm.specs import (
    EmbeddedLLMSpec,
    EmbeddedReasoningEffort,
    HarnessEffort,
    HarnessName,
    LLMProfileSpec,
    PiThinking,
)

# Centralized .env loading — all modules that need env vars import from config.py
# (or from modules that import config.py), so this runs once at import time.
load_dotenv(Path(__file__).resolve().parents[4] / ".env")


# ---------------------------------------------------------------------------
# LLM backend defaults (global)
# ---------------------------------------------------------------------------


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class EmbeddedLLMDefaults:
    """Defaults for ``harness: none`` (OpenRouter) contexts."""

    max_tokens: int = 65536
    timeout: int = 900
    reasoning_effort: EmbeddedReasoningEffort = "xhigh"


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class ClaudeCodeDefaults:
    """Defaults for ``harness: claude-code`` contexts."""

    bin: str = "claude"
    effort: HarnessEffort = "high"
    max_turns: int = 40
    max_budget_usd: float | None = None
    fallback_model: str | None = None


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class CodexDefaults:
    """Defaults for ``harness: codex`` contexts."""

    bin: str = "codex"
    reasoning_effort: HarnessEffort = "xhigh"
    service_tier: str = "fast"


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class PiDefaults:
    """Defaults for ``harness: pi`` contexts."""

    bin: str = "pi"
    provider: str = "openai-codex"
    thinking: PiThinking = "high"


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class LLMDefaults:
    """Global LLM backend defaults (one section per backend)."""

    embedded: EmbeddedLLMDefaults = field(default_factory=EmbeddedLLMDefaults)
    claude_code: ClaudeCodeDefaults = field(default_factory=ClaudeCodeDefaults)
    codex: CodexDefaults = field(default_factory=CodexDefaults)
    pi: PiDefaults = field(default_factory=PiDefaults)


# ---------------------------------------------------------------------------
# Agentic context configs
# ---------------------------------------------------------------------------


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class ExtractionWorkersConfig:
    """extraction: Support-Window Extraction (Workers).

    ``max_rpm`` only applies when ``llm.harness == 'none'``. extraction must
    stay on the embedded backend because fanning out thousands of workers
    through a harness CLI is economically and latency-wise untenable.
    """

    llm: EmbeddedLLMSpec
    max_concurrent_workers: int = 4
    max_events_per_window: int = 300
    max_rpm: int = 450
    worker_timeout: int = 120
    max_tool_turns: int = 40


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class InferenceConfig:
    """Shell settings alongside the single owned sampler specification."""

    compute_backend: Literal["local", "modal"] = "local"
    compute_loo_diagnostics: bool = True
    sampler: SamplerSpec = field(default_factory=SamplerSpec)


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class PipelineConfig:
    """Full pipeline configuration."""

    extraction_workers: ExtractionWorkersConfig
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    llm: LLMDefaults = field(default_factory=LLMDefaults)


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def get_secret(name: str) -> str | None:
    """Get a secret from environment variables."""
    return os.getenv(name)


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


def _find_config_path() -> Path:
    """Find config.yaml by walking up from this file to the project root."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        config_path = parent / "config.yaml"
        if config_path.exists():
            return config_path
    raise FileNotFoundError("config.yaml not found in any parent directory")


@lru_cache(maxsize=1)
def load_config(config_path: Path | None = None) -> PipelineConfig:
    """Load, parse, and validate the pipeline configuration."""
    config_path = config_path or _find_config_path()
    with config_path.open() as f:
        raw = yaml.safe_load(f) or {}

    config = TypeAdapter(PipelineConfig).validate_python(raw)

    schema_errors = validate_config(config)
    if schema_errors:
        raise ValueError(
            "config.yaml failed validation:\n" + "\n".join(f"  - {e}" for e in schema_errors)
        )
    return config


def get_config() -> PipelineConfig:
    """Get the pipeline configuration."""
    return load_config()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _iter_profile_llms(config: PipelineConfig) -> list[tuple[str, LLMProfileSpec]]:
    return [
        ("extraction_workers", config.extraction_workers.llm),
    ]


def validate_config(config: PipelineConfig) -> list[str]:
    """Check model naming after backend field validation at the YAML boundary."""
    return [
        f"{name}.llm.model: {llm.model!r} should start with 'openrouter/' for harness=none"
        for name, llm in _iter_profile_llms(config)
        if llm.harness == "none" and not llm.model.startswith("openrouter/")
    ]


def _check_embedded_prereqs() -> list[str]:
    errors: list[str] = []
    if not os.getenv("OPENROUTER_API_KEY"):
        errors.append("OPENROUTER_API_KEY is not set (required for harness=none)")
    return errors


@execution_failure_handler
def _check_claude_code_prereqs(config: PipelineConfig) -> list[str]:
    import subprocess

    errors: list[str] = []
    bin_name = config.llm.claude_code.bin
    bin_path = shutil.which(bin_name)
    if bin_path is None:
        errors.append(
            f"claude binary {bin_name!r} not found on PATH (required for harness=claude-code)"
        )
        return errors
    try:
        status = subprocess.run(
            [bin_path, "auth", "status", "--text"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        errors.append(f"`claude auth status` failed: {exc}")
    else:
        if status.returncode != 0:
            errors.append(
                f"claude is not logged in — run `claude auth login` (exit={status.returncode})"
            )
    return errors


def _check_codex_prereqs(config: PipelineConfig) -> list[str]:
    errors: list[str] = []
    bin_name = config.llm.codex.bin
    if shutil.which(bin_name) is None:
        errors.append(f"codex binary {bin_name!r} not found on PATH (required for harness=codex)")
    if not Path("~/.codex/auth.json").expanduser().exists():
        errors.append(
            "codex is not logged in — run `codex login` (expected ~/.codex/auth.json to exist)"
        )
    return errors


def _check_pi_prereqs(config: PipelineConfig) -> list[str]:
    errors: list[str] = []
    bin_name = config.llm.pi.bin
    if shutil.which(bin_name) is None:
        errors.append(f"pi binary {bin_name!r} not found on PATH (required for harness=pi)")
    if not Path("~/.pi/agent/auth.json").expanduser().exists():
        errors.append(
            "pi is not logged in — run `pi` and authenticate "
            "(expected ~/.pi/agent/auth.json to exist)"
        )
    return errors


_verified_harnesses: set[HarnessName] = set()


def ensure_harness_prereqs(harness: HarnessName) -> None:
    """Run the prereq check for ``harness``, once per process, or raise.

    Harness openers call this on first invocation so a pipeline with a
    logged-out CLI or a missing ``OPENROUTER_API_KEY`` fails within
    milliseconds of starting the relevant context, instead of crashing
    deep inside the subprocess or the OpenAI SDK.
    """
    if harness in _verified_harnesses:
        return
    config = get_config()
    if harness == "none":
        errors = _check_embedded_prereqs()
    elif harness == "claude-code":
        errors = _check_claude_code_prereqs(config)
    elif harness == "codex":
        errors = _check_codex_prereqs(config)
    elif harness == "pi":
        errors = _check_pi_prereqs(config)
    else:
        assert_never(harness)
    if errors:
        raise RuntimeError(
            f"Harness {harness!r} prereqs not satisfied:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    _verified_harnesses.add(harness)


@lru_cache(maxsize=1)
def configure_jax_persistent_cache() -> None:
    """Acquire the fit process's compilation cache; configuration errors propagate."""
    if os.getenv("NOF1_CAUSAL_LAB_DISABLE_JAX_PERSISTENT_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    import jax

    cache_dir = os.getenv("JAX_COMPILATION_CACHE_DIR") or str(
        Path.home() / ".cache" / "nof1-causal-lab" / "jax"
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if not jax.config.values.get("jax_compilation_cache_dir"):
        jax.config.update("jax_compilation_cache_dir", cache_dir)
