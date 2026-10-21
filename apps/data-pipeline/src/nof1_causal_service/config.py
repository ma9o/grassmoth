"""Configuration loader for the N-of-1 Causal Lab pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import ConfigDict, TypeAdapter, with_config

from nof1_causal_lab.artifacts.sampling import SamplerSpec
from nof1_causal_lab.llm.specs import EmbeddedLLMSpec, EmbeddedReasoningEffort

# Centralized .env loading — all modules that need env vars import from config.py
# (or from modules that import config.py), so this runs once at import time.
load_dotenv(Path(__file__).resolve().parents[4] / ".env")


# ---------------------------------------------------------------------------
# LLM backend defaults (global)
# ---------------------------------------------------------------------------


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class EmbeddedLLMDefaults:
    """Defaults for OpenRouter contexts."""

    max_tokens: int = 65536
    timeout: int = 900
    reasoning_effort: EmbeddedReasoningEffort = "xhigh"


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class LLMDefaults:
    """Global LLM defaults."""

    embedded: EmbeddedLLMDefaults = field(default_factory=EmbeddedLLMDefaults)


# ---------------------------------------------------------------------------
# Agentic context configs
# ---------------------------------------------------------------------------


@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class ExtractionWorkersConfig:
    """extraction: Support-Window Extraction (Workers).

    ``max_rpm`` bounds OpenRouter requests per minute across all workers.
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


def validate_config(config: PipelineConfig) -> list[str]:
    """Check model naming after field validation at the YAML boundary."""
    model = config.extraction_workers.llm.model
    if model.startswith("openrouter/"):
        return []
    return [f"extraction_workers.llm.model: {model!r} should start with 'openrouter/'"]


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
