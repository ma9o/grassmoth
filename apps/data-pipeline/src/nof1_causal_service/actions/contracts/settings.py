"""Partial sampler settings supplied by service callers."""

from __future__ import annotations

from pydantic import Field

from nof1_causal_lab.public import InferenceAlgorithmSpec, InferenceHardware, Value


class FitSettingsSpec(Value):
    """Optional numerical controls applied to the configured joint sampler."""

    hardware: InferenceHardware | None = None
    time_limit_seconds: int | None = Field(default=None, ge=1)
    algorithm: InferenceAlgorithmSpec | None = None
    num_samples_per_chain: int | None = Field(default=None, ge=0)
    num_warmup: int | None = Field(default=None, ge=0)
    num_chains: int | None = Field(default=None, ge=1)
    num_particles: int | None = Field(default=None, ge=2)
    seed: int | None = Field(default=None, ge=0)
