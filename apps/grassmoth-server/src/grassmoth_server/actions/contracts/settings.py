"""Partial sampler settings supplied by service callers."""

from __future__ import annotations

from pydantic import Field

from grassmoth.public import InferenceAlgorithmSpec, InferenceHardware, Value


class FitSettingsSpec(Value):
    """Optional numerical controls applied to the configured joint sampler.

    Each field means what the SamplerSpec field of the same name means. Null keeps the
    server's configured value, or the continued fit's.
    """

    hardware: InferenceHardware | None = None
    time_limit_seconds: int | None = Field(default=None, ge=1)
    algorithm: InferenceAlgorithmSpec | None = None
    num_samples_per_chain: int | None = Field(default=None, ge=0)
    num_warmup: int | None = Field(default=None, ge=0)
    num_chains: int | None = Field(default=None, ge=1)
    num_particles: int | None = Field(default=None, ge=2)
    seed: int | None = Field(default=None, ge=0)
