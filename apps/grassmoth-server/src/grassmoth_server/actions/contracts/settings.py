"""Partial sampler settings supplied by service callers."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from grassmoth.public import (
    ChainLaunchCount,
    InferenceAlgorithmLaunchSpec,
    InferenceHardware,
    SamplerLaunchSpec,
    Value,
)


class FitSettingsSpec(Value):
    """Optional controls for the configured whole-path MPGibbs Particle-mGRAD sampler.

    Each field means what the SamplerSpec field of the same name means. Null keeps the
    server's configured value, or the continued fit's. Profiling applies only to
    this execution.
    """

    profile: bool = Field(
        default=False,
        description="Profile this execution only, including continuations. Time native compiled parts on frozen copies after warmup, or after the first sweep without warmup. Report named production and profiling-only compile requests with cache status; save named compiled analysis beside the cached fit after timing completes.",
    )
    hardware: InferenceHardware | None = None
    time_limit_seconds: int | None = Field(default=None, ge=1)
    algorithm: InferenceAlgorithmLaunchSpec | None = Field(
        default=None,
        description="Replace the complete configured or continued MPGibbs Particle-mGRAD algorithm; omitted inherits it. Launches use whole-path updates and running-prefix parameter mixtures. Defaults fill unspecified algorithm fields.",
    )
    num_samples_per_chain: int | None = Field(default=None, ge=0)
    num_warmup: int | None = Field(default=None, ge=0)
    num_chains: ChainLaunchCount | None = None
    thinning: bool | None = None
    seed: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def one_automatic_dimension(self) -> Self:
        """Reject two explicit auto requests; merged configuration is checked again."""
        defaults = SamplerLaunchSpec()
        SamplerLaunchSpec(
            algorithm=defaults.algorithm if self.algorithm is None else self.algorithm,
            num_chains=defaults.num_chains if self.num_chains is None else self.num_chains,
        )
        return self
