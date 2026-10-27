"""Partial sampler settings supplied by service callers."""

from __future__ import annotations

from pydantic import Field

from grassmoth.public import (
    ConvergenceStoppingSpec,
    InferenceAlgorithmSpec,
    InferenceHardware,
    Value,
)


class FitSettingsSpec(Value):
    """Optional numerical controls applied to the configured joint sampler.

    Each field means what the SamplerSpec field of the same name means. Null keeps the
    server's configured value, or the continued fit's, except convergence stopping,
    which is opt-in for each execution. Profiling also applies only to this execution.
    """

    stop_when_converged: ConvergenceStoppingSpec | None = Field(
        default=None,
        description="Opt in for this execution only; absent/null keeps the fixed schedule, including continuations.",
    )
    profile: bool = Field(
        default=False,
        description="Profile this execution only, including continuations. Time native compiled parts on frozen copies after warmup, or after the first sweep without warmup. Report named production and profiling-only compile requests with cache status; save named compiled analysis beside the cached fit after timing completes.",
    )
    hardware: InferenceHardware | None = None
    time_limit_seconds: int | None = Field(default=None, ge=1)
    algorithm: InferenceAlgorithmSpec | None = None
    num_samples_per_chain: int | None = Field(default=None, ge=0)
    num_warmup: int | None = Field(default=None, ge=0)
    num_chains: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0)
