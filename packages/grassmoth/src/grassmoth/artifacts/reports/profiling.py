"""Synchronized timings of native compiled operations on frozen chain copies."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from grassmoth.artifacts.base import Value


class TimingSpread(Value):
    """Median and full observed range of ten synchronized wall-clock repetitions."""

    median_ms: float = Field(ge=0)
    min_ms: float = Field(ge=0)
    max_ms: float = Field(ge=0)

    @classmethod
    def from_seconds(cls, samples: tuple[float, ...]) -> TimingSpread:
        """Summarize measured intervals in milliseconds."""
        values = np.asarray(samples) * 1000
        return cls(
            median_ms=float(np.median(values)),
            min_ms=float(values.min()),
            max_ms=float(values.max()),
        )


class FitComponentTiming(Value):
    """One independently compiled native operation; timing includes synchronization."""

    component: Literal[
        "full_sweep",
        "parameter_move",
        "path_smoother",
        "nuts_companion",
        "mh_companion",
        "conditional_gradient",
    ]
    timing: TimingSpread


class NutsProfileReport(Value):
    """Companion work on the fixed timed state, one entry per chain."""

    leapfrog_steps: tuple[int, ...]
    step_size: tuple[float, ...]
    tree_depth: tuple[int, ...]
    gradient_count_representative: bool = Field(
        description="True only after this execution's warmup; otherwise the tree work is not representative of tuned sampling."
    )


class ParticleProfileDimensions(Value):
    """Particle-kernel work dimensions, including any temperature replicas."""

    particles: int = Field(ge=1)
    labels: int = Field(ge=2)
    replicas: int = Field(ge=1)


class ProgramCompileReport(Value):
    """One named request, from tracing until its executable is ready; excludes dumps."""

    name: str
    seconds: float = Field(ge=0)
    persistent_cache: Literal["unobserved", "disabled", "not_consulted", "miss", "hit"] = Field(
        default="unobserved",
        description="The server observes JAX's persistent-cache events in the compiling thread. Disabled means no configured persistent cache; not_consulted means no lookup event (for example, in-memory reuse). Unobserved is for a library-supplied compiler without this shell instrumentation.",
    )


class FitProfileReport(Value):
    """Wall-clock microbenchmarks on a copied state, never used by inference.

    Every operation is primed once and measured ten times at fixed state, tuning
    and random keys. Full sweep uses the production executable with count one;
    components are compiled separately. The path timing includes recurrence
    replay at the fixed selected context. Conditional gradient is one chain's
    exact conditional density and gradient, including replay. No device-trace
    kernel times are used. Separately compiled parts need not sum to a full sweep.
    """

    phase: Literal["after_warmup", "after_first_sweep", "compile_only"]
    repetitions: int = Field(ge=0)
    compile_seconds: float = Field(
        default=0,
        ge=0,
        description="Union of production compile-request intervals, including tracing, lowering and compilation or cache load. Excludes profiling-only programs and artifact writes.",
    )
    compilation: tuple[ProgramCompileReport, ...] = Field(
        default=(), description="Production compile requests, in request order."
    )
    profiling_compilation: tuple[ProgramCompileReport, ...] = Field(
        default=(), description="Separately compiled timing probes, excluded from compile_seconds."
    )
    device: tuple[str, ...] = ()
    chains: int = Field(ge=1)
    time_steps: int = Field(ge=1)
    particle_dimensions: ParticleProfileDimensions | None = None
    components: tuple[FitComponentTiming, ...] = ()
    split_total: TimingSpread | None = Field(
        default=None,
        description="Sum of parameter, path and companion timings per repetition, summarized alongside full_sweep. Other sweep bookkeeping is not part of this sum.",
    )
    nuts: NutsProfileReport | None = None
