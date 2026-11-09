"""Synchronized timings of native compiled operations on frozen chain copies."""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
from pydantic import Field

from grassmoth.artifacts.base import Value

type CompilationCacheStatus = Literal["unobserved", "disabled", "not_consulted", "miss", "hit"]


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
    ]
    timing: TimingSpread


class ParticleProfileDimensions(Value):
    """Particle-kernel work dimensions per chain."""

    particles: int = Field(ge=1)
    labels: int = Field(ge=2)


class ProgramCompileReport(Value):
    """One named request, from tracing until its executable is ready; excludes dumps."""

    name: str
    seconds: float = Field(ge=0)
    persistent_cache: CompilationCacheStatus = Field(
        default="unobserved",
        description="The server observes JAX's persistent-cache events in the compiling thread. Disabled means no configured persistent cache; not_consulted means no lookup event (for example, in-memory reuse). Unobserved is for a library-supplied compiler without this shell instrumentation.",
    )

    def with_cache_status(self, *, persistent_cache: CompilationCacheStatus) -> Self:
        """Return the compilation report with the observed cache status."""
        return self._revised(persistent_cache=persistent_cache)


class FitProfileReport(Value):
    """Wall-clock microbenchmarks on a copied state, never used by inference.

    Every operation is primed once and measured ten times at fixed state, tuning
    and random keys. Full sweep uses the production executable with count one;
    components are compiled separately. The path timing includes recurrence
    replay at the fixed selected context. No device-trace
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
        description="Sum of parameter and path timings per repetition, summarized alongside full_sweep. Other sweep bookkeeping is not part of this sum.",
    )

    def with_compile_timings(
        self,
        *,
        compile_seconds: float,
        compilation: tuple[ProgramCompileReport, ...],
        profiling_compilation: tuple[ProgramCompileReport, ...],
        device: tuple[str, ...],
    ) -> Self:
        """Return the profile with measured compilation timings and devices."""
        return self._revised(
            compile_seconds=compile_seconds,
            compilation=compilation,
            profiling_compilation=profiling_compilation,
            device=device,
        )

    def with_compilation(
        self,
        *,
        compilation: tuple[ProgramCompileReport, ...],
        profiling_compilation: tuple[ProgramCompileReport, ...],
    ) -> Self:
        """Return the profile with compilation reports for both execution phases."""
        return self._revised(compilation=compilation, profiling_compilation=profiling_compilation)
