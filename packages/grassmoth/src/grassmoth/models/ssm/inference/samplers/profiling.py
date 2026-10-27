"""Native compiled-operation measurements using the caller's injected clock."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax

from grassmoth.artifacts.reports.profiling import TimingSpread

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable, Sequence

    from grassmoth.artifacts.reports.profiling import FitProfileReport
    from grassmoth.models.ssm.inference.samplers.sweep_plan import CompiledSweepPhase

REPETITIONS = 10


def log_profile(report: FitProfileReport, logger: logging.Logger) -> None:
    """Publish the timing table in the fit's existing execution log as well as its result."""
    logger.info(
        "Fit profile: %s; repetitions=%d chains=%d time_steps=%d; device=%s compile=%.3fs",
        report.phase,
        report.repetitions,
        report.chains,
        report.time_steps,
        ", ".join(report.device),
        report.compile_seconds,
    )
    for group, entries in (
        ("production", report.compilation),
        ("profiling", report.profiling_compilation),
    ):
        for entry in entries:
            logger.info("Profile compile %s/%s: %.3fs", group, entry.name, entry.seconds)
            if entry.persistent_cache != "unobserved":
                logger.info("Profile cache %s: %s", entry.name, entry.persistent_cache)
    if report.particle_dimensions is not None:
        dimensions = report.particle_dimensions
        logger.info(
            "Profile particles=%d labels=%d replicas=%d",
            dimensions.particles,
            dimensions.labels,
            dimensions.replicas,
        )
    for row in report.components:
        logger.info(
            "Profile %s: median=%.3f ms range=[%.3f, %.3f] ms",
            row.component,
            row.timing.median_ms,
            row.timing.min_ms,
            row.timing.max_ms,
        )
    if report.split_total is not None:
        logger.info(
            "Profile split total: median=%.3f ms range=[%.3f, %.3f] ms",
            report.split_total.median_ms,
            report.split_total.min_ms,
            report.split_total.max_ms,
        )
    if report.nuts is not None:
        logger.info(
            "Profile NUTS: leapfrogs=%s step_size=%s tree_depth=%s gradient_count_representative=%s",
            report.nuts.leapfrog_steps,
            report.nuts.step_size,
            report.nuts.tree_depth,
            report.nuts.gradient_count_representative,
        )


def measure_sweep(
    state: object, phase: CompiledSweepPhase, clock: Callable[[], float]
) -> tuple[float, ...]:
    """Use the production block executable, with one active frozen-tuning sweep."""
    import jax.numpy as jnp

    count, posterior = jnp.asarray(1), jnp.asarray(True)
    durations, _ = measure(
        lambda: phase.executable(phase.arguments, state, count, posterior), clock
    )
    return durations


def measure[Result](
    operation: Callable[[], Result], clock: Callable[[], float]
) -> tuple[tuple[float, ...], Result]:
    """Prime once, then synchronize every repetition; outputs never feed the next call."""
    result = operation()
    jax.block_until_ready(result)
    durations: list[float] = []
    for _ in range(REPETITIONS):
        started = clock()
        result = operation()
        jax.block_until_ready(result)
        durations.append(clock() - started)
    return tuple(durations), result


def sum_timings(parts: Sequence[tuple[float, ...]]) -> TimingSpread:
    """Keep paired repetitions when comparing split operations with the full sweep."""
    return TimingSpread.from_seconds(tuple(sum(sample) for sample in zip(*parts, strict=True)))


def compilation_seconds(intervals: Sequence[tuple[float, float]]) -> float:
    """Union of explicit compiler intervals, counting concurrent work once."""
    total = end = 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total
