"""Resolve one launch count from analytical sampling memory before compute dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.public import (
    ComputeBudget,
    MarginalParticleGibbsSpec,
    MemoryDemand,
    ObservationDataset,
    ParticleMgradMemoryFailure,
    SamplerLaunchSpec,
    SamplerSpec,
    StructuralSelection,
    estimate_particle_mgrad_memory,
    particle_mgrad_memory_shape,
)
from grassmoth_server.study.records import Rejected

if TYPE_CHECKING:
    from grassmoth_server.actions.fit.selection import FitContinuation


def _continued_sampler(
    sampler: SamplerSpec, previous: SamplerSpec | None
) -> SamplerSpec | Rejected:
    """Apply the numerical continuation identity rule to the final resolved kernel."""
    if (
        previous is not None
        and sampler.kernel_identity != previous.kernel_identity
        and sampler.num_warmup == 0
    ):
        return Rejected(
            code="CONTINUATION_REQUIRES_WARMUP",
            subject="fit",
            detail="Changing the sampler kernel requires positive warmup",
        )
    return sampler


def admit_sampler(
    launch: SamplerLaunchSpec,
    selection: StructuralSelection,
    observations: ObservationDataset,
    *,
    budget: ComputeBudget,
    previous: FitContinuation | None = None,
) -> SamplerSpec | Rejected:
    """Size one free integer axis; every candidate is arithmetic, never a trial fit.

    N and M candidates follow the numerical owner's power-of-two and bank support
    rules. Chain counts are integers. Memory demand is monotone in each count;
    exponential bracketing and integer bisection select the largest admissible C.
    Admission uses sampling demand only; initialization and compilation do not
    determine launch counts or reject fixed counts at this planning boundary.
    Static non-mGRAD methods retain their existing admission path.
    """
    axis = launch.automatic_dimension
    previous_settings = None if previous is None else previous.settings
    if previous is not None and axis == "chains":
        return Rejected(
            code="INCOMPATIBLE_CONTINUATION",
            subject="fit",
            detail="A continuation fixes its chain count; inherit or pin that count instead of auto.",
        )
    initial_count = 1 if axis == "chains" else 2
    candidate = launch.resolve(initial_count)
    if previous is not None and (
        candidate.num_chains != previous.settings.num_chains
        or candidate.seed != previous.settings.seed
    ):
        return Rejected(
            code="INCOMPATIBLE_CONTINUATION",
            subject="fit",
            detail="Continuation inherits its chain count and random stream",
        )
    if (unsupported := candidate.support_failure) is not None:
        return Rejected(code="UNSUPPORTED_PATH_UPDATE", subject="fit", detail=unsupported.message)
    algorithm = candidate.algorithm
    if not (
        isinstance(algorithm, MarginalParticleGibbsSpec)
        and algorithm.path_update.method.kind == "particle_mgrad"
    ):
        return (
            _continued_sampler(candidate, previous_settings)
            if axis is None
            else Rejected(
                code="FIT_CAPACITY_UNAVAILABLE",
                subject="fit",
                detail="Automatic counts require MPGibbs with Particle-mGRAD.",
            )
        )
    retained_host_bytes = 0 if previous is None else previous.retained_host_bytes
    budget = budget.revised(
        occupied=MemoryDemand(
            host_bytes=budget.occupied.host_bytes + retained_host_bytes,
            device_bytes=budget.occupied.device_bytes,
        )
    )
    shape = particle_mgrad_memory_shape(selection, observations)
    if isinstance(shape, ParticleMgradMemoryFailure):
        return Rejected(code="FIT_CAPACITY_UNAVAILABLE", subject="fit", detail=shape.message)

    def fits(count: int) -> bool:
        proposed = launch.resolve(count)
        if proposed.support_failure is not None:
            return False
        report = estimate_particle_mgrad_memory(shape, proposed)
        return not isinstance(report, ParticleMgradMemoryFailure) and bool(
            budget.capacity(report.sampling, limit=1)
        )

    report = estimate_particle_mgrad_memory(shape, candidate)
    if isinstance(report, ParticleMgradMemoryFailure):
        return Rejected(code="FIT_CAPACITY_UNAVAILABLE", subject="fit", detail=report.message)
    if not budget.capacity(report.sampling, limit=1):
        return Rejected(
            code="FIT_CAPACITY_EXCEEDED",
            subject="fit",
            detail=(
                "The requested fixed counts"
                if axis is None
                else "The minimum permitted automatic count"
            )
            + " exceed the analytical sampling memory budget: "
            f"host={report.sampling.host_bytes} bytes, device={report.sampling.device_bytes} bytes. "
            f"Previous-result host reserve={retained_host_bytes} bytes. "
            "Reduce fixed counts or retained draws, or select hardware with sufficient memory.",
        )
    if axis is not None:
        low, high = initial_count, initial_count * 2
        while fits(high):
            low, high = high, high * 2
        if axis == "chains":
            while high - low > 1:
                middle = (low + high) // 2
                if fits(middle):
                    low = middle
                else:
                    high = middle
        candidate = launch.resolve(low)
    return _continued_sampler(candidate, previous_settings)
