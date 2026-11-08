"""Resolve one launch count from analytical sampling memory before compute dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.public import (
    ComputeBudget,
    MemoryDemand,
    ObservationDataset,
    ParticleMgradMemoryFailure,
    SamplerLaunchSpec,
    SamplerSpec,
    SharedMemoryBudget,
    StructuralSelection,
    Value,
    estimate_particle_mgrad_memory,
    particle_mgrad_memory_shape,
)
from grassmoth_server.study.records import Rejected

if TYPE_CHECKING:
    from grassmoth_server.actions.fit.selection import FitContinuation


def _memory_size(size: int) -> str:
    for unit, scale in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if size >= scale:
            return f"{size / scale:.2f} {unit}"
    return f"{size} bytes"


def _pool_usage(required: int, available: int) -> str:
    utilization = f"{100 * required / available:.2f}%" if available else "no capacity"
    shortage = f", short by {_memory_size(required - available)}" if required > available else ""
    return (
        f"{_memory_size(required)} required / {_memory_size(available)} available "
        f"({utilization}{shortage})"
    )


def _memory_usage(demand: MemoryDemand, budget: ComputeBudget) -> str:
    memory, occupied = budget.memory, budget.occupied
    if isinstance(memory, SharedMemoryBudget):
        return "shared host/device=" + _pool_usage(
            demand.host_bytes + demand.device_bytes,
            max(0, memory.available_bytes - occupied.host_bytes - occupied.device_bytes),
        )
    return (
        "host="
        + _pool_usage(demand.host_bytes, max(0, memory.host_bytes - occupied.host_bytes))
        + "; device="
        + _pool_usage(demand.device_bytes, max(0, memory.device_bytes - occupied.device_bytes))
    )


class FitAdmission(Value):
    """Resolved sampler and the capacity explanation recorded before dispatch."""

    sampler: SamplerSpec
    detail: str


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
) -> FitAdmission | Rejected:
    """Size one free integer axis; every candidate is arithmetic, never a trial fit.

    N and M candidates follow the numerical owner's power-of-two and bank support
    rules. Chain counts are integers. Memory demand is monotone in each count;
    exponential bracketing and integer bisection select the largest admissible C.
    Admission uses sampling demand only; initialization and compilation do not
    determine launch counts or reject fixed counts at this planning boundary.
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
            f"{_memory_usage(report.sampling, budget)}. "
            f"Occupied host={_memory_size(budget.occupied.host_bytes)}, "
            f"device={_memory_size(budget.occupied.device_bytes)}. "
            "Available budgets already exclude planning headroom and occupied memory. "
            "Reduce fixed counts or retained draws, or select hardware with sufficient memory.",
        )
    count = initial_count
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
        count = low
        candidate = launch.resolve(count)
        report = estimate_particle_mgrad_memory(shape, candidate)
        if isinstance(report, ParticleMgradMemoryFailure):
            return Rejected(code="FIT_CAPACITY_UNAVAILABLE", subject="fit", detail=report.message)
    continued = _continued_sampler(candidate, previous_settings)
    if isinstance(continued, Rejected):
        return continued
    algorithm = launch.algorithm.resolve(count)
    method = launch.algorithm.path_update.method.resolve(count)
    return FitAdmission(
        sampler=continued,
        detail=(
            f"{continued.hardware.value}: N={method.num_particles}, "
            f"M={algorithm.n_parameter_particles}, C={continued.num_chains}. "
            f"Estimated sampling memory: {_memory_usage(report.sampling, budget)}. "
            "Available budgets already exclude planning headroom and occupied memory."
        ),
    )
