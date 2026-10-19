"""Shared batch selection from compiler memory estimates without executing kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from nof1_causal_lab.compute.budget import NO_MEMORY_DEMAND, MemoryDemand

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.stages import Compiled

    from nof1_causal_lab.compute.budget import ComputeBudget


class _MemoryStatistics(Protocol):
    argument_size_in_bytes: int
    output_size_in_bytes: int
    alias_size_in_bytes: int
    temp_size_in_bytes: int
    host_argument_size_in_bytes: int
    host_output_size_in_bytes: int
    host_alias_size_in_bytes: int
    host_temp_size_in_bytes: int


def compiled_memory(compiled: Compiled) -> MemoryDemand:
    """Read the pinned JAX compiler's buffer estimates, including outputs and aliases."""
    raw = compiled.memory_analysis()
    if raw is None:
        raise RuntimeError("The JAX backend does not expose compiled memory estimates")
    stats = cast("_MemoryStatistics", raw)
    return MemoryDemand(
        host_bytes=(
            stats.host_argument_size_in_bytes
            + stats.host_output_size_in_bytes
            + stats.host_temp_size_in_bytes
            - stats.host_alias_size_in_bytes
        ),
        device_bytes=(
            stats.argument_size_in_bytes
            + stats.output_size_in_bytes
            + stats.temp_size_in_bytes
            - stats.alias_size_in_bytes
        ),
    )


@dataclass(frozen=True)
class CompiledBatch[Executable]:
    """One executable and the static batch width that fits the supplied budget."""

    executable: Executable
    batch_size: int
    memory: MemoryDemand


def compile_batched[Executable](
    compile_batch: Callable[[int], tuple[Executable, MemoryDemand]],
    *,
    max_batch_size: int,
    budget: ComputeBudget,
    overhead: MemoryDemand = NO_MEMORY_DEMAND,
) -> CompiledBatch[Executable]:
    """Estimate a width from one lane, then verify compiled memory before execution.

    Scaling the complete one-lane footprint conservatively includes fixed buffers.
    Separately supplied retained buffers are reserved once, independent of width.
    Compiler fusion can change that scaling, so each proposed width is compiled and
    checked. Halving is a compile-time search; no simulation or OOM retry is run.
    """
    single, single_memory = compile_batch(1)
    return compile_from_estimate(
        lambda size: (single, single_memory) if size == 1 else compile_batch(size),
        single_memory=single_memory,
        max_batch_size=max_batch_size,
        budget=budget,
        overhead=overhead,
    )


def compile_from_estimate[Executable](
    compile_batch: Callable[[int], tuple[Executable, MemoryDemand]],
    *,
    single_memory: MemoryDemand,
    max_batch_size: int,
    budget: ComputeBudget,
    overhead: MemoryDemand = NO_MEMORY_DEMAND,
) -> CompiledBatch[Executable]:
    """Size arithmetically, then verify the actual executable before accepting it.

    An estimate can overstate memory, so even an estimated zero capacity gets one
    minimum-width compilation. Only actual compiled demand can reject that lane.
    """
    capacity = budget.planning_capacity(single_memory, limit=max_batch_size, overhead=overhead)
    size = (
        max_batch_size if capacity == max_batch_size else 1 << (max(1, capacity).bit_length() - 1)
    )
    while True:
        executable, memory = compile_batch(size)
        if budget.planning_capacity(memory, limit=1, overhead=overhead):
            if not budget.capacity(memory, limit=1, overhead=overhead):
                raise MemoryError(
                    "The planned batch width exceeds currently available memory; "
                    "release other allocations before running it"
                )
            return CompiledBatch(executable, size, memory)
        if size == 1:
            raise MemoryError("One batch lane exceeds the numerical execution memory budget")
        size //= 2
