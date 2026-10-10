"""Shared batch selection from compiler memory estimates without executing kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from nof1_causal_lab.compute.budget import MemoryDemand

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
) -> CompiledBatch[Executable]:
    """Estimate a width from one lane, then verify compiled memory before execution.

    Scaling the complete one-lane footprint conservatively includes fixed buffers.
    Compiler fusion can change that scaling, so each proposed width is compiled and
    checked. Halving is a compile-time search; no simulation or OOM retry is run.
    """
    single, single_memory = compile_batch(1)
    capacity = budget.capacity(single_memory, limit=max_batch_size)
    if capacity == 0:
        raise MemoryError("One batch lane exceeds the numerical execution memory budget")
    size = max_batch_size if capacity == max_batch_size else 1 << (capacity.bit_length() - 1)
    while size > 1:
        executable, memory = compile_batch(size)
        if budget.capacity(memory, limit=1):
            return CompiledBatch(executable, size, memory)
        size //= 2
    return CompiledBatch(single, 1, single_memory)
