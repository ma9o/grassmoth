"""Compile bounded sweep blocks and select dSMC concurrency from buffer estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from nof1_causal_lab.artifacts.immutability import freeze_fields
from nof1_causal_lab.compute.batching import compile_from_estimate, compiled_memory
from nof1_causal_lab.compute.budget import NO_MEMORY_DEMAND, MemoryDemand
from nof1_causal_lab.compute.traced_memory import traced_memory

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jax.stages import Compiled, Traced

    from nof1_causal_lab.compute.budget import ComputeBudget


type SweepStage = Literal["warmup", "posterior"]


@dataclass(frozen=True)
class EmptySweepPhase:
    """Output shapes for a phase with no requested transitions."""

    shapes: Mapping[str, jax.ShapeDtypeStruct]

    @property
    def steps(self) -> int:
        """Expose the empty phase length."""
        return 0

    def __post_init__(self) -> None:
        """Freeze the owned output-shape mapping."""
        freeze_fields(self)


@dataclass(frozen=True)
class CompiledSweepPhase:
    """A nonempty phase's executable, output shapes and transition count."""

    steps: int
    executable: Compiled
    shapes: Mapping[str, jax.ShapeDtypeStruct]

    def __post_init__(self) -> None:
        """Freeze the owned output-shape mapping."""
        freeze_fields(self)


type SweepPhase = EmptySweepPhase | CompiledSweepPhase


@dataclass(frozen=True)
class _TracedSweepPhase:
    """Abstract phase graph ready for sizing or final compilation."""

    steps: int
    traced: Traced
    shapes: Mapping[str, jax.ShapeDtypeStruct]

    def __post_init__(self) -> None:
        """Freeze the owned output-shape mapping."""
        freeze_fields(self)


@dataclass(frozen=True)
class SweepPlan:
    """Warmup and posterior blocks keep chain concurrency fixed throughout the fit."""

    warmup: SweepPhase
    posterior: SweepPhase
    block_size: int
    batch_size: int
    compiled_step: Compiled
    memory: MemoryDemand = NO_MEMORY_DEMAND
    projection: Compiled | None = None


def compile_sweeps[Carry](
    step_factory: Callable[
        [int, SweepStage], Callable[[Carry], tuple[Carry, dict[str, jax.Array]]]
    ],
    initial: Carry,
    *,
    num_warmup: int,
    num_samples: int,
    max_batch_size: int,
    budget: ComputeBudget,
    additional_overhead: MemoryDemand = NO_MEMORY_DEMAND,
) -> SweepPlan:
    """Check compiled peak memory plus histories, without timing or executing a sweep.

    Trace one node/root lane and size concurrency arithmetically before compiling
    the selected width. Independent chains remain vectorized. The common cap of
    25 sequential sweeps bounds output buffers and host progress latency.
    """
    block_size = min(25, max(1, num_warmup, num_samples))

    def trace_phase(
        batch: int, stage: SweepStage, steps: int
    ) -> EmptySweepPhase | _TracedSweepPhase:
        # Reuse the step trace when embedding it in the bounded scan below.
        step = jax.jit(step_factory(batch, stage))
        _, shapes = jax.eval_shape(step, initial)
        if steps == 0 and num_warmup + num_samples > 0:
            return EmptySweepPhase(shapes)
        zeros = jax.tree.map(lambda shape: jnp.zeros(shape.shape, shape.dtype), shapes)

        def block(state: Carry, count: jax.Array) -> tuple[Carry, dict[str, jax.Array]]:
            def advance(carry: Carry, offset: jax.Array) -> tuple[Carry, dict[str, jax.Array]]:
                result: tuple[Carry, dict[str, jax.Array]] = jax.lax.cond(
                    offset < count, step, lambda current: (current, zeros), carry
                )
                return result

            return jax.lax.scan(advance, state, jnp.arange(block_size))

        traced = jax.jit(block).trace(initial, jax.ShapeDtypeStruct((), jnp.int32))
        return _TracedSweepPhase(steps, traced, shapes)

    single = (
        trace_phase(1, "warmup", num_warmup),
        trace_phase(1, "posterior", num_samples),
    )

    def compile_phase(phase: EmptySweepPhase | _TracedSweepPhase) -> SweepPhase:
        if isinstance(phase, EmptySweepPhase):
            return phase
        return CompiledSweepPhase(phase.steps, phase.traced.lower().compile(), phase.shapes)

    def compile_plan(batch: int) -> tuple[SweepPlan, MemoryDemand]:
        warmup = compile_phase(
            single[0] if batch == 1 else trace_phase(batch, "warmup", num_warmup)
        )
        posterior = compile_phase(
            single[1] if batch == 1 else trace_phase(batch, "posterior", num_samples)
        )
        phases = (warmup, posterior)
        executables = tuple(p.executable for p in phases if isinstance(p, CompiledSweepPhase))
        memories = tuple(compiled_memory(executable) for executable in executables)
        return SweepPlan(warmup, posterior, block_size, batch, executables[-1]), MemoryDemand(
            host_bytes=max(m.host_bytes for m in memories),
            device_bytes=max(m.device_bytes for m in memories),
        )

    memories = tuple(
        traced_memory(phase.traced.jaxpr)
        for phase in single
        if isinstance(phase, _TracedSweepPhase)
    )
    single_memory = MemoryDemand(
        host_bytes=max(memory.host_bytes for memory in memories),
        device_bytes=max(memory.device_bytes for memory in memories),
    )
    history_bytes = sum(
        p.steps * sum(shape.size * shape.dtype.itemsize for shape in p.shapes.values())
        for p in single
    )
    transfer_bytes = max(
        block_size * sum(shape.size * shape.dtype.itemsize for shape in p.shapes.values())
        for p in single
        if isinstance(p, _TracedSweepPhase)
    )
    # Host assembly buffers coexist with one device_get block, including its
    # padded tail. XLA accounts for device block buffers; returned phases and
    # combined diagnostics can also coexist on device. These do not scale with
    # node concurrency and must be reserved only once.
    overhead = MemoryDemand(
        host_bytes=history_bytes + transfer_bytes + additional_overhead.host_bytes,
        device_bytes=2 * history_bytes + additional_overhead.device_bytes,
    )
    chosen = compile_from_estimate(
        compile_plan,
        single_memory=single_memory,
        max_batch_size=max_batch_size,
        budget=budget,
        overhead=overhead,
    )
    from dataclasses import replace

    return replace(
        chosen.executable,
        memory=MemoryDemand(
            host_bytes=chosen.memory.host_bytes + overhead.host_bytes,
            device_bytes=chosen.memory.device_bytes + overhead.device_bytes,
        ),
    )
