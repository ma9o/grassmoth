"""Compile bounded sweep blocks and select dSMC concurrency from buffer estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Concatenate, cast

import equinox as eqx
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


def runtime_program[**P, Result](
    function: Callable[P, Result], *example_args: object, **example_kwargs: object
) -> tuple[Callable[Concatenate[tuple[jax.Array, ...], P], Result], tuple[jax.Array, ...]]:
    """Lift captured numerical arrays at the shared compiled-program boundary.

    Closure conversion traces through nested model/kernel functions, so their
    observations, priors, pilot geometry and tuning become runtime operands too.
    Nested factor JITs must inline their graphs to expose their captured arrays.
    Shapes, laws and control flow remain part of the compiled program.
    """
    converted = eqx.filter_closure_convert(function, *example_args, **example_kwargs)
    dynamic, static = eqx.partition(converted, eqx.is_array)
    leaves, structure = jax.tree.flatten(dynamic)

    def invoke(values: tuple[jax.Array, ...], *args: P.args, **kwargs: P.kwargs) -> Result:
        bound = cast(
            "Callable[P, Result]", eqx.combine(jax.tree.unflatten(structure, values), static)
        )
        return bound(*args, **kwargs)

    return invoke, tuple(leaves)


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
    arguments: tuple[jax.Array, ...]

    def __post_init__(self) -> None:
        """Freeze the owned output-shape mapping."""
        freeze_fields(self)


type SweepPhase = EmptySweepPhase | CompiledSweepPhase


@dataclass(frozen=True)
class _TracedSweepPhase:
    """Abstract phase graph ready for sizing or final compilation."""

    traced: Traced
    shapes: Mapping[str, jax.ShapeDtypeStruct]
    arguments: tuple[jax.Array, ...]

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
    projection_arguments: tuple[jax.Array, ...] = ()


def compile_sweeps[Carry](
    step_factory: Callable[[int], Callable[[Carry, jax.Array], tuple[Carry, dict[str, jax.Array]]]],
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

    def trace_phase(batch: int) -> _TracedSweepPhase:
        phase = jax.ShapeDtypeStruct((), jnp.bool_)
        invoke, arguments = runtime_program(step_factory(batch), initial, phase)
        step = jax.jit(invoke)
        _, shapes = jax.eval_shape(step, arguments, initial, phase)
        zeros = jax.tree.map(lambda shape: jnp.zeros(shape.shape, shape.dtype), shapes)

        def block(
            values: tuple[jax.Array, ...], state: Carry, count: jax.Array, posterior: jax.Array
        ) -> tuple[Carry, dict[str, jax.Array]]:
            def advance(carry: Carry, offset: jax.Array) -> tuple[Carry, dict[str, jax.Array]]:
                result: tuple[Carry, dict[str, jax.Array]] = jax.lax.cond(
                    offset < count,
                    lambda current: step(values, current, posterior),
                    lambda current: (current, zeros),
                    carry,
                )
                return result

            return jax.lax.scan(advance, state, jnp.arange(block_size))

        traced = jax.jit(block).trace(
            arguments, initial, jax.ShapeDtypeStruct((), jnp.int32), phase
        )
        return _TracedSweepPhase(traced, shapes, arguments)

    single = trace_phase(1)
    warmup_shapes = {
        name: shape for name, shape in single.shapes.items() if name not in {"positions", "paths"}
    }

    def compile_plan(batch: int) -> tuple[SweepPlan, MemoryDemand]:
        traced = single if batch == 1 else trace_phase(batch)
        executable = traced.traced.lower().compile()
        warmup = (
            CompiledSweepPhase(num_warmup, executable, warmup_shapes, traced.arguments)
            if num_warmup or num_warmup + num_samples == 0
            else EmptySweepPhase(warmup_shapes)
        )
        posterior = (
            CompiledSweepPhase(num_samples, executable, traced.shapes, traced.arguments)
            if num_samples or num_warmup + num_samples == 0
            else EmptySweepPhase(traced.shapes)
        )
        return SweepPlan(warmup, posterior, block_size, batch, executable), compiled_memory(
            executable
        )

    single_memory = traced_memory(single.traced.jaxpr)
    history_bytes = sum(
        steps * sum(shape.size * shape.dtype.itemsize for shape in shapes.values())
        for steps, shapes in ((num_warmup, warmup_shapes), (num_samples, single.shapes))
    )
    transfer_bytes = block_size * sum(
        shape.size * shape.dtype.itemsize for shape in single.shapes.values()
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
