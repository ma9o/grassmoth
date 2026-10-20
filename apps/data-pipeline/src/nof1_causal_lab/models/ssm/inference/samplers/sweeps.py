"""Host orchestration for compiled sweeps with one history transfer per block."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.immutability import freeze_fields
from nof1_causal_lab.compute.batching import compiled_memory
from nof1_causal_lab.compute.budget import NO_MEMORY_DEMAND, MemoryDemand
from nof1_causal_lab.compute.compilation_cache import prepare_program
from nof1_causal_lab.models.ssm.inference.samplers.progress import (
    inference_phase,
    log_inference_event,
)
from nof1_causal_lab.models.ssm.inference.samplers.sweep_plan import (
    CompiledSweepPhase,
    EmptySweepPhase,
    SweepPlan,
    sweep_block,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from concurrent.futures import Future

    from jax.stages import Compiled
    from jax.typing import DTypeLike

    from nof1_causal_lab.compute.budget import ComputeBudget
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget


@dataclass(frozen=True)
class SweepHistory:
    """Immutable chain-major phase outputs, detached from host assembly buffers."""

    warmup: Mapping[str, jax.Array]
    posterior: Mapping[str, jax.Array]

    def __post_init__(self) -> None:
        """Freeze phase mappings while leaving device arrays intact."""
        freeze_fields(self)


@dataclass(frozen=True)
class SweepRun[Carry]:
    """Final native JAX state, histories and actual execution durations."""

    state: Carry
    history: SweepHistory
    first_step_seconds: float
    sampling_loop_seconds: float


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

    Compile the maximum node/root width first and halve it until compiled memory
    fits the budget. Independent chains remain vectorized. The common cap of
    25 sequential sweeps bounds output buffers and host progress latency.
    """
    block_size = min(25, max(1, num_warmup, num_samples))

    def prepare_phase(
        batch: int,
    ) -> tuple[
        Callable[[], Compiled],
        tuple[jax.Array, ...],
        Mapping[str, jax.ShapeDtypeStruct],
    ]:
        compile_program, arguments, (_, block_shapes), _ = prepare_program(
            sweep_block(step_factory(batch), block_size),
            initial,
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.bool_),
        )
        shapes = {
            name: jax.ShapeDtypeStruct(shape.shape[1:], shape.dtype)
            for name, shape in block_shapes.items()
        }
        return compile_program, arguments, shapes

    maximum = prepare_phase(max_batch_size)
    _, _, history_shapes = maximum
    warmup_shapes = {
        name: shape for name, shape in history_shapes.items() if name not in {"positions", "paths"}
    }

    def compile_plan(batch: int) -> tuple[SweepPlan, MemoryDemand]:
        compile_program, arguments, shapes = (
            maximum if batch == max_batch_size else prepare_phase(batch)
        )
        executable = compile_program()
        warmup = (
            CompiledSweepPhase(num_warmup, executable, warmup_shapes, arguments)
            if num_warmup or num_warmup + num_samples == 0
            else EmptySweepPhase(warmup_shapes)
        )
        posterior = (
            CompiledSweepPhase(num_samples, executable, shapes, arguments)
            if num_samples or num_warmup + num_samples == 0
            else EmptySweepPhase(shapes)
        )
        return SweepPlan(warmup, posterior, block_size, batch, executable), compiled_memory(
            executable
        )

    history_bytes = sum(
        steps * sum(shape.size * shape.dtype.itemsize for shape in shapes.values())
        for steps, shapes in ((num_warmup, warmup_shapes), (num_samples, history_shapes))
    )
    transfer_bytes = block_size * sum(
        shape.size * shape.dtype.itemsize for shape in history_shapes.values()
    )
    # Host assembly buffers coexist with one device_get block, including its
    # padded tail. XLA accounts for device block buffers; returned phases and
    # combined diagnostics can also coexist on device. These do not scale with
    # node concurrency and must be reserved only once.
    overhead = MemoryDemand(
        host_bytes=history_bytes + transfer_bytes + additional_overhead.host_bytes,
        device_bytes=2 * history_bytes + additional_overhead.device_bytes,
    )
    batch = max_batch_size
    while True:
        plan, memory = compile_plan(batch)
        if budget.planning_capacity(memory, limit=1, overhead=overhead):
            if not budget.capacity(memory, limit=1, overhead=overhead):
                raise MemoryError("Compiled sweep exceeds available memory")
            return replace(
                plan,
                memory=MemoryDemand(
                    host_bytes=memory.host_bytes + overhead.host_bytes,
                    device_bytes=memory.device_bytes + overhead.device_bytes,
                ),
            )
        if batch == 1:
            raise MemoryError("One sweep lane exceeds the memory budget")
        batch //= 2


def compile_projected_sweeps[Carry](
    step_factory: Callable[[int], Callable[[Carry, jax.Array], tuple[Carry, dict[str, jax.Array]]]],
    initial: Carry,
    *,
    target: ParticleTarget,
    path_dtype: DTypeLike,
    num_chains: int,
    num_warmup: int,
    num_samples: int,
    max_batch_size: int,
    budget: ComputeBudget,
    clock: Callable[[], float],
    label: str,
) -> SweepPlan:
    """Compile independent exact observation projection while the sampler compiles."""
    block_size = min(25, max(1, num_warmup, num_samples))

    def project(position: jax.Array, path: jax.Array) -> jax.Array:
        return jnp.asarray(
            target.observation_log_probs(
                target.context(position, target.times), path, target.observations
            )
        )

    positions = jax.ShapeDtypeStruct(
        (block_size, num_chains, target.initial_position.size), target.initial_position.dtype
    )
    paths = jax.ShapeDtypeStruct(
        (block_size, num_chains, len(target.times), target.independent_mask.shape[-1]), path_dtype
    )
    compile_program, projection_arguments, shape, estimate = prepare_program(
        jax.vmap(jax.vmap(project)), positions, paths
    )
    transfer = shape.size * shape.dtype.itemsize
    retained = transfer // block_size * num_samples
    overhead = MemoryDemand(
        host_bytes=estimate.host_bytes + retained + transfer,
        device_bytes=estimate.device_bytes + 2 * retained,
    )

    def compile_projection() -> Compiled:
        with inference_phase(label, "compiling_observation_projection", clock=clock):
            return compile_program()

    with ThreadPoolExecutor(max_workers=1) as executor:
        context = copy_context()
        future: Future[Compiled] = executor.submit(lambda: context.run(compile_projection))
        plan = compile_sweeps(
            step_factory,
            initial,
            num_warmup=num_warmup,
            num_samples=num_samples,
            max_batch_size=max_batch_size,
            budget=budget,
            additional_overhead=overhead,
        )
        projection = future.result()
    actual = compiled_memory(projection)
    demand = MemoryDemand(
        host_bytes=plan.memory.host_bytes - estimate.host_bytes + actual.host_bytes,
        device_bytes=plan.memory.device_bytes - estimate.device_bytes + actual.device_bytes,
    )
    if not budget.capacity(demand, limit=1):
        raise MemoryError(
            "Sweep and observation projection exceed the numerical execution memory budget"
        )
    return replace(
        plan, projection=projection, projection_arguments=projection_arguments, memory=demand
    )


def run_sweeps[Carry](
    plan: SweepPlan,
    initial: Carry,
    *,
    clock: Callable[[], float],
    label: str,
) -> SweepRun[Carry]:
    """Keep only requested phase outputs; padded block slots execute no transition."""
    state: Carry = initial
    started = clock()
    first_step_seconds = 0.0
    completed = 0
    histories: list[dict[str, jax.Array]] = []
    for stage, phase in (("warmup", plan.warmup), ("sampling", plan.posterior)):
        # Assemble directly in the final chain-major layout. Each live block is
        # copied once; no retained chunks or full-history concatenation coexist.
        shapes = dict(phase.shapes)
        if stage == "sampling" and plan.projection is not None:
            projected_shape = plan.projection.out_info
            shapes["observation_log_probs"] = jax.ShapeDtypeStruct(
                projected_shape.shape[1:], projected_shape.dtype
            )
        buffers = {
            name: np.empty((shape.shape[0], phase.steps, *shape.shape[1:]), dtype=shape.dtype)
            for name, shape in shapes.items()
        }
        if isinstance(phase, CompiledSweepPhase):
            phase_started = clock()
            with inference_phase(
                label,
                stage,
                clock=clock,
                total=phase.steps,
                block_size=plan.block_size,
                particle_batch_size=plan.batch_size,
            ):
                remaining = phase.steps
                while remaining:
                    count = 1 if completed == 0 else min(plan.block_size, remaining)
                    transition: tuple[Carry, dict[str, jax.Array]] = phase.executable(
                        phase.arguments, state, jnp.asarray(count), jnp.asarray(stage == "sampling")
                    )
                    state, block = transition
                    if stage == "sampling" and plan.projection is not None:
                        block = {
                            **block,
                            "observation_log_probs": plan.projection(
                                plan.projection_arguments, block["positions"], block["paths"]
                            ),
                        }
                    host_block = jax.device_get(block)
                    start = phase.steps - remaining
                    for name in buffers:
                        buffers[name][:, start : start + count] = host_block[name][:count].swapaxes(
                            0, 1
                        )
                    del transition, block, host_block
                    completed += count
                    remaining -= count
                    if completed == 1:
                        first_step_seconds = clock() - started
                    log_inference_event(
                        label,
                        stage,
                        "progress",
                        completed=phase.steps - remaining,
                        total=phase.steps,
                        elapsed_seconds=clock() - phase_started,
                    )
        else:
            log_inference_event(label, stage, "skipped", total=0)
        history = {name: jnp.asarray(value) for name, value in buffers.items()}
        histories.append(history)
        del buffers
    return SweepRun(
        state, SweepHistory(histories[0], histories[1]), first_step_seconds, clock() - started
    )
