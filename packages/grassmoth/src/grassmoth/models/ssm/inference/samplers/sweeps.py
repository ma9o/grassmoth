"""Host orchestration for compiled sweeps with one history transfer per block."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from grassmoth.artifacts.immutability import freeze_fields
from grassmoth.compute.batching import compiled_memory
from grassmoth.compute.budget import NO_MEMORY_DEMAND, MemoryDemand
from grassmoth.models.ssm.inference.samplers.progress import (
    inference_phase,
    log_inference_event,
)
from grassmoth.models.ssm.inference.samplers.sweep_plan import (
    SWEEP_BLOCK_SIZE,
    CompiledSweepPhase,
    EmptySweepPhase,
    SweepPlan,
    sweep_block,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from concurrent.futures import Future

    from jax.stages import Compiled

    from grassmoth.artifacts.reports.profiling import FitProfileReport
    from grassmoth.artifacts.sampling import SamplerSpec
    from grassmoth.compute.context import ModelExecutionContext
    from grassmoth.models.ssm.inference.samplers.sweep_plan import SweepProgram
    from grassmoth.models.ssm.inference.targets.particle import ParticleTarget


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
    retained_draws: int
    profile: FitProfileReport | None = None


def compile_sweeps[Carry](
    program: SweepProgram[Carry],
    initial: Carry,
    *,
    execution_context: ModelExecutionContext,
    sampler: SamplerSpec,
    additional_overhead: MemoryDemand = NO_MEMORY_DEMAND,
) -> SweepPlan:
    """Check compiled peak memory plus histories, without timing or executing a sweep.

    Independent chains remain vectorized. The common cap of 25 sequential sweeps
    bounds output buffers and host progress latency.
    """
    block_size = SWEEP_BLOCK_SIZE
    num_warmup, num_samples = sampler.num_warmup, sampler.num_samples_per_chain
    compile_program, arguments, (_, block_shapes), _ = execution_context.prepare_program(
        sweep_block(program.step, block_size),
        initial,
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.bool_),
    )
    history_shapes = {
        name: jax.ShapeDtypeStruct(shape.shape[1:], shape.dtype)
        for name, shape in block_shapes.items()
        if name not in program.excluded_history
    }
    warmup_shapes = {
        name: shape for name, shape in history_shapes.items() if name not in {"positions", "paths"}
    }

    history_bytes = sum(
        steps * sum(shape.size * shape.dtype.itemsize for shape in shapes.values())
        for steps, shapes in ((num_warmup, warmup_shapes), (num_samples, history_shapes))
    )
    combined_diagnostic_bytes = (num_warmup + num_samples) * sum(
        shape.size * shape.dtype.itemsize for shape in warmup_shapes.values()
    )
    transfer_bytes = block_size * sum(
        shape.size * shape.dtype.itemsize for shape in history_shapes.values()
    )
    # Host assembly buffers coexist with one device_get block, including its
    # padded tail and the host-only combined diagnostics. XLA accounts for
    # device block buffers; returned phases have one device owner. Posterior
    # paths and positions are never concatenated with warmup.
    overhead = MemoryDemand(
        host_bytes=history_bytes
        + combined_diagnostic_bytes
        + transfer_bytes
        + additional_overhead.host_bytes,
        device_bytes=history_bytes + additional_overhead.device_bytes,
    )
    executable = compile_program()
    memory = compiled_memory(executable)
    if not execution_context.budget.planning_capacity(memory, limit=1, overhead=overhead):
        raise MemoryError("One sweep lane exceeds the memory budget")
    if not execution_context.budget.capacity(memory, limit=1, overhead=overhead):
        raise MemoryError("Compiled sweep exceeds available memory")
    return SweepPlan(
        CompiledSweepPhase(num_warmup, executable, warmup_shapes, arguments)
        if num_warmup or num_warmup + num_samples == 0
        else EmptySweepPhase(warmup_shapes),
        CompiledSweepPhase(num_samples, executable, history_shapes, arguments)
        if num_samples or num_warmup + num_samples == 0
        else EmptySweepPhase(history_shapes),
        executable,
        sampler=sampler.algorithm.method,
        memory=MemoryDemand(
            host_bytes=memory.host_bytes + overhead.host_bytes,
            device_bytes=memory.device_bytes + overhead.device_bytes,
        ),
    )


def compile_projected_sweeps[Carry](
    program: SweepProgram[Carry],
    initial: Carry,
    *,
    execution_context: ModelExecutionContext,
    sampler: SamplerSpec,
    target: ParticleTarget,
) -> SweepPlan:
    """Compile independent exact observation projection while the sampler compiles."""
    block_size = SWEEP_BLOCK_SIZE

    def observation_projection(position: jax.Array, path: jax.Array) -> jax.Array:
        return jnp.asarray(
            target.observation_log_probs(
                target.context(position, target.times), path, target.observations
            )
        )

    positions = jax.ShapeDtypeStruct(
        (block_size, sampler.num_chains, target.initial_position.size),
        target.initial_position.dtype,
    )
    paths = jax.ShapeDtypeStruct(
        (block_size, sampler.num_chains, len(target.times), target.independent_mask.shape[-1]),
        target.observations.dtype,
    )
    compile_program, projection_arguments, shape, estimate = execution_context.prepare_program(
        jax.vmap(jax.vmap(observation_projection)), positions, paths
    )
    transfer = shape.size * shape.dtype.itemsize
    retained = transfer // block_size * sampler.num_samples_per_chain
    overhead = MemoryDemand(
        host_bytes=estimate.host_bytes + retained + transfer,
        device_bytes=estimate.device_bytes + retained,
    )

    def compile_projection() -> Compiled:
        with inference_phase(
            sampler.algorithm.method,
            "compiling_observation_projection",
            execution_context=execution_context,
        ):
            return compile_program()

    with ThreadPoolExecutor(max_workers=1) as executor:
        context = copy_context()
        future: Future[Compiled] = executor.submit(lambda: context.run(compile_projection))
        plan = compile_sweeps(
            program,
            initial,
            execution_context=execution_context,
            sampler=sampler,
            additional_overhead=overhead,
        )
        projection = future.result()
    actual = compiled_memory(projection)
    demand = MemoryDemand(
        host_bytes=plan.memory.host_bytes - estimate.host_bytes + actual.host_bytes,
        device_bytes=plan.memory.device_bytes - estimate.device_bytes + actual.device_bytes,
    )
    if not execution_context.budget.capacity(demand, limit=1):
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
    execution_context: ModelExecutionContext,
    profile: Callable[[Carry, CompiledSweepPhase], FitProfileReport] | None = None,
) -> SweepRun[Carry]:
    """Keep only requested phase outputs; padded block slots execute no transition."""
    clock = execution_context.clock
    state: Carry = initial
    started = clock()
    first_step_seconds = 0.0
    completed = 0
    histories: list[dict[str, jax.Array]] = []
    retained_draws = 0
    profiling = None

    def profile_copy(phase: CompiledSweepPhase) -> FitProfileReport:
        assert profile is not None
        with inference_phase(plan.sampler, "profiling", execution_context=execution_context):
            copied = jax.block_until_ready(jax.tree.map(jnp.copy, state))
            return profile(copied, phase)

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
        phase_draws = 0
        if isinstance(phase, CompiledSweepPhase):
            phase_started = clock()
            with inference_phase(
                plan.sampler,
                stage,
                execution_context=execution_context,
                total=phase.steps,
                block_size=SWEEP_BLOCK_SIZE,
            ):
                remaining = phase.steps
                while remaining:
                    count = 1 if completed == 0 else min(SWEEP_BLOCK_SIZE, remaining)
                    transition: tuple[Carry, dict[str, jax.Array]] = phase.executable(
                        phase.arguments,
                        state,
                        jnp.asarray(count),
                        jnp.asarray(stage == "sampling"),
                    )
                    state, block = transition
                    if stage == "sampling" and plan.projection is not None:
                        block = {
                            **block,
                            "observation_log_probs": plan.projection(
                                plan.projection_arguments, block["positions"], block["paths"]
                            ),
                        }
                    host_block = jax.device_get({name: block[name] for name in buffers})
                    start = phase.steps - remaining
                    for name in buffers:
                        buffers[name][:, start : start + count] = host_block[name][:count].swapaxes(
                            0, 1
                        )
                    del transition, block, host_block
                    completed += count
                    remaining -= count
                    phase_draws += count
                    if completed == 1:
                        first_step_seconds = clock() - started
                        if profile is not None and plan.warmup.steps == 0:
                            profiling = profile_copy(phase)
                    log_inference_event(
                        plan.sampler,
                        stage,
                        "progress",
                        execution_context=execution_context,
                        completed=phase.steps - remaining,
                        total=phase.steps,
                        elapsed_seconds=clock() - phase_started,
                    )
            if stage == "warmup" and phase.steps and profile is not None:
                profiling = profile_copy(phase)
        else:
            log_inference_event(
                plan.sampler, stage, "skipped", execution_context=execution_context, total=0
            )
        history = {name: jnp.asarray(value[:, :phase_draws]) for name, value in buffers.items()}
        if stage == "sampling":
            retained_draws = phase_draws
        histories.append(history)
        del buffers
    return SweepRun(
        state,
        SweepHistory(histories[0], histories[1]),
        first_step_seconds,
        clock() - started,
        retained_draws,
        profiling,
    )
