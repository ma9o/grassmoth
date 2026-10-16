"""Host orchestration for compiled sweeps with one history transfer per block."""

from __future__ import annotations

from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.immutability import freeze_fields
from nof1_causal_lab.compute.batching import compiled_memory
from nof1_causal_lab.compute.budget import MemoryDemand
from nof1_causal_lab.compute.traced_memory import traced_memory
from nof1_causal_lab.models.ssm.inference.samplers.progress import (
    inference_phase,
    log_inference_event,
)
from nof1_causal_lab.models.ssm.inference.samplers.sweep_plan import CompiledSweepPhase, compile_sweeps

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from nof1_causal_lab.models.ssm.inference.samplers.sweep_plan import SweepPlan, SweepStage
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget
    from nof1_causal_lab.compute.budget import ComputeBudget
    from jax.stages import Compiled
    from jax.typing import DTypeLike


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


def compile_projected_sweeps[Carry](
    step_factory: Callable[[int, SweepStage], Callable[[Carry], tuple[Carry, dict[str, jax.Array]]]],
    initial: Carry,
    *, target: ParticleTarget, path_dtype: DTypeLike, num_chains: int, num_warmup: int, num_samples: int,
    max_batch_size: int, budget: ComputeBudget, clock: Callable[[], float], label: str,
) -> SweepPlan:
    """Compile independent exact observation projection while the sampler compiles."""
    block_size = min(25, max(1, num_warmup, num_samples))

    def project(position: jax.Array, path: jax.Array) -> jax.Array:
        return target.observation_log_probs(target.context(position, target.times), path, target.observations)

    positions = jax.ShapeDtypeStruct((block_size, num_chains, target.initial_position.size), target.initial_position.dtype)
    paths = jax.ShapeDtypeStruct((block_size, num_chains, len(target.times), target.independent_mask.shape[-1]), path_dtype)
    projection_trace = jax.jit(jax.vmap(jax.vmap(project))).trace(positions, paths)
    estimate = traced_memory(projection_trace.jaxpr)
    shape = projection_trace.out_info
    transfer = shape.size * shape.dtype.itemsize
    retained = transfer // block_size * num_samples
    overhead = MemoryDemand(host_bytes=estimate.host_bytes + retained + transfer,
                            device_bytes=estimate.device_bytes + 2 * retained)

    def compile_projection() -> Compiled:
        with inference_phase(label, "compiling_observation_projection", clock=clock):
            return projection_trace.lower().compile()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(copy_context().run, compile_projection)
        plan = compile_sweeps(step_factory, initial, num_warmup=num_warmup, num_samples=num_samples,
            max_batch_size=max_batch_size, budget=budget, additional_overhead=overhead)
        projection = future.result()
    actual = compiled_memory(projection)
    demand = MemoryDemand(host_bytes=plan.memory.host_bytes - estimate.host_bytes + actual.host_bytes,
                          device_bytes=plan.memory.device_bytes - estimate.device_bytes + actual.device_bytes)
    if not budget.capacity(demand, limit=1):
        raise MemoryError("Sweep and observation projection exceed the numerical execution memory budget")
    return replace(plan, projection=projection, memory=demand)


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
            shapes["observation_log_probs"] = jax.ShapeDtypeStruct(projected_shape.shape[1:], projected_shape.dtype)
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
                        state, jnp.asarray(count)
                    )
                    state, block = transition
                    if stage == "sampling" and plan.projection is not None:
                        block = {**block, "observation_log_probs": plan.projection(block["positions"], block["paths"])}
                    host_block = jax.device_get(block)
                    start = phase.steps - remaining
                    for name in host_block:
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
