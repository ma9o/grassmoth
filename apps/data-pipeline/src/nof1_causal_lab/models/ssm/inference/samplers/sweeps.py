"""Host orchestration for compiled sweeps with one history transfer per block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.immutability import freeze_fields
from nof1_causal_lab.models.ssm.inference.samplers.progress import (
    inference_phase,
    log_inference_event,
)
from nof1_causal_lab.models.ssm.inference.samplers.sweep_plan import CompiledSweepPhase

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from nof1_causal_lab.models.ssm.inference.samplers.sweep_plan import SweepPlan


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
        buffers = {
            name: np.empty((shape.shape[0], phase.steps, *shape.shape[1:]), dtype=shape.dtype)
            for name, shape in phase.shapes.items()
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
