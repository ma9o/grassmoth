"""Immutable sweep plans and pure bounded transition programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from grassmoth.artifacts.immutability import freeze_fields
from grassmoth.compute.budget import NO_MEMORY_DEMAND, MemoryDemand

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jax.stages import Compiled


SWEEP_BLOCK_SIZE = 25


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
class SweepPlan:
    """Warmup and posterior blocks keep chain concurrency fixed throughout the fit."""

    warmup: SweepPhase
    posterior: SweepPhase
    batch_size: int
    compiled_step: Compiled
    memory: MemoryDemand = NO_MEMORY_DEMAND
    projection: Compiled | None = None
    projection_arguments: tuple[jax.Array, ...] = ()


def sweep_block[Carry](
    step: Callable[[Carry, jax.Array], tuple[Carry, dict[str, jax.Array]]],
    block_size: int,
) -> Callable[[Carry, jax.Array, jax.Array], tuple[Carry, dict[str, jax.Array]]]:
    """Build a bounded block; inactive slots neither transition nor consume keys."""

    def block(
        state: Carry, count: jax.Array, posterior: jax.Array
    ) -> tuple[Carry, dict[str, jax.Array]]:
        transition = jax.jit(step, inline=jax.Inline.JAX_EARLY)
        _, shapes = jax.eval_shape(transition, state, posterior)
        zeros = jax.tree.map(lambda shape: jnp.zeros(shape.shape, shape.dtype), shapes)

        def advance(carry: Carry, offset: jax.Array) -> tuple[Carry, dict[str, jax.Array]]:
            result: tuple[Carry, dict[str, jax.Array]] = jax.lax.cond(
                offset < count,
                lambda current: transition(current, posterior),
                lambda current: (current, zeros),
                carry,
            )
            return result

        return jax.lax.scan(advance, state, jnp.arange(block_size))

    return block
