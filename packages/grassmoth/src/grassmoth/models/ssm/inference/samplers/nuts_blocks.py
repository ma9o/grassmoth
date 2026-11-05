"""Coordinate blocks for one Hamiltonian transition, optionally followed by a path kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from grassmoth.models.ssm.inference.joint_coordinates import JointChart, JointCoordinates
    from grassmoth.models.ssm.inference.samplers.particle.conditional import (
        ConditionalPathKernel,
        ConditionalPathTuning,
    )


class JointBlockState(NamedTuple):
    """The joint Hamiltonian block has no held path coordinates or path tuning."""


class ConditionalBlockState(NamedTuple):
    """Native path coordinates held by NUTS, and the path kernel's frozen tuning."""

    coordinates: jax.Array
    tuning: ConditionalPathTuning


type NutsBlockState = JointBlockState | ConditionalBlockState


@dataclass(frozen=True)
class JointNutsBlock:
    """Move all coordinates under the exact joint density."""

    coordinates: JointCoordinates

    def initialize(self, point: jax.Array) -> tuple[jax.Array, JointBlockState]:
        """Retain the complete joint vector as the Hamiltonian state."""
        return point, JointBlockState()

    def join(self, point: jax.Array, _state: NutsBlockState) -> jax.Array:
        """All free coordinates already belong to the moved block."""
        return point

    def log_density(self, point: jax.Array, _state: NutsBlockState, chart: JointChart) -> jax.Array:
        """Bind the numeric chart and score the full joint vector."""
        return self.coordinates.bind(chart).log_density(point)


@dataclass(frozen=True)
class ConditionalNutsBlock:
    """Move parameters at fixed native path coordinates, then update the physical path.

    Alternating exact kernels in different invertible charts leaves the same
    joint measure invariant. Refreshing the HMC score after the path update is
    essential: the next leapfrog must see the new conditional target.
    """

    coordinates: JointCoordinates
    smoother: ConditionalPathKernel

    def initialize(self, point: jax.Array) -> tuple[jax.Array, ConditionalBlockState]:
        """Split parameters from the path and initialize the shared smoother."""
        parameters, path = self.coordinates.decode(point)
        dimension = self.coordinates.parameter_dimension
        return (
            point[:dimension] if dimension else jnp.zeros((1,), dtype=point.dtype),
            ConditionalBlockState(point[dimension:], self.smoother.initialize(parameters, path)),
        )

    def join(self, point: jax.Array, state: NutsBlockState) -> jax.Array:
        """Restore the full joint vector for exact density and output projection."""
        assert isinstance(state, ConditionalBlockState)
        return jnp.concatenate((point[: self.coordinates.parameter_dimension], state.coordinates))

    def log_density(self, point: jax.Array, state: NutsBlockState, chart: JointChart) -> jax.Array:
        """Score parameters at fixed states, innovations, or their configured mixture."""
        density = self.coordinates.bind(chart).log_density(self.join(point, state))
        return (
            density if self.coordinates.parameter_dimension else density - 0.5 * jnp.sum(point**2)
        )

    def smooth(
        self,
        point: jax.Array,
        state: NutsBlockState,
        chart: JointChart,
        key: jax.Array,
        iteration: jax.Array,
    ) -> tuple[ConditionalBlockState, dict[str, jax.Array]]:
        """Apply the selected exact path update, then return to the same native chart."""
        assert isinstance(state, ConditionalBlockState)
        coordinates = self.coordinates.bind(chart)
        parameters, path = coordinates.decode(self.join(point, state))
        updated, tuning, info = self.smoother.step(parameters, path, state.tuning, key, iteration)
        encoded = coordinates.encode(parameters, updated)
        # A rejected path move must preserve its native coordinates bit-for-bit.
        held = jnp.where(
            jnp.all(updated == path), state.coordinates, encoded[coordinates.parameter_dimension :]
        )
        return ConditionalBlockState(held, tuning), info


type NutsBlock = JointNutsBlock | ConditionalNutsBlock
