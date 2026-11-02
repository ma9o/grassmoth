"""Centred coordinates of the exact joint parameter and trajectory posterior."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from grassmoth.compute.programs import ProgramCompiler
    from grassmoth.models.ssm.inference.targets.particle import (
        ParticleContext,
        ParticleTarget,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class CentredParameterization:
    """The original frozen affine path scale; independent of the sampled parameters."""

    path_center: jax.Array
    path_chol: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class JointChart:
    """Seed-dependent numerical inputs; the compiled coordinate policy stays fixed."""

    parameter_center: jax.Array
    affine: CentredParameterization


@dataclass(frozen=True, eq=False)
class JointCoordinates:
    """One owned parameterization, with exact constraints excluded from its sampled vector."""

    target: ParticleTarget
    parameter_center: jax.Array
    free_mask: jax.Array
    free_indices: jax.Array
    parameterization: CentredParameterization

    @property
    def chart(self) -> JointChart:
        """Expose the numerical chart independently of masks and model structure."""
        return JointChart(self.parameter_center, self.parameterization)

    def bind(self, chart: JointChart) -> JointCoordinates:
        """Bind a fit's numerical inputs without changing its coordinate policy."""
        return replace(self, parameter_center=chart.parameter_center, parameterization=chart.affine)

    @property
    def parameter_dimension(self) -> int:
        """Number of unconstrained scientific parameter coordinates."""
        return self.parameter_center.size

    @property
    def dimension(self) -> int:
        """Parameters and free path coordinates, excluding exact recurrences."""
        return self.parameter_dimension + self.free_indices.size

    def _unpack(self, position: jax.Array) -> tuple[jax.Array, jax.Array]:
        values = jnp.zeros(self.free_mask.size, dtype=position.dtype)
        values = values.at[self.free_indices].set(
            position[self.parameter_dimension : self.dimension]
        )
        return self.parameter_center + position[: self.parameter_dimension], values.reshape(
            self.free_mask.shape
        )

    def _path(self, context: ParticleContext, values: jax.Array) -> jax.Array:
        center, chol = self.parameterization.path_center, self.parameterization.path_chol
        return self.target.replay_path(context, center + jnp.einsum("tij,tj->ti", chol, values))

    def decode(self, position: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Recover aligned scientific parameters and their exact-model trajectory."""
        parameters, values = self._unpack(position)
        context = self.target.context(parameters, self.target.times)
        return parameters, self._path(context, values)


def build_joint_coordinates(
    target: ParticleTarget, *, compiler: ProgramCompiler
) -> JointCoordinates:
    """Freeze the affine path scale at the target's initial parameters."""
    position = target.initial_position
    free = target.independent_mask
    if target.exact_constraints is not None:
        free = free & target.exact_constraints.free_mask
    indices = jnp.asarray(np.flatnonzero(np.asarray(free)), dtype=jnp.int32)

    def affine_chart(parameters: jax.Array) -> CentredParameterization:
        context = target.context(parameters, target.times)
        initial_mean, initial_covariance = target.initial_moments(context)
        path = target.replay_path(
            context, jnp.broadcast_to(initial_mean, (target.times.size, initial_mean.size))
        )
        transition_covariance = jax.vmap(
            lambda previous, index: target.transition_moments(context, previous, index)[1]
        )(path[:-1], jnp.arange(1, target.times.size))
        covariance = jnp.concatenate((initial_covariance[None], transition_covariance))
        embedded = jnp.where(free[:, :, None] & free[:, None, :], covariance, 0.0)
        embedded = embedded + jax.vmap(jnp.diag)((~free).astype(path.dtype))
        return CentredParameterization(path, jnp.linalg.cholesky(embedded))

    chart = compiler(affine_chart, position).call(position)
    return JointCoordinates(target, position, free, indices, chart)
