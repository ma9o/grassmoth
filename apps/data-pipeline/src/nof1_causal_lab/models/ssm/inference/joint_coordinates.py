"""Invertible coordinates for the exact joint parameter and latent-trajectory target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget


@dataclass(frozen=True, eq=False)
class JointCoordinates:
    """An invertible affine chart on independent, unobserved model coordinates."""

    target: ParticleTarget
    parameter_center: jax.Array
    path_center: jax.Array
    path_chol: jax.Array
    free_mask: jax.Array
    free_indices: jax.Array

    @property
    def parameter_dimension(self) -> int:
        """Number of unconstrained scientific parameter coordinates."""
        return self.parameter_center.size

    @property
    def dimension(self) -> int:
        """Number of parameter and free path coordinates, excluding exact recurrences."""
        return self.parameter_dimension + self.free_indices.size

    def _coordinates(self, position: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Recover parameters and affine path coordinates before deterministic replay."""
        innovations = jnp.zeros(self.path_center.size, dtype=position.dtype)
        innovations = innovations.at[self.free_indices].set(
            position[self.parameter_dimension : self.dimension]
        )
        innovations = innovations.reshape(self.path_center.shape)
        parameters = self.parameter_center + position[: self.parameter_dimension]
        coordinates = self.path_center + jnp.einsum("tij,tj->ti", self.path_chol, innovations)
        return parameters, coordinates

    def decode(self, position: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Recover parameters and their matching exact-model trajectory."""
        parameters, coordinates = self._coordinates(position)
        context = self.target.context(parameters, self.target.times)
        path = self.target.replay_path(context, coordinates)
        return parameters, path

    def log_density(self, position: jax.Array) -> jax.Array:
        """Score the exact nonlinear model after the frozen coordinate change."""
        # The exact joint target owns deterministic replay. Decoding here would
        # reconstruct and differentiate through the same trajectory twice.
        parameters, path = self._coordinates(position)
        log_jacobian = jnp.sum(
            jnp.where(
                self.free_mask,
                jnp.log(jnp.diagonal(self.path_chol, axis1=-2, axis2=-1)),
                0.0,
            )
        )
        joint_density = jnp.asarray(
            self.target.log_posterior(parameters, path, self.target.observations, self.target.times)
        )
        auxiliary = position[self.dimension :]
        return (
            joint_density
            + log_jacobian
            - 0.5 * jnp.sum(auxiliary**2)
            - 0.5 * auxiliary.size * jnp.log(2 * jnp.pi)
        )


def build_joint_coordinates(target: ParticleTarget) -> JointCoordinates:
    """Scale free path coordinates by exact initial and transition noise at one start.

    Independent path coordinates start near the initial-state mean, without an
    unconditional forward simulation or an unstable linearized autoregression.
    The target itself reconstructs inputs, static values and deterministic states.
    """
    position = target.initial_position
    context = target.context(position, target.times)
    initial_mean, initial_covariance = target.initial_moments(context)
    path = target.replay_path(
        context, jnp.broadcast_to(initial_mean, (target.times.size, initial_mean.size))
    )
    transition_covariance = jax.vmap(
        lambda previous, index: target.transition_moments(context, previous, index)[1]
    )(path[:-1], jnp.arange(1, target.times.size))
    covariance = jnp.concatenate((initial_covariance[None], transition_covariance))
    free = target.independent_mask
    if target.exact_constraints is not None:
        free = free & target.exact_constraints.free_mask
    embedded = jnp.where(free[:, :, None] & free[:, None, :], covariance, 0.0)
    embedded = embedded + jax.vmap(jnp.diag)((~free).astype(path.dtype))
    indices = jnp.asarray(np.flatnonzero(np.asarray(free)), dtype=jnp.int32)
    return JointCoordinates(target, position, path, jnp.linalg.cholesky(embedded), free, indices)
