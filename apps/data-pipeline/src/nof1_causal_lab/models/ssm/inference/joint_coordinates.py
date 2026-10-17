"""Invertible parameterizations of the same exact joint parameter and trajectory posterior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.joint_parameterization import (
    CentredParameterizationRecord,
    JointParameterizationRecord,
    MixedParameterizationRecord,
    NonCentredParameterizationRecord,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.sampling import JointParameterization
    from nof1_causal_lab.models.ssm.inference.targets.particle import (
        ParticleContext,
        ParticleTarget,
    )


@dataclass(frozen=True, eq=False)
class CentredParameterization:
    """The original frozen affine path scale; independent of the sampled parameters."""

    path_center: jax.Array
    path_chol: jax.Array


@dataclass(frozen=True)
class NonCentredParameterization:
    """Every free stochastic path coordinate is an exact conditional innovation."""


@dataclass(frozen=True, eq=False)
class MixedParameterization:
    """Available direct point readings select centred free coordinates, once per fit."""

    centred_mask: jax.Array


@dataclass(frozen=True, eq=False)
class JointCoordinates:
    """One owned parameterization, with exact constraints excluded from its sampled vector."""

    target: ParticleTarget
    parameter_center: jax.Array
    free_mask: jax.Array
    free_indices: jax.Array
    parameterization: CentredParameterization | NonCentredParameterization | MixedParameterization

    @property
    def record(self) -> JointParameterizationRecord:
        """Freeze native-coordinate provenance outside numerical tracing."""
        match self.parameterization:
            case CentredParameterization():
                return CentredParameterizationRecord()
            case NonCentredParameterization():
                return NonCentredParameterizationRecord()
            case MixedParameterization(centred_mask=mask):
                return MixedParameterizationRecord(
                    centred_mask=tuple(
                        tuple(bool(value) for value in row) for row in np.asarray(mask)
                    )
                )

    @property
    def parameter_dimension(self) -> int:
        """Number of unconstrained scientific parameter coordinates."""
        return self.parameter_center.size

    @property
    def dimension(self) -> int:
        """Parameters and free path coordinates, excluding exact recurrences."""
        return self.parameter_dimension + self.free_indices.size

    @property
    def innovation_mask(self) -> jax.Array:
        """Stochastic coordinates transformed under this frozen policy."""
        match self.parameterization:
            case CentredParameterization():
                return jnp.zeros_like(self.free_mask)
            case NonCentredParameterization():
                return self.free_mask
            case MixedParameterization(centred_mask=mask):
                return self.free_mask & ~mask

    def _unpack(self, position: jax.Array) -> tuple[jax.Array, jax.Array]:
        values = jnp.zeros(self.free_mask.size, dtype=position.dtype)
        values = values.at[self.free_indices].set(
            position[self.parameter_dimension : self.dimension]
        )
        return self.parameter_center + position[: self.parameter_dimension], values.reshape(
            self.free_mask.shape
        )

    def _path(self, context: ParticleContext, values: jax.Array) -> tuple[jax.Array, jax.Array]:
        match self.parameterization:
            case CentredParameterization(path_center=center, path_chol=chol):
                path = center + jnp.einsum("tij,tj->ti", chol, values)
                log_jacobian = jnp.sum(
                    jnp.where(self.free_mask, jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), 0.0)
                )
                return self.target.replay_path(context, path), log_jacobian
            case NonCentredParameterization() | MixedParameterization():
                return self.target.innovation_map(context, self.innovation_mask).decode(values)

    def encode(self, parameters: jax.Array, path: jax.Array) -> jax.Array:
        """Encode the same scientific starts into this sampler's native coordinates."""
        match self.parameterization:
            case CentredParameterization(path_center=center, path_chol=chol):
                values = jax.vmap(jnp.linalg.solve)(chol, path - center)
            case NonCentredParameterization() | MixedParameterization():
                context = self.target.context(parameters, self.target.times)
                path = self.target.replay_path(context, path)
                values = self.target.innovation_map(context, self.innovation_mask).encode(path)
        result = jnp.concatenate(
            (parameters - self.parameter_center, values.ravel()[self.free_indices])
        )
        return result if self.dimension else jnp.zeros(1, dtype=parameters.dtype)

    def decode(self, position: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Recover aligned scientific parameters and their exact-model trajectory."""
        parameters, values = self._unpack(position)
        context = self.target.context(parameters, self.target.times)
        path, _ = self._path(context, values)
        return parameters, path

    def log_density(self, position: jax.Array) -> jax.Array:
        """Score the authoritative scientific target plus the exact parameterization Jacobian.

        The full density retains centred/pinned marginals and correlated static
        initial factors. It also transforms a scientific Gaussian tempering
        reference consistently, with the same Jacobian at every temperature.
        """
        parameters, values = self._unpack(position)
        context = self.target.context(parameters, self.target.times)
        path, log_jacobian = self._path(context, values)
        density = jnp.asarray(
            self.target.log_prior(parameters)
            + self.target.factor_log_prob(context, path, self.target.observations)
        )
        auxiliary = position[self.dimension :]
        return (
            density
            + log_jacobian
            - 0.5 * jnp.sum(auxiliary**2)
            - 0.5 * auxiliary.size * jnp.log(2 * jnp.pi)
        )


def build_joint_coordinates(
    target: ParticleTarget, parameterization: JointParameterization = "centred"
) -> JointCoordinates:
    """Freeze the policy; parameter-dependent innovation maps bind at each evaluation."""
    position = target.initial_position
    free = target.independent_mask
    if target.exact_constraints is not None:
        free = free & target.exact_constraints.free_mask
    indices = jnp.asarray(np.flatnonzero(np.asarray(free)), dtype=jnp.int32)
    if parameterization == "non_centred":
        return JointCoordinates(target, position, free, indices, NonCentredParameterization())
    if parameterization == "mixed":
        return JointCoordinates(
            target, position, free, indices, MixedParameterization(free & target.point_reading_mask)
        )
    context = target.context(position, target.times)
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
    return JointCoordinates(
        target,
        position,
        free,
        indices,
        CentredParameterization(path, jnp.linalg.cholesky(embedded)),
    )
