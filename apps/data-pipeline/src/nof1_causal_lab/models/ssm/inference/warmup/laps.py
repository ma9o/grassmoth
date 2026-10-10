"""LAPS initialization of the exact joint parameter and latent-path density.

A frozen affine coordinate change scales the free path coordinates before LAPS runs.
It changes coordinates, never the target: every score uses the production joint
density, with the transform's constant Jacobian. Neither a particle smoother nor
an approximate marginal likelihood is evaluated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.models.ssm.inference.warmup.laps_ensemble import run_laps_ensemble
from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.posterior_diagnostics import LapsInitializationDiagnostics
    from nof1_causal_lab.compute.budget import ComputeBudget
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget
    from nof1_causal_lab.sampler_config import LapsInitializationSpec


@dataclass(frozen=True, eq=False)
class LapsCoordinates:
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

    @property
    def sampling_dimension(self) -> int:
        """MCLMC needs at least two dimensions; independent normal auxiliaries fill tiny models."""
        return max(2, self.dimension)

    def decode(self, position: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Recover parameters and their matching exact-model trajectory."""
        innovations = jnp.zeros(self.path_center.size, dtype=position.dtype)
        innovations = innovations.at[self.free_indices].set(
            position[self.parameter_dimension : self.dimension]
        )
        innovations = innovations.reshape(self.path_center.shape)
        parameters = self.parameter_center + position[: self.parameter_dimension]
        coordinates = self.path_center + jnp.einsum("tij,tj->ti", self.path_chol, innovations)
        context = self.target.context(parameters, self.target.times)
        path = self.target.replay_path(context, coordinates)
        return parameters, path

    def log_density(self, position: jax.Array) -> jax.Array:
        """Score the exact nonlinear model after the frozen coordinate change."""
        parameters, path = self.decode(position)
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


def build_laps_coordinates(target: ParticleTarget) -> LapsCoordinates:
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
    return LapsCoordinates(target, position, path, jnp.linalg.cholesky(embedded), free, indices)


@dataclass(frozen=True)
class LapsInitializationResult:
    """Paired chain starts and ensemble proposal moments in the production coordinates."""

    positions: jax.Array
    paths: jax.Array
    parameter_chol: jax.Array
    pilot_mean: jax.Array
    pilot_variance: jax.Array
    diagnostics: LapsInitializationDiagnostics


def prepare_laps_initialization(
    target: ParticleTarget,
    *,
    spec: LapsInitializationSpec,
    key: jax.Array,
    num_chains: int,
    compute_budget: ComputeBudget,
) -> LapsInitializationResult | InitializationFailure:
    """Produce one initialization package without executing a particle smoother."""
    if num_chains > spec.ensemble_size:
        return InitializationFailure.unavailable(
            "LAPS ensemble_size must cover all requested particle chains"
        )
    coordinates = build_laps_coordinates(target)
    if not all(
        np.isfinite(np.asarray(value)).all()
        for value in (coordinates.parameter_center, coordinates.path_center, coordinates.path_chol)
    ):
        return InitializationFailure.unavailable(
            "The native starting point has no finite LAPS coordinate scale"
        )
    ensemble_key, selection_key = jax.random.split(key)
    ensemble = run_laps_ensemble(
        coordinates.log_density,
        dimension=coordinates.sampling_dimension,
        scientific_dimension=coordinates.dimension,
        key=ensemble_key,
        spec=spec,
        compute_budget=compute_budget,
    )
    if isinstance(ensemble, InitializationFailure):
        return ensemble
    positions, paths = jax.jit(jax.vmap(coordinates.decode))(ensemble.positions)
    centered = positions - jnp.mean(positions, axis=0)
    covariance = centered.T @ centered / (spec.ensemble_size - 1)
    covariance = covariance + spec.covariance_regularization * jnp.eye(
        coordinates.parameter_dimension, dtype=positions.dtype
    )
    chol = jnp.linalg.cholesky(covariance)
    if not np.isfinite(np.asarray(chol)).all() or not np.isfinite(np.asarray(paths)).all():
        return InitializationFailure.unavailable(
            "The LAPS ensemble has non-finite paths or parameter covariance"
        )
    selected = jax.random.choice(
        selection_key, spec.ensemble_size, shape=(num_chains,), replace=False
    )
    return LapsInitializationResult(
        positions=positions[selected],
        paths=paths[selected],
        parameter_chol=chol,
        pilot_mean=jnp.mean(paths, axis=0),
        pilot_variance=jnp.var(paths, axis=0),
        diagnostics=ensemble.diagnostics,
    )
