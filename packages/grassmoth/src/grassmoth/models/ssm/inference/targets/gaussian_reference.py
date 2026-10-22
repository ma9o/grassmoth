"""Frozen Gaussian reference on unconstrained parameters and free path coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from grassmoth.models.ssm.execution.affine import compose_affine_maps

if TYPE_CHECKING:
    from collections.abc import Callable


def _normal_log_prob(residual: jax.Array, chol: jax.Array) -> jax.Array:
    whitened = solve_triangular(chol, residual[..., None], lower=True)[..., 0]
    return (
        -0.5 * jnp.sum(whitened**2, axis=-1)
        - jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
        - 0.5 * residual.shape[-1] * jnp.log(2 * jnp.pi)
    )


@dataclass(frozen=True, eq=False)
class GaussianReference:
    """Normalized Gaussian law in the compiler's independent, unfixed coordinates.

    Inactive coordinates have identity covariance only for embedded linear algebra;
    they contribute neither noise nor density. Replaying the resulting coordinates
    through the exact model reconstructs deterministic states without adding noise.
    """

    parameter_mean: jax.Array
    parameter_chol: jax.Array
    path_mean: jax.Array
    autoregression: jax.Array
    innovation_chol: jax.Array
    free_mask: jax.Array

    def parameter_log_prob(self, position: jax.Array) -> jax.Array:
        """Score the normalized reference in unconstrained parameter coordinates."""
        return _normal_log_prob(position - self.parameter_mean, self.parameter_chol)

    def initial_log_prob(self, state: jax.Array) -> jax.Array:
        """Score free initial path coordinates; static unknowns belong to the parameter law."""
        return self._innovation_log_prob(state - self.path_mean[0], jnp.asarray(0))

    def transition_log_prob(
        self, previous: jax.Array, current: jax.Array, index: jax.Array
    ) -> jax.Array:
        """Score a Gaussian innovation, ignoring replayed and exactly observed values."""
        previous_residual = jnp.where(
            self.free_mask[index - 1], previous - self.path_mean[index - 1], 0.0
        )
        residual = current - self.path_mean[index] - self.autoregression[index] @ previous_residual
        return self._innovation_log_prob(residual, index)

    def _innovation_log_prob(self, residual: jax.Array, index: jax.Array) -> jax.Array:
        free = self.free_mask[index]
        return _normal_log_prob(jnp.where(free, residual, 0.0), self.innovation_chol[index]) + (
            0.5 * jnp.sum(~free) * jnp.log(2 * jnp.pi)
        )

    def path_log_prob(self, path: jax.Array) -> jax.Array:
        """Score the Gaussian law on the independent coordinates of a replayed path."""
        transitions = jax.vmap(self.transition_log_prob)(
            path[:-1], path[1:], jnp.arange(1, path.shape[0])
        )
        return self.initial_log_prob(path[0]) + jnp.sum(transitions)

    def sample(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Draw independent parameters and a correlated path using a parallel affine scan."""
        parameter_key, path_key = jax.random.split(key)
        position = self.parameter_mean + self.parameter_chol @ jax.random.normal(
            parameter_key, self.parameter_mean.shape, dtype=self.parameter_mean.dtype
        )
        noise = jnp.einsum(
            "tij,tj->ti",
            self.innovation_chol,
            jax.random.normal(path_key, self.path_mean.shape, dtype=self.path_mean.dtype),
        )
        noise = jnp.where(self.free_mask, noise, 0.0)

        _, residuals = jax.lax.associative_scan(compose_affine_maps, (self.autoregression, noise))
        return position, jnp.asarray(self.path_mean + residuals)


@dataclass(frozen=True, eq=False)
class GaussianReferenceBridge:
    """Geometric path r^(1-beta) p^beta with exact endpoint evaluations."""

    reference: GaussianReference
    beta: jax.Array

    def combine(self, exact: Callable[[], jax.Array], reference: jax.Array) -> jax.Array:
        """Interpolate log densities without evaluating the exact target at beta zero."""
        # NumPyro's empty-parameter prior and a model's path can have different
        # precisions. Resolve the common result type while tracing, so endpoint
        # branches agree without downcasting the exact cold density.
        dtype = jnp.result_type(jax.eval_shape(exact).dtype, reference.dtype, self.beta.dtype)
        reference = reference.astype(dtype)

        def exact_density() -> jax.Array:
            return jnp.asarray(exact(), dtype=dtype)

        result: jax.Array = jax.lax.cond(
            self.beta == 0,
            lambda: reference,
            lambda: jax.lax.cond(
                self.beta == 1,
                exact_density,
                lambda: self.beta * exact_density() + (1 - self.beta) * reference,
            ),
        )
        return result

    def observation(self, exact: Callable[[], jax.Array]) -> jax.Array:
        """Temper a scalar likelihood factor; the reference has no observation factors."""
        return self.combine(exact, jnp.asarray(0.0, dtype=self.reference.path_mean.dtype))
