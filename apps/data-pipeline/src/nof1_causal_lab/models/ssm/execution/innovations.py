"""Exact conditional Gaussian coordinates on the compiled stochastic support."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


@dataclass(frozen=True, eq=False)
class GaussianInnovationStep:
    """A conditional Gaussian map, with identity padding outside its innovations."""

    mean: jax.Array
    chol: jax.Array
    innovation_mask: jax.Array

    def decode(self, coordinates: jax.Array) -> jax.Array:
        """Keep centred values and replace innovations by their conditional state."""
        noise = jnp.where(self.innovation_mask, coordinates, 0.0)
        return jnp.where(self.innovation_mask, self.mean + self.chol @ noise, coordinates)

    def encode(self, state: jax.Array) -> jax.Array:
        """Recover innovations without changing the conditioned coordinates."""
        residual = jnp.where(self.innovation_mask, state - self.mean, 0.0)
        noise = solve_triangular(self.chol, residual, lower=True)
        return jnp.where(self.innovation_mask, noise, state)

    @property
    def log_abs_det(self) -> jax.Array:
        """Jacobian on the innovation coordinates; every padding diagonal is one."""
        return jnp.sum(jnp.log(jnp.diag(self.chol)))


def conditional_gaussian_step(
    mean: jax.Array,
    covariance: jax.Array,
    *,
    density_mask: jax.Array,
    innovation_mask: jax.Array,
    values: jax.Array,
) -> GaussianInnovationStep:
    """Condition on centred, pinned and global stochastic coordinates exactly.

    Padding permits one compiled scan even when the mask changes at each time.
    It adds no stochastic coordinate to the scientific law. The centred marginal
    remains owned by the exact target; this map supplies only the conditional
    transformation and its Jacobian.
    """
    centred = density_mask & ~innovation_mask
    cc = jnp.where(centred[:, None] & centred[None, :], covariance, 0.0)
    cc = cc + jnp.diag((~centred).astype(covariance.dtype))
    nc = jnp.where(innovation_mask[:, None] & centred[None, :], covariance, 0.0)
    regression = jnp.linalg.solve(cc, nc.T).T
    conditional_mean = mean + regression @ jnp.where(centred, values - mean, 0.0)
    conditional_covariance = jnp.where(
        innovation_mask[:, None] & innovation_mask[None, :],
        covariance - regression @ nc.T,
        0.0,
    ) + jnp.diag((~innovation_mask).astype(covariance.dtype))
    return GaussianInnovationStep(
        conditional_mean, jnp.linalg.cholesky(conditional_covariance), innovation_mask
    )


@dataclass(frozen=True, eq=False)
class GaussianPathMap:
    """Exact forward recurrence and inverse on a parameter-bound scientific law."""

    initial_moments: tuple[jax.Array, jax.Array]
    transition_moments: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
    density_mask: jax.Array
    independent_mask: jax.Array
    innovation_mask: jax.Array
    fixed_values: jax.Array

    def _step(
        self, moments: tuple[jax.Array, jax.Array], values: jax.Array, index: jax.Array | int
    ) -> GaussianInnovationStep:
        return conditional_gaussian_step(
            *moments,
            density_mask=self.density_mask[index],
            innovation_mask=self.innovation_mask[index],
            values=values,
        )

    def _values(self, coordinates: jax.Array, mean: jax.Array, index: jax.Array | int) -> jax.Array:
        values = jnp.where(self.independent_mask[index], coordinates, mean)
        return jnp.where(jnp.isnan(self.fixed_values[index]), values, self.fixed_values[index])

    def decode(self, coordinates: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Replay the declared dynamics while transforming only stochastic coordinates."""
        values = self._values(coordinates[0], self.initial_moments[0], 0)
        initial_step = self._step(self.initial_moments, values, 0)
        initial = initial_step.decode(values)

        def advance(previous: jax.Array, index: jax.Array):
            moments = self.transition_moments(previous, index)
            values = self._values(coordinates[index], moments[0], index)
            step = self._step(moments, values, index)
            current = step.decode(values)
            return current, (current, step.log_abs_det)

        _, (tail, determinants) = jax.lax.scan(
            advance, initial, jnp.arange(1, coordinates.shape[0])
        )
        return jnp.concatenate((initial[None], tail)), initial_step.log_abs_det + determinants.sum()

    def encode(self, path: jax.Array) -> jax.Array:
        """Invert the recurrence on a valid scientific path, including its initial law."""
        initial = self._step(self.initial_moments, path[0], 0).encode(path[0])

        def inverse(previous: jax.Array, current: jax.Array, index: jax.Array) -> jax.Array:
            return self._step(self.transition_moments(previous, index), current, index).encode(
                current
            )

        tail = jax.vmap(inverse)(path[:-1], path[1:], jnp.arange(1, path.shape[0]))
        return jnp.concatenate((initial[None], tail))
