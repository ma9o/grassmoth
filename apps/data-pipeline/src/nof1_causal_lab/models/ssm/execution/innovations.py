"""Exact conditional Gaussian coordinates on the compiled stochastic support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

if TYPE_CHECKING:
    from collections.abc import Callable


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


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class GaussianInnovationFactors:
    """State-independent conditional geometry, reusable across nonlinear means."""

    regression: jax.Array
    chol: jax.Array
    centred_mask: jax.Array
    innovation_mask: jax.Array

    def condition(self, mean: jax.Array, values: jax.Array) -> GaussianInnovationStep:
        """Bind the current drift and conditioned values without refactorizing."""
        conditional_mean = mean + self.regression @ jnp.where(self.centred_mask, values - mean, 0.0)
        return GaussianInnovationStep(conditional_mean, self.chol, self.innovation_mask)


def conditional_gaussian_factors(
    covariance: jax.Array,
    *,
    density_mask: jax.Array,
    innovation_mask: jax.Array,
) -> GaussianInnovationFactors:
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
    conditional_covariance = jnp.where(
        innovation_mask[:, None] & innovation_mask[None, :],
        covariance - regression @ nc.T,
        0.0,
    ) + jnp.diag((~innovation_mask).astype(covariance.dtype))
    return GaussianInnovationFactors(
        regression, jnp.linalg.cholesky(conditional_covariance), centred, innovation_mask
    )


def conditional_gaussian_step(
    mean: jax.Array,
    covariance: jax.Array,
    *,
    density_mask: jax.Array,
    innovation_mask: jax.Array,
    values: jax.Array,
) -> GaussianInnovationStep:
    """Bind one exact conditional Gaussian map, retaining its centred marginal."""
    return conditional_gaussian_factors(
        covariance, density_mask=density_mask, innovation_mask=innovation_mask
    ).condition(mean, values)


@dataclass(frozen=True, eq=False)
class GaussianPathMap:
    """Exact recurrence with additive covariance bound independently of the path."""

    initial_mean: jax.Array
    transition_mean: Callable[[jax.Array, jax.Array], jax.Array]
    covariances: jax.Array
    density_mask: jax.Array
    independent_mask: jax.Array
    innovation_mask: jax.Array
    fixed_values: jax.Array

    def _transition_factors(self) -> GaussianInnovationFactors:
        return jax.vmap(
            lambda covariance, density, innovation: conditional_gaussian_factors(
                covariance, density_mask=density, innovation_mask=innovation
            )
        )(self.covariances[1:], self.density_mask[1:], self.innovation_mask[1:])

    def _initial_step(self, values: jax.Array) -> GaussianInnovationStep:
        return conditional_gaussian_step(
            self.initial_mean,
            self.covariances[0],
            density_mask=self.density_mask[0],
            innovation_mask=self.innovation_mask[0],
            values=values,
        )

    def _values(self, coordinates: jax.Array, mean: jax.Array, index: jax.Array | int) -> jax.Array:
        values = jnp.where(self.independent_mask[index], coordinates, mean)
        return jnp.where(jnp.isnan(self.fixed_values[index]), values, self.fixed_values[index])

    def decode(self, coordinates: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Replay the declared dynamics while transforming only stochastic coordinates."""
        factors = self._transition_factors()
        values = self._values(coordinates[0], self.initial_mean, 0)
        first = self._initial_step(values)
        initial = first.decode(values)
        log_abs_det = first.log_abs_det + jnp.sum(
            jnp.log(jnp.diagonal(factors.chol, axis1=-2, axis2=-1))
        )

        def advance(
            previous: jax.Array, step: tuple[jax.Array, GaussianInnovationFactors]
        ) -> tuple[jax.Array, jax.Array]:
            index, geometry = step
            mean = self.transition_mean(previous, index)
            values = self._values(coordinates[index], mean, index)
            current = geometry.condition(mean, values).decode(values)
            return current, current

        _, tail = jax.lax.scan(advance, initial, (jnp.arange(1, coordinates.shape[0]), factors))
        return jnp.concatenate((initial[None], tail)), log_abs_det

    def encode(self, path: jax.Array) -> jax.Array:
        """Invert the recurrence on a valid scientific path, including its initial law."""
        factors = self._transition_factors()
        initial = self._initial_step(path[0]).encode(path[0])

        def inverse(
            previous: jax.Array,
            current: jax.Array,
            index: jax.Array,
            geometry: GaussianInnovationFactors,
        ) -> jax.Array:
            return geometry.condition(self.transition_mean(previous, index), current).encode(
                current
            )

        tail = jax.vmap(inverse)(path[:-1], path[1:], jnp.arange(1, path.shape[0]), factors)
        return jnp.concatenate((initial[None], tail))
