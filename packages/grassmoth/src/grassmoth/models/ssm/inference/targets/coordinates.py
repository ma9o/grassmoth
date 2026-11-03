"""Frozen time-indexed affine charts of the independent latent path coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class ScaledDeviationFrame:
    """One parameter label's map from chart to physical coordinates.

    ``x_t = center_t + offset_t + exp(log_scale_t) * (z_t - center_t)``.
    Both endpoints of a transition use their own time-indexed frame.
    """

    offset: jax.Array
    log_scale: jax.Array
    center: jax.Array

    def decode(self, state: jax.Array, index: jax.Array | int) -> jax.Array:
        """Restore physical coordinates at the supplied times."""
        center = self.center[index]
        return center + self.offset[index] + jnp.exp(self.log_scale[index]) * (state - center)

    def encode(self, state: jax.Array, index: jax.Array | int) -> jax.Array:
        """Express physical coordinates in this frame at the supplied times."""
        center = self.center[index]
        return center + (state - center - self.offset[index]) * jnp.exp(-self.log_scale[index])


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class PathChart:
    """One frozen translation and diagonal scaling of deviations from a Gaussian mean.

    ``x_t = c_t + B_t (position-reference) + exp(A_t (position-reference)) * (z_t-c_t)``.
    Slopes B and exponents A vanish on excluded coordinates at each time. All
    coefficients share one reference surrogate. Their approximation affects mixing;
    physical densities and time-indexed Jacobians own the exact target.
    """

    slopes: jax.Array
    reference: jax.Array
    exponents: jax.Array
    center: jax.Array

    def offset(self, position: jax.Array) -> jax.Array:
        """Return the translation at every time slice."""
        return self.slopes @ (position - self.reference)

    def log_scale(self, position: jax.Array) -> jax.Array:
        """Return the log scale at every time slice."""
        return self.exponents @ (position - self.reference)

    def representable(self, position: jax.Array) -> jax.Array:
        """Require finite normal scale squares and reciprocals in the runtime dtype.

        Distant labels outside this numerical domain receive zero proposal weight,
        before they contribute to common guide geometry. Never clip the exponent:
        every active label uses the declared exponential map and its exact Jacobian.
        Initial and rebuilt charts must represent every current chain position.
        """
        logarithm = self.log_scale(position)
        limit = (
            jnp.minimum(
                jnp.log(jnp.finfo(logarithm.dtype).max), -jnp.log(jnp.finfo(logarithm.dtype).tiny)
            )
            / 2
        )
        return jnp.all(jnp.isfinite(logarithm) & (jnp.abs(logarithm) < limit))

    def usable_at(self, positions: jax.Array) -> jax.Array:
        """Check complete coefficients and the current population at the builder boundary."""
        finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value))
                    for value in (self.slopes, self.exponents, self.center, self.reference)
                )
            )
        )
        return finite & jnp.all(jax.vmap(self.representable)(positions))

    def frame(self, position: jax.Array) -> ScaledDeviationFrame:
        """Bind one label's map for every factor and proposal."""
        return ScaledDeviationFrame(self.offset(position), self.log_scale(position), self.center)

    def encode(self, position: jax.Array, path: jax.Array) -> jax.Array:
        """Express a physical trajectory in the chart."""
        return self.frame(position).encode(path, jnp.arange(path.shape[0]))

    def decode(self, position: jax.Array, path: jax.Array) -> jax.Array:
        """Restore a physical trajectory from the chart."""
        return self.frame(position).decode(path, jnp.arange(path.shape[0]))
