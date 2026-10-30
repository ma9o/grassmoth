"""Frozen parameter-dependent charts of latent paths: translations and scaled deviations."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class TranslationChart:
    """Time-indexed latent/parameter slopes, fixed throughout each exact update.

    The joint map is a shear: ``x = z + slopes @ (position - reference)``.
    Its joint Jacobian is one even though the offset depends on the parameter.
    Computing the slopes belongs to warmup; a poorly fitted chart affects mixing,
    never the exact physical target evaluated through its inverse.
    """

    slopes: jax.Array
    reference: jax.Array

    def offset(self, position: jax.Array) -> jax.Array:
        """Return the translation at every time slice at this position."""
        return self.slopes @ (position - self.reference)

    def encode(self, position: jax.Array, path: jax.Array) -> jax.Array:
        """Express a physical trajectory in the chart."""
        return path - self.offset(position)

    def decode(self, position: jax.Array, path: jax.Array) -> jax.Array:
        """Restore a physical trajectory from the chart."""
        return path + self.offset(position)

    def frame(self, position: jax.Array) -> ScaledDeviationFrame:
        """Bind the time-indexed shear for every particle factor and proposal."""
        offset = self.offset(position)
        return ScaledDeviationFrame(
            offset, jnp.zeros(offset.shape[-1], offset.dtype), jnp.zeros_like(offset)
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class ScaledDeviationFrame:
    """One parameter label's affine map from chart to physical coordinates.

    ``x_t = center_t + offset_t + exp(log_scale) * (z_t - center_t)``. The map
    depends on time through the frozen slopes and center path, so every state is
    decoded with its own time index.
    """

    offset: jax.Array
    log_scale: jax.Array
    center: jax.Array

    def decode(self, state: jax.Array, index: jax.Array | int) -> jax.Array:
        """Restore physical coordinates of a state, batch or path at these times."""
        center = self.center[index]
        return center + self.offset[index] + jnp.exp(self.log_scale) * (state - center)

    def encode(self, state: jax.Array, index: jax.Array | int) -> jax.Array:
        """Express physical coordinates of a state, batch or path at these times."""
        center = self.center[index]
        return center + (state - center - self.offset[index]) * jnp.exp(-self.log_scale)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class ScaleChart:
    """A translation composed with parameter-dependent scaling of path deviations.

    ``x_t = c_t + B (position - reference) + exp(A (position - reference)) * (z_t - c_t)``.
    The exponents A take each stochastic state's change in log diffusion scale, so
    changing that scale with z fixed rescales the path's deviations from the frozen
    center path c instead of leaving its increments in place. The per-time Jacobian
    is ``exp(sum(A (position - reference)))`` over the modeled coordinates; every
    label pays it exactly. The center and exponents only shape mixing: deviations
    that are not diffusion-driven, such as data-pinned slow modes, a poor center or
    a nonlinear diffusion, cost efficiency, never the exact physical target.
    """

    translation: TranslationChart
    exponents: jax.Array
    center: jax.Array

    def log_scale(self, position: jax.Array) -> jax.Array:
        """Return each coordinate's log scale relative to the chart reference."""
        return self.exponents @ (position - self.translation.reference)

    def frame(self, position: jax.Array) -> ScaledDeviationFrame:
        """Bind this position's map for every particle density and proposal."""
        return ScaledDeviationFrame(
            self.translation.offset(position), self.log_scale(position), self.center
        )

    def encode(self, position: jax.Array, path: jax.Array) -> jax.Array:
        """Express a physical trajectory in the chart."""
        return self.frame(position).encode(path, jnp.arange(path.shape[0]))

    def decode(self, position: jax.Array, path: jax.Array) -> jax.Array:
        """Restore a physical trajectory from the chart."""
        return self.frame(position).decode(path, jnp.arange(path.shape[0]))


type PathChart = TranslationChart | ScaleChart
