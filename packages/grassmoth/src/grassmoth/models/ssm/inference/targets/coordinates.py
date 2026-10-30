"""Frozen unit-Jacobian parameter-dependent translations of latent paths."""

from dataclasses import dataclass

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class TranslationChart:
    """Time-constant latent/parameter slopes, fixed throughout each exact update.

    The joint map is a shear: ``x = z + slopes @ (position - reference)``.
    Its joint Jacobian is one even though the offset depends on the parameter.
    Learning the slopes belongs to warmup; a poorly fitted chart affects mixing,
    never the exact physical target evaluated through its inverse.
    """

    slopes: jax.Array
    reference: jax.Array

    def offset(self, position: jax.Array) -> jax.Array:
        """Return the shared translation of every time slice at this position."""
        return self.slopes @ (position - self.reference)

    def encode(self, position: jax.Array, path: jax.Array) -> jax.Array:
        """Express a physical state or trajectory in the chart."""
        return path - self.offset(position)

    def decode(self, position: jax.Array, path: jax.Array) -> jax.Array:
        """Restore a physical state or trajectory from the chart."""
        return path + self.offset(position)
