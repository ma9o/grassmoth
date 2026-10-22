"""Gaussian laws with exact values on compiler-declared deterministic coordinates."""

from typing import override

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike


class SupportedNormal(dist.MultivariateNormal):
    """Embed a full-rank Gaussian in a state vector without perturbing its support.

    Native Gaussian sampling and moments apply to the embedded Cholesky factor.
    Density evaluation uses the stochastic coordinates and exact point masses on
    their complement, avoiding a solve with the singular full-state factor.
    """

    pytree_aux_fields: tuple[str, ...] = ("indices",)  # noqa: V107 -- NumPyro reads this registration when flattening distributions.
    indices: tuple[int, ...]

    def __init__(self, mean: jax.Array, covariance: jax.Array, indices: tuple[int, ...]) -> None:
        """Factor only the declared stochastic covariance and embed it in state space."""
        self.indices = indices
        active = jnp.asarray(indices, dtype=jnp.int32)
        factor = (
            jnp.zeros_like(covariance)
            .at[jnp.ix_(active, active)]
            .set(jnp.linalg.cholesky(covariance[jnp.ix_(active, active)]))
        )
        super().__init__(mean, scale_tril=factor, validate_args=False)

    @override
    def log_prob(self, value: ArrayLike, intermediates: list[object] | None = None) -> jax.Array:
        del intermediates
        value = jnp.asarray(value)
        active = jnp.asarray(self.indices, dtype=jnp.int32)
        stochastic = (
            dist.MultivariateNormal(
                jnp.asarray(self.loc)[active],
                scale_tril=jnp.asarray(self.scale_tril)[jnp.ix_(active, active)],
            ).log_prob(value[..., active])
            if self.indices
            else jnp.zeros(value.shape[:-1], dtype=value.dtype)
        )
        fixed = jnp.ones(self.event_shape, dtype=bool).at[active].set(False)
        on_support = jnp.all(jnp.where(fixed, value == self.loc, True), axis=-1)
        return jnp.where(on_support, stochastic, -jnp.inf)
