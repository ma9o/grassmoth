"""Gaussian laws with exact values on compiler-declared deterministic coordinates."""

from typing import Literal, override

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
    pytree_data_fields: tuple[str, ...] = ("covariance_matrix",)  # noqa: V107 -- Retain the declared covariance across NumPyro pytree reconstruction.
    indices: tuple[int, ...]

    def __init__(
        self,
        mean: jax.Array,
        covariance: jax.Array,
        indices: tuple[int, ...],
        *,
        covariance_structure: Literal["diagonal", "dense"],
    ) -> None:
        """Factor only the declared stochastic covariance and embed it in state space."""
        self.indices = indices
        active = jnp.asarray(indices, dtype=jnp.int32)
        block = covariance[jnp.ix_(active, active)]
        variance = jnp.diag(block)
        positive = jnp.all(variance > 0)
        if covariance_structure == "diagonal":
            active_factor = jnp.where(
                positive, jnp.diag(jnp.sqrt(variance)), jnp.tril(jnp.full_like(block, jnp.nan))
            )
        else:
            active_factor = jnp.linalg.cholesky(block)
        factor = jnp.zeros_like(covariance).at[jnp.ix_(active, active)].set(active_factor)
        super().__init__(mean, scale_tril=factor, validate_args=False)
        if covariance_structure == "diagonal":
            # Retain the declared covariance so moment readers do not reconstruct L L.T.
            embedded = jnp.zeros_like(covariance).at[jnp.ix_(active, active)].set(block)
            stochastic = jnp.zeros(mean.shape[-1], dtype=bool).at[active].set(True)
            invalid = jnp.where(stochastic[:, None] | stochastic[None, :], jnp.nan, 0.0)
            self.covariance_matrix = jnp.where(positive, embedded, invalid)  # pyright: ignore[reportAttributeAccessIssue] -- NumPyro caches concrete arrays in its lazy_property instance slots.

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
