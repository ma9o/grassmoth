"""Exact conditional Gaussian factors on the compiled stochastic support."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class GaussianInnovationFactors:
    """State-independent conditional geometry, reusable across nonlinear means."""

    regression: jax.Array
    chol: jax.Array
    centred_mask: jax.Array


def conditional_gaussian_factors(
    covariance: jax.Array,
    *,
    density_mask: jax.Array,
    innovation_mask: jax.Array,
) -> GaussianInnovationFactors:
    """Condition on centred, pinned and global stochastic coordinates exactly.

    Padding permits one compiled scan even when the mask changes at each time.
    It adds no stochastic coordinate to the scientific law. The centred marginal
    remains owned by the exact target; these factors supply only the conditional
    regression and covariance.
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
        regression, jnp.linalg.cholesky(conditional_covariance), centred
    )
