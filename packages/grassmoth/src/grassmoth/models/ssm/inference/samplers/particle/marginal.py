"""Particle-mGRAD's integrated Gaussian bank density (CF24, section 4.2/D.2).

For candidate n, integrate the common auxiliary out of the other N particles:
u ~ N(x_n + E g_n, E), x_i | u ~ N(v_i + H u, D). Their joint covariance
is I_N tensor D + 11' tensor H E H'. Splitting residuals into their mean and
contrasts gives log q^{-n}, up to a constant common to all candidates, in O(N).
This form avoids subtracting two O(N^2) quadratic terms in float32.

The common guide is fixed by the parameter ensemble. Its nonlinear means enter
v_i, while additive diffusion makes D/H independent of the parent particle.
Exact label-mixture increments multiply these bank densities; the guide never
replaces the target. Poor guide overlap and separated modes can cost mixing.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.shapes import Array, Bool, Float


class MarginalGeometry(NamedTuple):
    """Factors of the individual proposal and the marginal particle mean."""

    auxiliary_map: jax.Array
    whiten: jax.Array
    mean_whiten: jax.Array

    def log_weights(
        self,
        particles: Float[Array, "P D"],
        prior_centers: Float[Array, "P D"],
        gradients: Float[Array, "P D"],
        variance: Float[Array, " D"],
        mask: Bool[Array, " D"],
    ) -> Float[Array, " P"]:
        """Log density of all other particles under each candidate reference."""
        count = particles.shape[0]
        others = count - 1
        residual = jnp.where(mask, particles - prior_centers, 0.0)
        mean = jnp.mean(residual, axis=0)
        contrast = (residual - mean) @ self.whiten.T
        auxiliary_mean = jnp.where(mask, particles + variance * gradients, 0.0)
        other_mean = (count * mean - residual) / others
        difference = (other_mean - auxiliary_mean @ self.auxiliary_map.T) @ self.mean_whiten.T
        return 0.5 * (
            count / others * jnp.sum(contrast**2, axis=-1)
            - others * jnp.sum(difference**2, axis=-1)
        )


def marginal_geometry(
    chol: Float[Array, "D D"],
    variance: Float[Array, " D"],
    mask: Bool[Array, " D"],
    count: int,
) -> MarginalGeometry:
    """Factor once per time; dummy pinned coordinates have zero residuals."""
    covariance = chol @ chol.T
    auxiliary_map = covariance * (mask / variance)[None, :]
    shared = (auxiliary_map * variance[None, :]) @ auxiliary_map.T
    mean_chol = jnp.linalg.cholesky(covariance + (count - 1) * shared)
    identity = jnp.eye(chol.shape[0], dtype=chol.dtype)
    return MarginalGeometry(
        auxiliary_map,
        jax.scipy.linalg.solve_triangular(chol, identity, lower=True),
        jax.scipy.linalg.solve_triangular(mean_chol, identity, lower=True),
    )
