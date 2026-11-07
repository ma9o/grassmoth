"""Exact Gaussian observation blocks."""

from typing import Literal

import jax.numpy as jnp
from dynestyx.observation_missingness import masked_observation_log_prob

from grassmoth.models.ssm.covariance import CHOL_JITTER, symmetrize_with_jitter
from grassmoth.models.ssm.execution.gaussian_noise import (
    DiagonalGaussianNoiseFactors,
    GaussianNoiseFactors,
    PreparedGaussianNoise,
)
from grassmoth.models.ssm.execution.observation_distributions import gaussian_distribution
from grassmoth.models.ssm.shapes import FloatScalar


def prepare_gaussian_block_noise(
    covariance: jnp.ndarray, observed: jnp.ndarray, structure: Literal["diagonal", "dense"]
) -> PreparedGaussianNoise:
    """Factor the observed marginal once, for scans that reuse a fixed covariance.

    It is the law `gaussian_block_log_prob` scores, with the same symmetrization,
    jitter and missing-dimension normalization.
    """
    if structure == "diagonal":
        variance = jnp.where(observed, jnp.diag(covariance) + CHOL_JITTER, 1.0)
        factors = DiagonalGaussianNoiseFactors.from_scale(jnp.sqrt(variance))
        return factors.marginalize_dummies(jnp.sum(~observed))
    covariance = symmetrize_with_jitter(covariance)
    masked = jnp.where(observed[:, None] & observed[None, :], covariance, 0.0)
    masked = masked + jnp.diag((~observed).astype(covariance.dtype))
    factors = GaussianNoiseFactors.from_cholesky(jnp.linalg.cholesky(masked))
    return factors.marginalize_dummies(jnp.sum(~observed))


def gaussian_block_log_prob(
    y_t: jnp.ndarray, eta: jnp.ndarray, R: jnp.ndarray, obs_mask_t: jnp.ndarray
) -> FloatScalar:
    """Score the exact observed Gaussian marginal with Dynestyx's fixed-shape mask."""
    mask = obs_mask_t > 0.5
    law = gaussian_distribution(eta, R)
    return masked_observation_log_prob(
        law,
        y=jnp.where(mask, y_t, eta),
        obs_mask=mask,
    )
