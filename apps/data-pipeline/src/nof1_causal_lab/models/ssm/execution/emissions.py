"""Exact Gaussian observation blocks."""

import jax.numpy as jnp
from dynestyx.observation_missingness import masked_observation_log_prob

from nof1_causal_lab.models.ssm.execution.observation_distributions import gaussian_distribution
from nof1_causal_lab.models.ssm.shapes import FloatScalar


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
