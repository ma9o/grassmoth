"""Exact Gaussian blocks and consumed analytical initialization scores."""

import jax
import jax.numpy as jnp
import jax.scipy.special
import jax.scipy.stats as jstats
from dynestyx.observation_missingness import masked_observation_log_prob

from nof1_causal_lab.models.ssm.execution.contracts import NUMERICAL_EPSILON, PROB_CLIP_MIN
from nof1_causal_lab.models.ssm.execution.observation_distributions import gaussian_distribution
from nof1_causal_lab.models.ssm.shapes import Array, Bool, Float, FloatScalar, Int, Shaped


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


def _logistic_pdf(x: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    p = jax.nn.sigmoid(x)
    return p * (1.0 - p)


def _logistic_pdf_prime(x: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    p = jax.nn.sigmoid(x)
    pdf = p * (1.0 - p)
    return pdf * (1.0 - 2.0 * p)


def _normalize_discrete_observation(
    y_t: Float[Array, " M"],
    level_counts: Int[Array, " M"],
) -> tuple[Int[Array, " M"], Bool[Array, " M"]]:
    rounded = jnp.rint(y_t)
    y_idx = rounded.astype(jnp.int32)
    valid = (
        jnp.isfinite(y_t)
        & jnp.isclose(y_t, rounded, atol=1e-4)
        & (y_idx >= 0)
        & (y_idx < level_counts)
    )
    safe_idx = jnp.clip(y_idx, 0, jnp.maximum(level_counts - 1, 0))
    return safe_idx, valid


def _select_rowwise(values: Float[Array, "M C"], indices: Int[Array, " M"]) -> Float[Array, " M"]:
    return jnp.take_along_axis(values, indices[:, None], axis=1).squeeze(axis=1)


def categorical_probabilities(
    eta: Float[Array, " M"],
    intercepts: Float[Array, "M cut"],
    slopes: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> Float[Array, "M C"]:
    """Return per-channel softmax probabilities over encoded categories."""
    eta = jnp.asarray(eta)
    intercepts = jnp.asarray(intercepts)
    slopes = jnp.asarray(slopes)
    level_counts = jnp.asarray(level_counts, dtype=jnp.int32)

    max_levels = intercepts.shape[1] + 1
    nonbaseline_mask = jnp.arange(1, max_levels)[None, :] < level_counts[:, None]
    logits_extra = intercepts + slopes * eta[:, None]
    logits_extra = jnp.where(nonbaseline_mask, logits_extra, -1e30)
    logits = jnp.concatenate([jnp.zeros((eta.shape[0], 1)), logits_extra], axis=1)

    probs = jax.nn.softmax(logits, axis=1)
    class_mask = jnp.arange(max_levels)[None, :] < level_counts[:, None]
    probs = jnp.where(class_mask, probs, 0.0)
    norm = jnp.sum(probs, axis=1, keepdims=True)
    return probs / norm


def _score_weight_poisson(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Poisson (log link): score = y - lambda, neg-Hessian = lambda."""
    lam = jnp.exp(eta)
    return (y_t - lam) * obs_mask_t, lam * obs_mask_t


def _score_weight_bernoulli_logit(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Bernoulli (logit link): score = y - p, neg-Hessian = p(1-p)."""
    p = jax.nn.sigmoid(eta)
    return (y_t - p) * obs_mask_t, (p * (1.0 - p)) * obs_mask_t


def _score_weight_beta_logit(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    concentration: float | jnp.ndarray,
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Beta (logit link): exact score and neg-Hessian via digamma/polygamma."""
    phi = concentration
    mu = jax.nn.sigmoid(eta)
    mu_c = jnp.clip(mu, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    alpha = phi * mu_c
    beta_ = phi * (1.0 - mu_c)
    y_c = jnp.clip(y_t, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    logit_y = jnp.log(y_c) - jnp.log(1.0 - y_c)
    sig_deriv = mu_c * (1.0 - mu_c)
    psi_diff = jax.scipy.special.digamma(beta_) - jax.scipy.special.digamma(alpha)
    score_mu = phi * (logit_y + psi_diff)
    g = sig_deriv * score_mu * obs_mask_t
    psi1_sum = jax.lax.polygamma(1.0, alpha) + jax.lax.polygamma(1.0, beta_)
    w_raw = phi * sig_deriv * (phi * sig_deriv * psi1_sum - (1.0 - 2.0 * mu_c) * score_mu)
    return g, jnp.maximum(w_raw, 0.0) * obs_mask_t


def _score_weight_gamma_log(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    shape: float | jnp.ndarray,
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Gamma (log link): score = a*(y/mu - 1), neg-Hessian = a*y/mu."""
    mu = jnp.maximum(jnp.exp(eta), 1e-8)
    ratio = y_t / mu
    return shape * (ratio - 1.0) * obs_mask_t, shape * ratio * obs_mask_t


def _score_weight_gamma_inverse(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    shape: float | jnp.ndarray,
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Gamma (inverse link): score = a*(mu - y), neg-Hessian = a*mu^2."""
    valid_eta = jnp.isfinite(eta) & (eta > 0.0)
    safe_eta = jnp.where(valid_eta, eta, 1.0)
    mu = 1.0 / safe_eta
    g = shape * (mu - y_t) * obs_mask_t
    w = (shape * mu**2) * obs_mask_t
    invalid_observed = (obs_mask_t > 0.5) & ~valid_eta
    g = jnp.where(invalid_observed, jnp.nan, g)
    w = jnp.where(invalid_observed, jnp.nan, w)
    return g, w


def _score_weight_negative_binomial(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    r: float | jnp.ndarray,
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Negative Binomial (log link): exact score and neg-Hessian."""
    logits = eta - jnp.log(r)
    p = jax.nn.sigmoid(logits)
    q = jax.nn.sigmoid(-logits)
    g = (y_t * q - r * p) * obs_mask_t
    w = (r + y_t) * p * q * obs_mask_t
    return g, w


def _score_weight_ordered_logistic(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    cutpoints: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Ordered-logistic score and neg-Hessian w.r.t. the linear predictor."""
    y_idx, valid_obs = _normalize_discrete_observation(y_t, level_counts)
    max_cutpoints = cutpoints.shape[1]
    lower_idx = jnp.clip(y_idx - 1, 0, max_cutpoints - 1)
    upper_idx = jnp.clip(y_idx, 0, max_cutpoints - 1)

    lower_arg = _select_rowwise(cutpoints, lower_idx) - eta
    upper_arg = _select_rowwise(cutpoints, upper_idx) - eta

    has_lower = y_idx > 0
    has_upper = y_idx < (level_counts - 1)

    lower_cdf = jnp.where(has_lower, jax.nn.sigmoid(lower_arg), 0.0)
    upper_cdf = jnp.where(has_upper, jax.nn.sigmoid(upper_arg), 1.0)
    lower_pdf = jnp.where(has_lower, _logistic_pdf(lower_arg), 0.0)
    upper_pdf = jnp.where(has_upper, _logistic_pdf(upper_arg), 0.0)
    lower_pdf_prime = jnp.where(has_lower, _logistic_pdf_prime(lower_arg), 0.0)
    upper_pdf_prime = jnp.where(has_upper, _logistic_pdf_prime(upper_arg), 0.0)

    prob = jnp.maximum(upper_cdf - lower_cdf, NUMERICAL_EPSILON)
    dprob = lower_pdf - upper_pdf
    d2prob = upper_pdf_prime - lower_pdf_prime

    valid_mask = obs_mask_t * valid_obs.astype(obs_mask_t.dtype)
    score = dprob / prob
    weight = jnp.maximum(score**2 - d2prob / prob, 0.0)
    return score * valid_mask, weight * valid_mask


def _score_weight_categorical(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    intercepts: Float[Array, "M cut"],
    slopes: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Categorical softmax score and neg-Hessian w.r.t. the linear predictor."""
    probs = categorical_probabilities(eta, intercepts, slopes, level_counts)
    y_idx, valid_obs = _normalize_discrete_observation(y_t, level_counts)
    slope_matrix = jnp.concatenate([jnp.zeros((eta.shape[0], 1)), slopes], axis=1)
    chosen_slope = _select_rowwise(slope_matrix, y_idx)
    mean_slope = jnp.sum(probs * slope_matrix, axis=1)
    second_moment = jnp.sum(probs * (slope_matrix**2), axis=1)

    valid_mask = obs_mask_t * valid_obs.astype(obs_mask_t.dtype)
    score = chosen_slope - mean_slope
    weight = jnp.maximum(second_moment - mean_slope**2, 0.0)
    return score * valid_mask, weight * valid_mask


def _score_weight_bernoulli_probit(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Bernoulli (probit link): exact score; Fisher information Hessian."""
    mu = jnp.clip(jstats.norm.cdf(eta), PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    phi_eta = jnp.exp(jstats.norm.logpdf(eta))
    var = mu * (1.0 - mu)
    return (y_t - mu) * phi_eta / var * obs_mask_t, (phi_eta**2 / var) * obs_mask_t


def _score_weight_beta_probit(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    concentration: float | jnp.ndarray,
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Beta (probit link): exact score; Gauss-Newton Hessian."""
    phi = concentration
    mu = jnp.clip(jstats.norm.cdf(eta), PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    alpha = phi * mu
    beta_ = phi * (1.0 - mu)
    y_c = jnp.clip(y_t, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    logit_y = jnp.log(y_c) - jnp.log(1.0 - y_c)
    phi_eta = jnp.exp(jstats.norm.logpdf(eta))
    score_mu = phi * (logit_y + jax.scipy.special.digamma(beta_) - jax.scipy.special.digamma(alpha))
    g = phi_eta * score_mu * obs_mask_t
    psi1_sum = jax.lax.polygamma(1.0, alpha) + jax.lax.polygamma(1.0, beta_)
    w = jnp.maximum(phi_eta**2 * phi**2 * psi1_sum, 0.0) * obs_mask_t
    return g, w
