"""Bounded float32 JAX evaluation of native rank, split-chain and Geyer diagnostics."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtri
from numpy.typing import NDArray


def _split(values: jax.Array) -> jax.Array:
    half = values.shape[-1] // 2
    return jnp.concatenate((values[..., :half], values[..., -half:]), axis=-2)


def _rank_normalize(values: jax.Array, ordered: jax.Array) -> jax.Array:
    """Return normal scores with Blom's offset.

    As in Vehtari et al. (2021), https://arxiv.org/abs/1903.08008, equation 14.
    """
    pooled = values.reshape(values.shape[0], -1)
    left = jax.vmap(lambda a, v: jnp.searchsorted(a, v, side="left"))(ordered, pooled)
    right = jax.vmap(lambda a, v: jnp.searchsorted(a, v, side="right"))(ordered, pooled)
    # Integer bounds give exact average ranks, including long runs of rejected moves.
    ranks = (left.astype(jnp.float32) + right.astype(jnp.float32) + 1) / 2
    return ndtri((ranks - 3 / 8) / (pooled.shape[-1] + 1 / 4)).reshape(values.shape)


def _quantile(ordered: jax.Array, probability: jax.Array) -> jax.Array:
    """Linear quantile interpolation on an already sorted coordinate pool."""
    index = (ordered.shape[-1] - 1) * probability
    lower = jnp.floor(index).astype(jnp.int32)
    upper = jnp.ceil(index).astype(jnp.int32)
    weight = index - lower
    return (1 - weight) * ordered[:, lower] + weight * ordered[:, upper]


def _rhat(values: jax.Array) -> jax.Array:
    draws = values.shape[-1]
    between = draws * jnp.var(values.mean(axis=-1), axis=-1, ddof=1)
    within = jnp.var(values, axis=-1, ddof=1).mean(axis=-1)
    return jnp.sqrt((between / within + draws - 1) / draws)


def _ess(values: jax.Array) -> jax.Array:
    """Reduce all lags together, preserving ArviZ's terminal-even-lag convention.

    ESS is capped at S log10(S) for short antithetic chains (Vehtari et al. 2021, section 3.2).
    The absolute 1e-15 constant-input threshold is ArviZ's, not float32 resolution.
    """
    values = values.astype(jnp.float32)
    chains, draws = values.shape[-2:]
    fft_size = 1 << (2 * draws - 1).bit_length()
    centered = values - values.mean(axis=-1, keepdims=True)
    spectrum = jnp.fft.rfft(centered, n=fft_size, axis=-1)
    # The inverse transform is linear. Average the chain power spectra first,
    # so only one inverse FFT per coordinate is needed, not one per split chain.
    mean_power = (spectrum * jnp.conj(spectrum)).mean(axis=-2)
    acov = jnp.fft.irfft(mean_power, n=fft_size, axis=-1)[..., :draws] / draws
    mean_var = acov[:, 0] * draws / (draws - 1)
    var_plus = mean_var * (draws - 1) / draws
    if chains > 1:
        var_plus = var_plus + jnp.var(values.mean(axis=-1), axis=-1, ddof=1)
    rho = 1 - (mean_var[:, None] - acov) / var_plus[:, None]
    rho = rho.at[:, 0].set(1)

    # The last visited pair supplies only its even lag. Earlier positive pairs
    # are clipped to a non-increasing sequence before they enter the sum.
    pairs = max(1, (draws - 1) // 2)
    paired = rho[:, : 2 * pairs].reshape(-1, pairs, 2)
    pair_sums = paired.sum(axis=-1)
    last = jnp.minimum(jnp.cumprod(pair_sums > 0, axis=-1).sum(axis=-1), pairs - 1)
    monotone = jax.lax.associative_scan(jnp.minimum, pair_sums, axis=-1)
    prefix = jnp.sum(jnp.where(jnp.arange(pairs) < last[:, None], monotone, 0), axis=-1)
    rows = jnp.arange(values.shape[0])
    even = paired[rows, last, 0]
    terminal = jnp.where((pair_sums[rows, last] >= 0) | (even > 0), even, 0)
    total = chains * draws
    tau = jnp.maximum(-1 + 2 * prefix + terminal, 1 / jnp.log10(jnp.float32(total)))
    result = total / tau
    result = jnp.where(jnp.isnan(rho[:, :2]).any(axis=-1), jnp.nan, result)
    constant = jnp.ptp(values, axis=(-2, -1)) < 1e-15
    return jnp.where(constant, total, result)


@partial(jax.jit, static_argnames=("tail_probability",))
def _chunk_statistics(samples: jax.Array, tail_probability: float) -> jax.Array:
    """Return rank R-hat, bulk ESS, tail ESS and mean MCSE for each coordinate."""
    split = _split(samples)
    ordered = jnp.sort(split.reshape(split.shape[0], -1), axis=-1, stable=False)
    normalized = _rank_normalize(split, ordered)
    bulk_rhat = _rhat(normalized)
    median = _quantile(ordered, jnp.float32(0.5))
    folded = jnp.abs(split - median[:, None, None])
    folded_ordered = jnp.sort(folded.reshape(folded.shape[0], -1), axis=-1, stable=False)
    tail_rhat = _rhat(_rank_normalize(folded, folded_ordered))
    rank_rhat = jnp.where(tail_rhat > bulk_rhat, tail_rhat, bulk_rhat)
    if samples.shape[-2] < 2:
        rank_rhat = jnp.full(samples.shape[0], jnp.nan, dtype=jnp.float32)
    # Even-length splitting retains the full pool: reuse its ordering for both
    # tail quantiles. Odd-length splitting omits the middle draw of each chain.
    full_ordered = (
        jnp.sort(samples.reshape(samples.shape[0], -1), axis=-1, stable=False)
        if samples.shape[-1] % 2
        else ordered
    )
    # An indicator at a linearly interpolated quantile includes exactly the
    # lower order statistic (and all its ties). Do not round the interpolant
    # to float32: it can round up to the next draw and change the indicator.
    # Probability-to-index arithmetic is static host work, not float64 device work.
    low, high = (
        full_ordered[:, int((full_ordered.shape[-1] - 1) * probability), None, None]
        for probability in sorted((tail_probability, 1 - tail_probability))
    )
    low_ess = _ess(_split(samples <= low))
    high_ess = _ess(_split(samples <= high))
    tail_ess = jnp.where(high_ess < low_ess, high_ess, low_ess)
    mean_mcse = jnp.std(samples, axis=(-2, -1), ddof=1) / jnp.sqrt(_ess(split))
    result = jnp.stack((rank_rhat, _ess(normalized), tail_ess, mean_mcse), axis=-1)
    return jnp.where(jnp.isnan(samples).any(axis=(-2, -1))[:, None], jnp.nan, result)


def chain_statistic_batch_size(chains: int, draws: int) -> int:
    """Bound coordinate batches by their padded FFT chain-lag workspace.

    The 2**23 workspace and the 256 cap are heuristics; no paper or recorded benchmark sets them.
    """
    fft_size = 1 << (2 * (draws // 2) - 1).bit_length()
    return min(256, max(1, 2**23 // (2 * chains * fft_size)))


def chain_statistics(samples: jax.Array | NDArray) -> NDArray:
    """Summarize (chain, draw, coordinate) inputs on JAX's default device.

    The columns are rank R-hat, bulk ESS, tail ESS and mean MCSE. Batches target
    eight million padded FFT chain-lag values, with at least one and at most
    256 coordinates. The last batch is padded to reuse the
    same executable; only the small summaries return to the host. No draws are
    thinned. Definitions and short-chain behavior match arviz-stats.
    """
    from arviz_stats.validate import validate_ci_prob

    if samples.shape[0] < 1 or samples.shape[1] < 4:
        return np.full((samples.shape[-1], 4), np.nan, dtype=np.float32)
    # The xarray diagnostics resolve omitted tail probabilities through ArviZ's
    # configured interval probability, which need not be the conventional 0.95.
    tail_probability = validate_ci_prob(None)
    batch_size = min(samples.shape[-1], chain_statistic_batch_size(*samples.shape[:2]))
    batches = []
    for start in range(0, samples.shape[-1], batch_size):
        batch = jnp.asarray(samples[:, :, start : start + batch_size], dtype=jnp.float32).transpose(
            2, 0, 1
        )
        count = batch.shape[0]
        padded = (
            jnp.pad(batch, ((0, batch_size - count), (0, 0), (0, 0)))
            if count < batch_size
            else batch
        )
        # Synchronize each small result before enqueueing another large workspace.
        batches.append(np.asarray(_chunk_statistics(jnp.asarray(padded), tail_probability))[:count])
    return np.concatenate(batches, axis=0)
