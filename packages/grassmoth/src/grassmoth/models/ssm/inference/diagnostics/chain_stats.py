"""Bounded host evaluation of native float32 rank, split-chain and Geyer diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.fft import irfft, rfft
from scipy.special import ndtri

if TYPE_CHECKING:
    import jax
    from numpy.typing import NDArray


def _split(values: NDArray) -> NDArray:
    half = values.shape[-1] // 2
    return np.concatenate((values[..., :half], values[..., -half:]), axis=-2)


def _rank_normalize(values: NDArray, ordered: NDArray) -> NDArray:
    """Return normal scores with Blom's offset.

    As in Vehtari et al. (2021), https://arxiv.org/abs/1903.08008, equation 14.
    """
    pooled = values.reshape(values.shape[0], -1)
    left = np.stack(
        [np.searchsorted(a, v, side="left") for a, v in zip(ordered, pooled, strict=True)]
    )
    right = np.stack(
        [np.searchsorted(a, v, side="right") for a, v in zip(ordered, pooled, strict=True)]
    )
    # Integer bounds give exact average ranks, including long runs of rejected moves.
    ranks = (left.astype(np.float32) + right.astype(np.float32) + 1) / 2
    return ndtri((ranks - 3 / 8) / (pooled.shape[-1] + 1 / 4)).reshape(values.shape)


def _quantile(ordered: NDArray, probability: np.float32) -> NDArray:
    """Linear quantile interpolation on an already sorted coordinate pool."""
    index = (ordered.shape[-1] - 1) * probability
    lower = np.floor(index).astype(np.int32)
    upper = np.ceil(index).astype(np.int32)
    weight = np.float32(index - lower)
    return (1 - weight) * ordered[:, lower] + weight * ordered[:, upper]


def _rhat(values: NDArray) -> NDArray:
    draws = values.shape[-1]
    between = draws * np.var(values.mean(axis=-1), axis=-1, ddof=1)
    # Subtracting an observed value leaves variances unchanged and makes a
    # chain of identical nonzero normal scores have exactly zero variance.
    within = np.var(values - values[..., :1], axis=-1, ddof=1).mean(axis=-1)
    return np.sqrt((between / within + draws - 1) / draws)


def _ess(values: NDArray) -> NDArray:
    """Reduce all lags together, preserving ArviZ's terminal-even-lag convention.

    ESS is capped at S log10(S) for short antithetic chains (Vehtari et al. 2021, section 3.2).
    The absolute 1e-15 constant-input threshold is ArviZ's, not float32 resolution.
    """
    values = values.astype(np.float32, copy=False)
    chains, draws = values.shape[-2:]
    fft_size = 1 << (2 * draws - 1).bit_length()
    centered = values - values.mean(axis=-1, keepdims=True)
    spectrum = np.asarray(rfft(centered, n=fft_size, axis=-1))
    # The inverse transform is linear. Average the chain power spectra first,
    # so only one inverse FFT per coordinate is needed, not one per split chain.
    mean_power = (spectrum * np.conj(spectrum)).mean(axis=-2)
    acov = np.asarray(irfft(mean_power, n=fft_size, axis=-1))[..., :draws] / draws
    mean_var = acov[:, 0] * draws / (draws - 1)
    var_plus = mean_var * (draws - 1) / draws
    if chains > 1:
        var_plus = var_plus + np.var(values.mean(axis=-1), axis=-1, ddof=1)
    rho = 1 - (mean_var[:, None] - acov) / var_plus[:, None]
    rho = np.concatenate((np.ones_like(rho[:, :1]), rho[:, 1:]), axis=-1)

    # The last visited pair supplies only its even lag. Earlier positive pairs
    # are clipped to a non-increasing sequence before they enter the sum.
    pairs = max(1, (draws - 1) // 2)
    paired = rho[:, : 2 * pairs].reshape(-1, pairs, 2)
    pair_sums = paired.sum(axis=-1)
    last = np.minimum(np.cumprod(pair_sums > 0, axis=-1).sum(axis=-1), pairs - 1)
    monotone = np.minimum.accumulate(pair_sums, axis=-1)
    prefix = np.sum(np.where(np.arange(pairs) < last[:, None], monotone, 0), axis=-1)
    rows = np.arange(values.shape[0])
    even = paired[rows, last, 0]
    terminal = np.where((pair_sums[rows, last] >= 0) | (even > 0), even, 0)
    total = chains * draws
    tau = np.maximum(-1 + 2 * prefix + terminal, 1 / np.log10(np.float32(total)))
    result = total / tau
    result = np.where(np.isnan(rho[:, :2]).any(axis=-1), np.nan, result)
    constant = np.ptp(values, axis=(-2, -1)) < 1e-15
    return np.where(constant, total, result)


def _chunk_statistics(samples: NDArray, tail_probability: float) -> NDArray:
    """Return rank R-hat, bulk ESS, tail ESS and mean MCSE for each coordinate."""
    split = _split(samples)
    ordered = np.sort(split.reshape(split.shape[0], -1), axis=-1)
    normalized = _rank_normalize(split, ordered)
    bulk_rhat = _rhat(normalized)
    median = _quantile(ordered, np.float32(0.5))
    folded = np.abs(split - median[:, None, None])
    folded_ordered = np.sort(folded.reshape(folded.shape[0], -1), axis=-1)
    tail_rhat = _rhat(_rank_normalize(folded, folded_ordered))
    rank_rhat = np.where(tail_rhat > bulk_rhat, tail_rhat, bulk_rhat)
    if samples.shape[-2] < 2:
        rank_rhat = np.full(samples.shape[0], np.nan, dtype=np.float32)
    # Even-length splitting retains the full pool: reuse its ordering for both
    # tail quantiles. Odd-length splitting omits the middle draw of each chain.
    full_ordered = (
        np.sort(samples.reshape(samples.shape[0], -1), axis=-1)
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
    tail_ess = np.where(high_ess < low_ess, high_ess, low_ess)
    mean_mcse = np.std(samples, axis=(-2, -1), ddof=1) / np.sqrt(_ess(split))
    result = np.stack((rank_rhat, _ess(normalized), tail_ess, mean_mcse), axis=-1)
    return np.where(np.isnan(samples).any(axis=(-2, -1))[:, None], np.nan, result)


def chain_statistic_batch_size(chains: int, draws: int) -> int:
    """Bound host coordinate batches by their padded FFT chain-lag workspace.

    The 2**23 workspace and the 256 cap are heuristics; no paper or recorded benchmark sets them.
    """
    fft_size = 1 << (2 * (draws // 2) - 1).bit_length()
    return min(256, max(1, 2**23 // (2 * chains * fft_size)))


def chain_statistics(samples: jax.Array | NDArray) -> NDArray:
    """Summarize (chain, draw, coordinate) inputs without allocating accelerator memory.

    The columns are rank R-hat, bulk ESS, tail ESS and mean MCSE. Batches target
    eight million padded FFT chain-lag values, with at least one and at most
    256 coordinates. Inputs transfer to the host before selection; SciPy's FFT
    preserves float32/complex64 arithmetic. No draws are thinned. Definitions
    and short-chain behavior match arviz-stats. The workspace heuristic bounds
    host batching only: reporting has no separate accelerator memory envelope.
    """
    from arviz_stats.validate import validate_ci_prob

    if samples.shape[0] < 1 or samples.shape[1] < 4:
        return np.full((samples.shape[-1], 4), np.nan, dtype=np.float32)
    samples = np.asarray(samples)
    # The xarray diagnostics resolve omitted tail probabilities through ArviZ's
    # configured interval probability, which need not be the conventional 0.95.
    tail_probability = validate_ci_prob(None)
    batch_size = min(samples.shape[-1], chain_statistic_batch_size(*samples.shape[:2]))
    batches = []
    for start in range(0, samples.shape[-1], batch_size):
        batch = np.asarray(samples[:, :, start : start + batch_size], dtype=np.float32).transpose(
            2, 0, 1
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            batches.append(_chunk_statistics(batch, tail_probability))
    return np.concatenate(batches, axis=0)
