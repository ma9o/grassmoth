"""Bounded vectorization of the native ArviZ rank, split-chain and Geyer diagnostics."""

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def _split(values: NDArray) -> NDArray:
    half = values.shape[-1] // 2
    return np.concatenate((values[..., :half], values[..., -half:]), axis=-2)


def _rank_normalize(values: NDArray) -> NDArray:
    pooled = values.reshape(values.shape[0], -1)
    ranks = stats.rankdata(pooled, axis=-1, method="average", nan_policy="omit")
    return stats.norm.ppf((ranks - 3 / 8) / (pooled.shape[-1] + 1 / 4)).reshape(values.shape)


def _rhat(values: NDArray) -> NDArray:
    draws = values.shape[-1]
    between = draws * np.var(values.mean(axis=-1), axis=-1, ddof=1)
    within = np.var(values, axis=-1, ddof=1).mean(axis=-1)
    return np.sqrt((between / within + draws - 1) / draws)


def _ess(values: NDArray) -> NDArray:
    """Reduce all lags together, preserving ArviZ's terminal-even-lag convention."""
    from arviz_stats.base.array import array_stats

    values = np.asarray(values, dtype=float)
    chains, draws = values.shape[-2:]
    acov = array_stats.autocov(values, axis=-1).mean(axis=-2)
    mean_var = acov[:, 0] * draws / (draws - 1)
    var_plus = mean_var * (draws - 1) / draws
    if chains > 1:
        var_plus = var_plus + np.var(values.mean(axis=-1), axis=-1, ddof=1)
    rho = 1 - (mean_var[:, None] - acov) / var_plus[:, None]
    rho[:, 0] = 1

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
    tau = np.maximum(-1 + 2 * prefix + terminal, 1 / np.log10(total))
    result = total / tau
    result = np.where(np.isnan(rho[:, :2]).any(axis=-1), np.nan, result)
    constant = np.ptp(values, axis=(-2, -1)) < np.finfo(float).resolution
    return np.where(constant, total, result)


def _chunk_statistics(samples: NDArray, tail_probability: float) -> NDArray:
    """Return rank R-hat, bulk ESS, tail ESS and mean MCSE for each coordinate."""
    split = _split(samples)
    normalized = _rank_normalize(split)
    bulk_rhat = _rhat(normalized)
    folded = np.abs(split - np.median(split, axis=(-2, -1), keepdims=True))
    tail_rhat = _rhat(_rank_normalize(folded))
    rank_rhat = np.where(tail_rhat > bulk_rhat, tail_rhat, bulk_rhat)
    if samples.shape[-2] < 2:
        rank_rhat = np.full(samples.shape[0], np.nan)
    low, high = np.quantile(
        samples,
        sorted((tail_probability, 1 - tail_probability)),
        axis=(-2, -1),
        keepdims=True,
    )
    low_ess = _ess(_split(samples <= low))
    high_ess = _ess(_split(samples <= high))
    tail_ess = np.where(high_ess < low_ess, high_ess, low_ess)
    mean_mcse = np.std(samples, axis=(-2, -1), ddof=1) / np.sqrt(_ess(split))
    result = np.stack((rank_rhat, _ess(normalized), tail_ess, mean_mcse), axis=-1)
    return np.where(np.isnan(samples).any(axis=(-2, -1))[:, None], np.nan, result)


def chain_statistics(samples: NDArray) -> NDArray:
    """Summarize (chain, draw, coordinate) inputs without a Python loop per coordinate.

    The columns are rank R-hat, bulk ESS, tail ESS and mean MCSE. At most 256
    coordinates enter a batch; no chain or draw is thinned. The definitions and
    short-chain behavior match arviz-stats' native sampling diagnostics.
    """
    from arviz_stats.validate import validate_ci_prob

    if samples.shape[0] < 1 or samples.shape[1] < 4:
        return np.full((samples.shape[-1], 4), np.nan)
    # The xarray diagnostics resolve omitted tail probabilities through ArviZ's
    # configured interval probability, which need not be the conventional 0.95.
    tail_probability = validate_ci_prob(None)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.concatenate(
            tuple(
                _chunk_statistics(
                    np.ascontiguousarray(samples[:, :, start : start + 256].transpose(2, 0, 1)),
                    tail_probability,
                )
                for start in range(0, samples.shape[-1], 256)
            ),
            axis=0,
        )
