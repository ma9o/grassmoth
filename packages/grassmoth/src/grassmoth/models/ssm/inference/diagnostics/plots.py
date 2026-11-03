"""Typed plot evidence reduced directly from retained native posterior buffers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.identity import ParameterRef
from grassmoth.artifacts.model.parameter import ParameterCoordinate
from grassmoth.artifacts.reports.empirical import EmpiricalDistribution
from grassmoth.artifacts.reports.posterior_diagnostics import (
    DensityCurve,
    ParetoKPoint,
    PosteriorMarginal,
    RankHistogram,
    TraceSeries,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


type ParameterReferences = Mapping[ParameterCoordinate, tuple[str, ParameterRef] | None]

HIST_PADDING_RATIO = 0.05
HIST_PADDING_DEFAULT = 0.5


def sample_coordinates(
    samples: Mapping[str, jnp.ndarray | NDArray], *, sample_dims: int
) -> Iterator[tuple[ParameterCoordinate, jnp.ndarray | NDArray]]:
    """Yield scalar event coordinates while preserving the requested leading sampling dimensions."""
    for name, values in samples.items():
        for indices in np.ndindex(values.shape[sample_dims:]):
            coordinate = ParameterCoordinate(site_name=name, indices=indices)
            yield coordinate, values[(slice(None),) * sample_dims + indices]


def build_trace_data(
    chain_samples: Mapping[str, jnp.ndarray | NDArray], references: ParameterReferences
) -> tuple[TraceSeries, ...]:
    """Keep every draw; auxiliary coordinates have an explicit None binding."""
    traces = []
    for coordinate, values in sample_coordinates(chain_samples, sample_dims=2):
        reference = references[coordinate]
        if reference is not None:
            _, subject = reference
            traces.append(
                TraceSeries(
                    subject=subject,
                    chains=tuple(tuple(float(v) for v in chain) for chain in np.asarray(values)),
                )
            )
    return tuple(traces)


def build_rank_histograms(
    chain_samples: Mapping[str, jnp.ndarray | NDArray],
    references: ParameterReferences,
    n_bins: int = 20,
) -> tuple[RankHistogram, ...]:
    """Build per-chain pooled-rank histograms for coordinates with scientific parameter references."""
    histograms = []
    for coordinate, values in sample_coordinates(chain_samples, sample_dims=2):
        reference = references[coordinate]
        if reference is None:
            continue
        _, subject = reference
        n_chains, n_samples = values.shape
        counts = np.asarray(_rank_counts(jnp.asarray(values), n_bins))
        histograms.append(
            RankHistogram(
                subject=subject,
                expected_per_bin=float(n_samples / n_bins),
                chains=tuple(
                    tuple(int(v) for v in chain)
                    for chain in np.asarray(counts).reshape(n_chains, n_bins)
                ),
            )
        )
    return tuple(histograms)


@partial(jax.jit, static_argnums=(1,))
def _rank_counts(values: jax.Array, n_bins: int) -> jax.Array:
    # Rejected MCMC moves repeat floats. Average ties, as in rank-normalized
    # convergence diagnostics, instead of assigning arbitrary sort-order ranks.
    pooled = values.reshape(-1)
    ordered = jnp.sort(pooled)
    left = jnp.searchsorted(ordered, pooled, side="left")
    right = jnp.searchsorted(ordered, pooled, side="right")
    ranks = ((left.astype(jnp.float32) + right + 1) / 2).reshape(values.shape)
    return jax.vmap(lambda chain: jnp.histogram(chain, bins=n_bins, range=(1, pooled.size + 1))[0])(
        ranks
    )


@partial(jax.jit, static_argnums=(1,))
def _marginal_summary(draws: jax.Array, n_bins: int) -> tuple[jax.Array, ...]:
    ordered = jnp.sort(draws)
    width = min(int(0.94 * draws.size), draws.size - 1)
    start = jnp.argmin(ordered[width:] - ordered[: draws.size - width])
    padding = jnp.where(
        ordered[-1] > ordered[0],
        (ordered[-1] - ordered[0]) * HIST_PADDING_RATIO,
        HIST_PADDING_DEFAULT,
    )
    density, edges = jnp.histogram(
        draws, bins=n_bins, range=(ordered[0] - padding, ordered[-1] + padding), density=True
    )
    return (
        (edges[:-1] + edges[1:]) / 2,
        density,
        jnp.stack((jnp.mean(draws), jnp.std(draws), ordered[start], ordered[start + width])),
    )


def param_marginal(
    parameter: str, subject: ParameterRef, values: jnp.ndarray | NDArray, n_bins: int = 50
) -> PosteriorMarginal:
    """Reduce full-chain moments, density, HDI and empirical CDF on the device."""
    draws = jnp.asarray(values, dtype=jnp.float32)
    centers, density, summary = (np.asarray(value) for value in _marginal_summary(draws, n_bins))
    unique, counts = jnp.unique(draws[jnp.isfinite(draws)], return_counts=True)
    cumulative = jnp.cumsum(counts) / jnp.sum(counts)
    return PosteriorMarginal(
        parameter=parameter,
        subject=subject,
        density_curve=DensityCurve(
            x=tuple(float(v) for v in centers), density=tuple(float(v) for v in density)
        ),
        empirical=EmpiricalDistribution(
            values=NumericalArray.from_numpy(np.asarray(unique)),
            probabilities=NumericalArray.from_numpy(np.asarray(cumulative)),
        ),
        mean=float(summary[0]),
        sd=float(summary[1]),
        interval_kind="hdi",
        interval_mass=0.94,
        lower=float(summary[2]),
        upper=float(summary[3]),
    )


def compute_posterior_marginals(
    samples: Mapping[str, jnp.ndarray | NDArray], references: ParameterReferences, n_bins: int = 50
) -> tuple[PosteriorMarginal, ...]:
    """Build posterior marginals for the sampled coordinates that have scientific parameter owners."""
    return tuple(
        param_marginal(reference[0], reference[1], values, n_bins)
        for coordinate, values in sample_coordinates(samples, sample_dims=1)
        if (reference := references[coordinate]) is not None
    )


def pareto_k_points(values: Sequence[float], timesteps: Sequence[int]) -> tuple[ParetoKPoint, ...]:
    """PSIS influence classes retain each original row in decreasing influence order."""
    return tuple(
        ParetoKPoint(
            timestep=timesteps[index],
            k=value
            if np.isfinite(value)
            else "undefined"
            if np.isnan(value)
            else "infinity"
            if value > 0
            else "-infinity",
            status="not_evaluated"
            if np.isnan(value)
            else "failed"
            if value > 0.7
            else "failed"
            if value > 0.5
            else "passed",
        )
        for index, value in sorted(
            enumerate(values),
            key=lambda item: (not np.isnan(item[1]), item[1] if not np.isnan(item[1]) else 0),
            reverse=True,
        )
    )
