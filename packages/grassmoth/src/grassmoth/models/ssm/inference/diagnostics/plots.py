"""Typed plot evidence reduced directly from retained native posterior buffers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from grassmoth.artifacts.identity import ParameterRef
from grassmoth.artifacts.model.parameter import ParameterCoordinate
from grassmoth.artifacts.reports.histograms import empirical_points
from grassmoth.artifacts.reports.posterior_diagnostics import (
    DensityCurve,
    EnergyDiagnostics,
    ParetoKPoint,
    PosteriorMarginal,
    RankHistogram,
    TraceSeries,
)

if TYPE_CHECKING:
    import jax.numpy as jnp
    from numpy.typing import NDArray


type ParameterReferences = Mapping[ParameterCoordinate, tuple[str, ParameterRef] | None]

HIST_PADDING_RATIO = 0.05
HIST_PADDING_DEFAULT = 0.5


def sample_coordinates(
    samples: Mapping[str, jnp.ndarray], *, sample_dims: int
) -> Iterator[tuple[ParameterCoordinate, jnp.ndarray]]:
    """Yield scalar event coordinates while preserving the requested leading sampling dimensions."""
    for name, values in samples.items():
        for indices in np.ndindex(values.shape[sample_dims:]):
            coordinate = ParameterCoordinate(site_name=name, indices=indices)
            yield coordinate, values[(slice(None),) * sample_dims + indices]


def build_trace_data(
    chain_samples: Mapping[str, jnp.ndarray], references: ParameterReferences
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
                    chains=tuple(tuple(float(v) for v in chain) for chain in values),
                )
            )
    return tuple(traces)


def build_rank_histograms(
    chain_samples: Mapping[str, jnp.ndarray], references: ParameterReferences, n_bins: int = 20
) -> tuple[RankHistogram, ...]:
    """Build per-chain pooled-rank histograms for coordinates with scientific parameter references."""
    from arviz_stats.base.array import array_stats

    histograms = []
    for coordinate, values in sample_coordinates(chain_samples, sample_dims=2):
        reference = references[coordinate]
        if reference is None:
            continue
        _, subject = reference
        n_chains, n_samples = values.shape
        total = n_chains * n_samples
        ranks = np.asarray(
            array_stats.compute_ranks(np.asarray(values).reshape(-1), axis=-1)
        ).reshape(values.shape)
        counts, _ = array_stats.histogram(
            ranks, bins=n_bins, range=(1, total + 1), axis=1, density=False
        )
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


def _density_histogram(values: NDArray, n_bins: int) -> DensityCurve:
    from arviz_stats.base.array import array_stats

    v_min, v_max = float(np.min(values)), float(np.max(values))
    padding = (v_max - v_min) * HIST_PADDING_RATIO if v_max > v_min else HIST_PADDING_DEFAULT
    density, edges = array_stats.histogram(
        values, bins=n_bins, range=(v_min - padding, v_max + padding), axis=-1, density=True
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    return DensityCurve(
        x=tuple(float(v) for v in centers), density=tuple(float(v) for v in density)
    )


def param_marginal(
    parameter: str, subject: ParameterRef, values: jnp.ndarray, n_bins: int = 50
) -> PosteriorMarginal:
    """Summarize scalar posterior draws with a density histogram, moments, and a 94% HDI."""
    from arviz_stats.base.array import array_stats

    draws = np.asarray(values)
    histogram = _density_histogram(draws, n_bins)
    low, high = array_stats.hdi(draws, prob=0.94, axis=-1)
    return PosteriorMarginal(
        parameter=parameter,
        subject=subject,
        density_curve=histogram,
        empirical=empirical_points(draws),
        mean=float(np.mean(draws)),
        sd=float(np.std(draws)),
        interval_kind="hdi",
        interval_mass=0.94,
        lower=float(low),
        upper=float(high),
    )


def build_energy_diagnostics(energy: jnp.ndarray | NDArray, n_bins: int = 40) -> EnergyDiagnostics:
    """Summarize per-chain energy and consecutive energy changes, including each chain's BFMI."""
    from arviz_stats.base.array import array_stats

    chains = np.atleast_2d(np.asarray(energy))
    transitions = np.diff(chains, axis=1)
    return EnergyDiagnostics(
        energy_hist=_density_histogram(chains.reshape(-1), n_bins),
        energy_transition_hist=_density_histogram(transitions.reshape(-1), n_bins),
        bfmi=tuple(float(v) for v in array_stats.bfmi(chains, chain_axis=0, draw_axis=1)),
    )


def compute_posterior_marginals(
    samples: Mapping[str, jnp.ndarray], references: ParameterReferences, n_bins: int = 50
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
