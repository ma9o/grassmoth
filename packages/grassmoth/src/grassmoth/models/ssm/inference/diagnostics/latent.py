"""Convergence evidence on every independent latent coordinate, grouped by construct."""

from dataclasses import dataclass
from typing import Self

import jax
import numpy as np
from numpy.typing import NDArray

from grassmoth.artifacts.identity import ConstructRef
from grassmoth.artifacts.immutability import freeze_fields
from grassmoth.artifacts.probability.joint_layout import JointLawLayout
from grassmoth.artifacts.reports.posterior_diagnostics import LatentDiagnostics
from grassmoth.models.ssm.inference.diagnostics.chain_stats import (
    chain_statistic_batch_size,
    chain_statistics,
)


def joint_latent_paths(layout: JointLawLayout, atoms: NDArray, chains: int) -> NDArray:
    """View the owned joint columns as paths without stacking a second full history."""
    return (
        atoms[:, len(layout.parameter_columns) :]
        .reshape(chains, atoms.shape[0] // chains, len(layout.constructs), len(layout.time_points))
        .swapaxes(-1, -2)
    )


def _selected_statistics(paths: NDArray, dimension: int, selected: NDArray) -> NDArray | None:
    """Gather and check only one coordinate batch; a frozen chain fails the construct."""
    batch_size = chain_statistic_batch_size(*paths.shape[:2])
    batches = []
    for start in range(0, selected.size, batch_size):
        # Coordinate selection happens only after the complete paths have a
        # host owner, so even tiny samplers need no reporting GPU workspace.
        samples = paths[:, :, selected[start : start + batch_size], dimension]
        if np.any(np.all(samples == samples[:, :1], axis=1)):
            return None
        batches.append(chain_statistics(samples))
    return np.concatenate(batches, axis=0)


@dataclass(frozen=True)
class LatentChainEvidence:
    """Host-owned paths and full-chain summaries computed after transfer."""

    paths: NDArray
    free_mask: NDArray
    states: tuple[tuple[ConstructRef, str], ...]
    diagnostics: tuple[LatentDiagnostics, ...]

    @classmethod
    def from_draws(
        cls,
        paths: jax.Array | NDArray,
        free_mask: NDArray,
        states: tuple[tuple[ConstructRef, str], ...],
    ) -> Self:
        """Transfer paths once, then reduce bounded host batches without device gathers."""
        host_paths = np.asarray(paths)
        diagnostics = _latent_chain_diagnostics(host_paths, free_mask, states)
        return cls(host_paths, free_mask, states, diagnostics)

    def __post_init__(self) -> None:
        """Share immutable host evidence between conditioning and report consumers."""
        freeze_fields(self)


def _latent_chain_diagnostics(
    paths: NDArray,
    free_mask: NDArray,
    states: tuple[tuple[ConstructRef, str], ...],
) -> tuple[LatentDiagnostics, ...]:
    """Report worst-time rank R-hat/ESS and explicitly fail frozen independent chains."""
    result = []
    for dimension, (subject, label) in enumerate(states):
        selected = np.flatnonzero(free_mask[:, dimension])
        if not selected.size:
            continue
        metrics = _selected_statistics(paths, dimension, selected)

        def worst(values: NDArray, *, largest: bool) -> float | None:
            if not np.all(np.isfinite(values)):
                return None
            return float(np.max(values) if largest else np.min(values))

        result.append(
            LatentDiagnostics(
                subject=subject,
                label=label,
                coordinates=selected.size,
                r_hat=None if metrics is None else worst(metrics[:, 0], largest=True),
                ess_bulk=0.0 if metrics is None else worst(metrics[:, 1], largest=False),
                ess_tail=0.0 if metrics is None else worst(metrics[:, 2], largest=False),
                mcse_mean=None if metrics is None else worst(metrics[:, 3], largest=True),
            )
        )
    return tuple(result)
