"""Convergence evidence on every independent latent coordinate, grouped by construct."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nof1_causal_lab.artifacts.identity import ConstructRef
from nof1_causal_lab.artifacts.immutability import freeze_fields
from nof1_causal_lab.artifacts.reports.posterior_diagnostics import LatentDiagnostics
from nof1_causal_lab.models.ssm.inference.diagnostics.chain_stats import chain_statistics


@dataclass(frozen=True)
class LatentChainEvidence:
    """Post-warmup paths and the compiler's independent, unobserved coordinates."""

    paths: NDArray
    free_mask: NDArray
    states: tuple[tuple[ConstructRef, str], ...]

    def __post_init__(self) -> None:
        """Own immutable numerical evidence before projecting scalar diagnostics."""
        freeze_fields(self)


def latent_chain_diagnostics(evidence: LatentChainEvidence) -> tuple[LatentDiagnostics, ...]:
    """Report worst-time rank R-hat/ESS and explicitly fail frozen independent chains."""
    paths = evidence.paths
    result = []
    for dimension, (subject, label) in enumerate(evidence.states):
        selected = evidence.free_mask[:, dimension]
        if not np.any(selected):
            continue
        samples = paths[:, :, selected, dimension]
        frozen = np.any(np.all(samples == samples[:, :1], axis=1))
        metrics = None if frozen else chain_statistics(samples)

        def worst(values: NDArray, *, largest: bool) -> float | None:
            if not np.all(np.isfinite(values)):
                return None
            return float(np.max(values) if largest else np.min(values))

        result.append(
            LatentDiagnostics(
                subject=subject,
                label=label,
                coordinates=int(np.sum(selected)),
                r_hat=None if metrics is None else worst(metrics[:, 0], largest=True),
                ess_bulk=0.0 if metrics is None else worst(metrics[:, 1], largest=False),
                ess_tail=0.0 if metrics is None else worst(metrics[:, 2], largest=False),
                mcse_mean=None if metrics is None else worst(metrics[:, 3], largest=True),
            )
        )
    return tuple(result)
