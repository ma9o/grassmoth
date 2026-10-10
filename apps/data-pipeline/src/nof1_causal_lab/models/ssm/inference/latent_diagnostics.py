"""Convergence evidence on every independent latent coordinate, grouped by construct."""

from dataclasses import dataclass

import numpy as np
from arviz_stats.sampling_diagnostics import ess, mcse, rhat
from numpy.typing import NDArray

from nof1_causal_lab.artifacts.identity import ConstructRef
from nof1_causal_lab.artifacts.posterior_diagnostics import LatentDiagnostics
from nof1_causal_lab.utils.immutability import freeze_fields


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
    import arviz_base as az_base

    paths = evidence.paths
    result = []
    for dimension, (subject, label) in enumerate(evidence.states):
        selected = evidence.free_mask[:, dimension]
        if not np.any(selected):
            continue
        samples = paths[:, :, selected, dimension]
        frozen = np.any(np.all(samples == samples[:, :1], axis=1))
        data = az_base.from_dict({"posterior": {"state": samples}})

        def worst(values: NDArray, *, largest: bool) -> float | None:
            if not np.all(np.isfinite(values)):
                return None
            return float(np.max(values) if largest else np.min(values))

        result.append(
            LatentDiagnostics(
                subject=subject,
                label=label,
                coordinates=int(np.sum(selected)),
                r_hat=None if frozen else worst(rhat(data)["state"].values, largest=True),
                ess_bulk=0.0
                if frozen
                else worst(ess(data, method="bulk")["state"].values, largest=False),
                ess_tail=0.0
                if frozen
                else worst(ess(data, method="tail")["state"].values, largest=False),
                mcse_mean=None
                if frozen
                else worst(mcse(data, method="mean")["state"].values, largest=True),
            )
        )
    return tuple(result)
