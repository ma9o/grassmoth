"""Deterministic semantic helpers shared across model-spec and SSM compilation."""

from __future__ import annotations

from grassmoth.artifacts.model.likelihood import LinkFunction
from grassmoth.artifacts.probability.families import DistributionFamily

_NONSTANDARDIZABLE_SCALAR_FAMILIES = frozenset(
    {
        DistributionFamily.POISSON,
        DistributionFamily.NEGATIVE_BINOMIAL,
        DistributionFamily.BERNOULLI,
        DistributionFamily.GAMMA,
        DistributionFamily.BETA,
    }
)
_LOCATION_FAMILIES = frozenset({DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T})
_THRESHOLD_FAMILIES = frozenset(
    {
        DistributionFamily.ORDERED_LOGISTIC,
        DistributionFamily.CATEGORICAL,
    }
)


def indicator_requires_observation_intercept(
    distribution: DistributionFamily,
    link: LinkFunction,
    *,
    standardized: bool,
) -> bool:
    """Return whether a manifest channel needs a free observation intercept."""
    family = distribution

    if family in _THRESHOLD_FAMILIES:
        return False

    if family in _NONSTANDARDIZABLE_SCALAR_FAMILIES:
        return True

    if (
        family in _LOCATION_FAMILIES or family == DistributionFamily.DELTA
    ) and link == LinkFunction.IDENTITY:
        return not standardized

    return False
