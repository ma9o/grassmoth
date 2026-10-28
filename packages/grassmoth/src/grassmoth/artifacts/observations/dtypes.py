"""Shared measurement-domain type aliases."""

from __future__ import annotations

from typing import Final, Literal

from grassmoth.artifacts.probability.families import DistributionFamily

type NumericDtype = Literal["continuous", "binary", "count"]
type LabeledDtype = Literal["ordinal", "categorical"]
type MeasurementDtype = NumericDtype | LabeledDtype


VALID_LIKELIHOODS_FOR_DTYPE: Final[dict[str, tuple[DistributionFamily, ...]]] = {
    "continuous": (
        DistributionFamily.GAUSSIAN,
        DistributionFamily.STUDENT_T,
        DistributionFamily.GAMMA,
        DistributionFamily.BETA,
        DistributionFamily.DELTA,
    ),
    "binary": (DistributionFamily.BERNOULLI, DistributionFamily.DELTA),
    "count": (
        DistributionFamily.POISSON,
        DistributionFamily.NEGATIVE_BINOMIAL,
        DistributionFamily.DELTA,
    ),
    "ordinal": (DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.DELTA),
    "categorical": (
        DistributionFamily.CATEGORICAL,
        DistributionFamily.ORDERED_LOGISTIC,
        DistributionFamily.DELTA,
    ),
}
