"""Static support of observation predictors and Gaussian noise."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CoordinateLoadings:
    """Each predictor reads at most one compiled state coordinate."""

    columns: tuple[int, ...]


@dataclass(frozen=True)
class DenseLoadings:
    """Predictors with multiple supported state coefficients use a matrix product."""


type ObservationLoadings = CoordinateLoadings | DenseLoadings


@dataclass(frozen=True)
class ObservationStructure:
    """Compiler-owned support, independent of the current parameter values."""

    loadings: ObservationLoadings
    covariance: Literal["diagonal", "dense"]
