"""Recorded empirical distribution coordinates shared by data and posterior reports."""

import numpy as np
from pydantic import Field, model_validator

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.base import Value


class EmpiricalDistribution(Value):
    """Aligned binary coordinates of an empirical cumulative distribution."""

    values: NumericalArray = Field(
        description="Distinct finite observed or sampled values in ascending order."
    )
    probabilities: NumericalArray = Field(
        description="Cumulative fractions of finite values less than or equal to each aligned value."
    )

    @model_validator(mode="after")
    def aligned_coordinates(self) -> "EmpiricalDistribution":
        """Require aligned finite vectors for the empirical CDF coordinates."""
        shape, _ = self.values.layout
        if len(shape) != 1 or shape != self.probabilities.layout[0]:
            raise ValueError("Empirical values and probabilities must be aligned vectors")
        if (
            not np.isfinite(self.values.values).all()
            or not np.isfinite(self.probabilities.values).all()
        ):
            raise ValueError("Empirical coordinates must be finite")
        return self
