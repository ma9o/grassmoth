"""Revision-pinned plot coordinates; scientific reductions belong to the server."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import Field, model_validator

from grassmoth.artifacts.arrays import ArrayVector, NumericalArray, ScalarValues, resolve_vector
from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import IndicatorId
from grassmoth.artifacts.reports.empirical import EmpiricalDistribution


class ObservationHistory(Value):
    """All prepared observations, their true anchors and their measurement support."""

    times: NumericalArray = Field(
        description="Finite model-day anchors as a one-dimensional binary array."
    )
    values: ScalarValues
    support_start: ScalarValues
    support_end: ScalarValues
    empirical: EmpiricalDistribution

    @model_validator(mode="after")
    def aligned_vectors(self) -> ObservationHistory:
        """Establish aligned vector extents without admitting observation semantics."""
        shape, _ = self.times.layout
        if len(shape) != 1:
            raise ValueError("Observation times must be a vector")
        for values in (self.values, self.support_start, self.support_end):
            length = values.length if isinstance(values, ArrayVector) else len(values)
            if length != shape[0]:
                raise ValueError("Observation values and support must align with anchors")
        return self

    def readings(self) -> tuple[tuple[float, float | None, float | None, float | None], ...]:
        """Read each saved anchor, value and support interval with native masks applied."""
        return tuple(
            zip(
                (float(time) for time in self.times.values),
                resolve_vector(self.values),
                resolve_vector(self.support_start),
                resolve_vector(self.support_end),
                strict=True,
            )
        )


type ObservationData = Mapping[IndicatorId, ObservationHistory]
