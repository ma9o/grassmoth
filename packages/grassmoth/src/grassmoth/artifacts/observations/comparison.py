"""Parsed observation histories supplied to the scientific comparison engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import AwareDatetime, Field

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import GitOid, IndicatorId
from grassmoth.artifacts.observations.reference import DataRef
from grassmoth.artifacts.observations.spec import ResolvedObservationSpec
from grassmoth.artifacts.reports.data_comparison import DataPoint

if TYPE_CHECKING:
    from datetime import datetime

    from grassmoth.artifacts.observations.dataset import ObservationDataset


class DataSeries(Value):
    """One variable's recorded measurements in one history; no pooling across replicas."""

    variable: ResolvedObservationSpec
    points: tuple[DataPoint, ...]

    def measurement_support(self, point: DataPoint) -> tuple[datetime | None, datetime | None]:
        """Use model-owned point/interval semantics independently of the extraction window."""
        if self.variable.requires_interval_summary_measurement:
            return point.support_start, point.support_end
        return point.anchor_time, point.anchor_time


class Dataset(Value):
    """One parsed observation history, identified by its immutable source."""

    source: DataRef[GitOid, int]
    time_origin: AwareDatetime = Field(description="Calendar instant of model day zero.")
    series: Mapping[IndicatorId, DataSeries]

    @classmethod
    def from_observations(
        cls, source: DataRef[GitOid, int], observations: ObservationDataset
    ) -> Dataset:
        """Project one parsed history with its exact source reference."""
        import polars as pl

        frame = observations.recorded.frame.with_columns(
            pl.col("anchor_time", "support_start", "support_end").dt.replace_time_zone("UTC")
        )
        series = {
            variable.id: DataSeries(
                variable=variable,
                points=tuple(
                    DataPoint(anchor_time=anchor, support_start=start, support_end=end, value=value)
                    for anchor, start, end, value in frame.filter(
                        pl.col("indicator_id") == variable.id
                    )
                    .select("anchor_time", "support_start", "support_end", "value")
                    .iter_rows()
                ),
            )
            for variable in observations.variables
        }

        return cls(source=source, time_origin=observations.time_origin, series=series)
