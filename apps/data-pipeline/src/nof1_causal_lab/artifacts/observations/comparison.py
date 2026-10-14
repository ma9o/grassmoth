"""Parsed observation histories supplied to the scientific comparison engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import AwareDatetime, Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import GitOid, IndicatorId
from nof1_causal_lab.artifacts.observations.reference import DataRef
from nof1_causal_lab.artifacts.observations.spec import ResolvedObservationSpec
from nof1_causal_lab.artifacts.reports.data_comparison import DataPoint

if TYPE_CHECKING:
    from datetime import datetime


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
