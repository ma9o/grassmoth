"""Revision-pinned plot coordinates; scientific reductions belong to the server."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import FiniteFloat

from nof1_causal_lab.artifacts.arrays import ScalarValues, resolve_vector
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import IndicatorId
from nof1_causal_lab.artifacts.reports.empirical import EmpiricalPoint


class ObservationHistory(Value):
    """All prepared observations, their true anchors and their measurement support."""

    times: tuple[FiniteFloat, ...]
    values: ScalarValues
    support_start: ScalarValues
    support_end: ScalarValues
    empirical: tuple[EmpiricalPoint, ...]

    def readings(self) -> tuple[tuple[float, float | None, float | None, float | None], ...]:
        """Read each saved anchor, value and support interval with native masks applied."""
        return tuple(
            zip(
                self.times,
                resolve_vector(self.values),
                resolve_vector(self.support_start),
                resolve_vector(self.support_end),
                strict=True,
            )
        )


type ObservationData = Mapping[IndicatorId, ObservationHistory]
