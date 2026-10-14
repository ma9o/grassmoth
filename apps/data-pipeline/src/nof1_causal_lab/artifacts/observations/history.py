"""Revision-pinned plot coordinates; scientific reductions belong to the server."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import FiniteFloat

from nof1_causal_lab.artifacts.arrays import ScalarValues
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


type ObservationData = Mapping[IndicatorId, ObservationHistory]
