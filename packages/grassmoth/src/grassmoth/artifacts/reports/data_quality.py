"""Measurement validation findings and empirical profiles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import IndicatorId, IndicatorRef
from grassmoth.artifacts.reports.checks import Assessment

type DataFinding = Assessment[IndicatorRef | Literal["dataset"], str]


class IndicatorEmpiricalProfile(Value):
    """Count, recorded range, quartiles and mean of one indicator's usable observations."""

    n_obs: int
    min: float | None
    q25: float | None
    q50: float | None
    q75: float | None
    max: float | None
    mean: float | None


class IndicatorAudit(Value):
    """Empirical measurements and validation findings for one observation indicator."""

    profile: IndicatorEmpiricalProfile | None = None
    findings: tuple[DataFinding, ...]

    def with_finding(self, finding: DataFinding) -> Self:
        """Append a finding as a new audit, preserving the old value."""
        return self.revised(findings=(*self.findings, finding))


class DataProfileReport(Value):
    """Model-independent empirical measurements and data-quality findings."""

    indicators: Mapping[IndicatorId, IndicatorAudit]
    findings: tuple[DataFinding, ...]
