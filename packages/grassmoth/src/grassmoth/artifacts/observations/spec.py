"""Observed-variable identity, codebooks and time-support semantics."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import IndicatorId
from grassmoth.artifacts.observations.dtypes import LabeledDtype, NumericDtype
from grassmoth.artifacts.observations.duration import (
    CalendarDuration,
    Duration,
    parse_observation_window,
)
from grassmoth.artifacts.observations.semantics import (
    AnchorPolicy,
    IndicatorObservationSemantics,
    SummaryOperator,
    SupportKind,
    derive_indicator_observation_semantics,
    supported_summary_operators_text,
)

type ObservationWindow = Duration | CalendarDuration


def canonical_observation_window(window: ObservationWindow | str) -> ObservationWindow:
    """Parse and compare fixed windows by elapsed time while retaining calendar meaning."""
    parsed = parse_observation_window(window) if isinstance(window, str) else window
    if isinstance(parsed, CalendarDuration):
        return CalendarDuration(f"{parsed.months}mo")
    return Duration(f"{parsed.seconds}s")


class NumericMeasurement(Value):
    """Readings recorded as numbers: real values, 0/1 outcomes or non-negative counts."""

    dtype: NumericDtype = Field(description="'continuous', 'binary' or 'count'")


class LabeledMeasurement(Value):
    """Readings recorded as labels, coded 0..K-1 in level order."""

    dtype: LabeledDtype = Field(description="'ordinal' or 'categorical'")
    levels: tuple[str, ...] = Field(
        min_length=2,
        description=(
            "At least two distinct level labels, coded 0..K-1 in this order. Ordinal levels run "
            "from lowest to highest (e.g., ['low', 'medium', 'high']); categorical levels list "
            "every category (e.g., ['home', 'work', 'other'])."
        ),
    )

    @field_validator("levels")
    @classmethod
    def distinct_levels(cls, levels: tuple[str, ...]) -> tuple[str, ...]:
        """Keep each code's label unique."""
        if len(levels) != len(set(levels)):
            raise ValueError("levels must not contain duplicate labels")
        return levels


type Measurement = Annotated[NumericMeasurement | LabeledMeasurement, Field(discriminator="dtype")]


class ObservationDefinitionSpec(Value):
    """Measurement meaning, independent of variable identity and display spelling."""

    measurement: Measurement
    aggregation: SummaryOperator
    window: ObservationWindow


class ObservationSpec[WindowT: ObservationWindow | None](Value):
    """A stable observed variable, reusable across scientific model definitions."""

    model_config = ConfigDict(revalidate_instances="always")

    id: IndicatorId = Field(description="Persistent identity. Preserve when revising or renaming.")
    name: str = Field(description="Indicator name (e.g., 'hrv', 'self_reported_stress')")
    measurement: Measurement = Field(
        description=(
            "How readings are recorded: {dtype: 'continuous' | 'binary' | 'count'}, or "
            "{dtype: 'ordinal' | 'categorical', levels: [...]} for labels coded 0..K-1."
        )
    )
    aggregation: SummaryOperator = Field(
        description=(
            "Aggregation function applied when bucketing raw extractions within the "
            "indicator support window. Supported operators: "
            f"{supported_summary_operators_text()}. A computed_rule must produce this same summary."
        ),
    )
    observation_window: WindowT = Field(
        description=(
            "Optional support window: positive fixed units s, m, h, d or w (for example '2w'), "
            "or whole calendar months/years parsed by pytimeparse2 "
            "(for example '3mo', '2 years', '1y6mo'). "
            "Calendar windows align to whole-month boundaries anchored to January 1970 "
            "and retain their actual lengths, including leap days. "
            "Resolved by the preparation window or the generative model clock."
        ),
    )

    def resolved(self, window: ObservationWindow) -> ResolvedObservationSpec:
        """Retain the observation definition with its owned, resolved window."""
        return ResolvedObservationSpec(
            id=self.id,
            name=self.name,
            measurement=self.measurement,
            aggregation=self.aggregation,
            observation_window=window,
        )

    @property
    def definition(self: ObservationSpec[ObservationWindow]) -> ObservationDefinitionSpec:
        """Resolved measurement equality; codebook order remains scientifically meaningful."""
        return ObservationDefinitionSpec(
            measurement=self.measurement,
            aggregation=self.aggregation,
            window=canonical_observation_window(self.observation_window),
        )

    @property
    def levels(self) -> tuple[str, ...]:
        """Labels in code order for labeled readings; numeric readings have none."""
        return self.measurement.levels if isinstance(self.measurement, LabeledMeasurement) else ()

    @model_validator(mode="after")
    def validate_observation_semantics(self) -> Self:
        """Reject aggregation/dtype combinations the measurement stack cannot model."""
        semantics = derive_indicator_observation_semantics(self.aggregation, self.measurement.dtype)
        if (
            isinstance(self.observation_window, CalendarDuration)
            and semantics.support_kind != SupportKind.INTERVAL
        ):
            raise ValueError(
                "Calendar windows require an interval summary: sum, count, mean or std"
            )
        return self

    def _observation_semantics(self) -> IndicatorObservationSemantics:
        return derive_indicator_observation_semantics(self.aggregation, self.measurement.dtype)

    @property
    def support_kind(self) -> SupportKind:
        """Whether this indicator is point-local or interval-summary."""
        return self._observation_semantics().support_kind

    @property
    def summary_operator(self) -> SummaryOperator:
        """Canonical summary operator used by extraction and likelihoods."""
        return self._observation_semantics().summary_operator

    @property
    def anchor_policy(self) -> AnchorPolicy:
        """Which support boundary receives the observation anchor."""
        return self._observation_semantics().anchor_policy

    @property
    def requires_interval_summary_measurement(self) -> bool:
        """Whether this indicator requires an interval-summary measurement equation."""
        return self.support_kind == SupportKind.INTERVAL


AuthoredObservationSpec = ObservationSpec[Annotated[ObservationWindow | None, Field(default=None)]]
ResolvedObservationSpec = ObservationSpec[ObservationWindow]
