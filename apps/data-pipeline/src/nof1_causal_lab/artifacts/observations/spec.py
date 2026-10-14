"""Observed-variable identity, codebooks and time-support semantics."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import ConfigDict, Field, model_validator

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import IndicatorId
from nof1_causal_lab.artifacts.observations.dtypes import MeasurementDtype
from nof1_causal_lab.artifacts.observations.duration import (
    CalendarDuration,
    Duration,
    parse_observation_window,
)
from nof1_causal_lab.artifacts.observations.semantics import (
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


class ObservationDefinitionSpec(Value):
    """Measurement meaning, independent of variable identity and display spelling."""

    measurement_dtype: MeasurementDtype
    aggregation: SummaryOperator
    levels: tuple[str, ...]
    window: ObservationWindow


class ObservationSpec[WindowT: ObservationWindow | None](Value):
    """A stable observed variable, reusable across scientific model definitions."""

    model_config = ConfigDict(revalidate_instances="always")

    id: IndicatorId = Field(description="Persistent identity. Preserve when revising or renaming.")
    name: str = Field(description="Indicator name (e.g., 'hrv', 'self_reported_stress')")
    measurement_dtype: MeasurementDtype = Field(
        description="'continuous', 'binary', 'count', 'ordinal', 'categorical'"
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
    ordinal_levels: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Ordered list of level labels from lowest to highest for ordinal indicators "
            "(e.g., ['low', 'medium', 'high']). Required when measurement_dtype='ordinal' "
            "to ensure correct numeric encoding."
        ),
    )
    categorical_levels: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Exhaustive list of level labels for categorical indicators "
            "(e.g., ['home', 'work', 'other']). Required when "
            "measurement_dtype='categorical' to ensure correct numeric encoding."
        ),
    )

    def resolved(self, window: ObservationWindow) -> ResolvedObservationSpec:
        """Retain the observation definition with its owned, resolved window."""
        return ResolvedObservationSpec(
            id=self.id,
            name=self.name,
            measurement_dtype=self.measurement_dtype,
            aggregation=self.aggregation,
            observation_window=window,
            ordinal_levels=self.ordinal_levels,
            categorical_levels=self.categorical_levels,
        )

    @property
    def definition(self: ObservationSpec[ObservationWindow]) -> ObservationDefinitionSpec:
        """Resolved measurement equality; codebook order remains scientifically meaningful."""
        codebooks = (
            ("ordinal", self.ordinal_levels),
            ("categorical", self.categorical_levels),
        )
        return ObservationDefinitionSpec(
            measurement_dtype=self.measurement_dtype,
            aggregation=self.aggregation,
            levels=tuple(
                level
                for dtype, levels in codebooks
                if dtype == self.measurement_dtype and levels is not None
                for level in levels
            ),
            window=canonical_observation_window(self.observation_window),
        )

    @model_validator(mode="after")
    def validate_discrete_levels(self) -> Self:
        """Require at least two unique labels for ordinal and categorical indicators."""
        for dtype, levels in (
            ("ordinal", self.ordinal_levels),
            ("categorical", self.categorical_levels),
        ):
            if levels is not None and self.measurement_dtype != dtype:
                raise ValueError(f"{dtype}_levels requires measurement_dtype={dtype!r}")
        if self.measurement_dtype not in {"ordinal", "categorical"}:
            return self

        field_name = f"{self.measurement_dtype}_levels"
        levels = (
            self.ordinal_levels if self.measurement_dtype == "ordinal" else self.categorical_levels
        )
        label_description = (
            "ordered level labels" if self.measurement_dtype == "ordinal" else "level labels"
        )
        if not levels:
            raise ValueError(
                f"{field_name} is required when measurement_dtype={self.measurement_dtype!r} "
                f"(provide at least 2 {label_description})"
            )
        if len(levels) < 2:
            raise ValueError(f"{field_name} must have at least 2 items, got {len(levels)}")
        if len(levels) != len(set(levels)):
            raise ValueError(f"{field_name} must not contain duplicate labels")
        return self

    @model_validator(mode="after")
    def validate_observation_semantics(self) -> Self:
        """Reject aggregation/dtype combinations the measurement stack cannot model."""
        semantics = derive_indicator_observation_semantics(self.aggregation, self.measurement_dtype)
        if (
            isinstance(self.observation_window, CalendarDuration)
            and semantics.support_kind != SupportKind.INTERVAL
        ):
            raise ValueError(
                "Calendar windows require an interval summary: sum, count, mean or std"
            )
        return self

    def _observation_semantics(self) -> IndicatorObservationSemantics:
        return derive_indicator_observation_semantics(self.aggregation, self.measurement_dtype)

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
