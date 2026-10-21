"""Forward generation from a model's current laws and its recorded evidence."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime, time
from itertools import pairwise
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, FiniteFloat, model_validator

from nof1_causal_lab.artifacts.arrays import NumericalArray
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId
from nof1_causal_lab.artifacts.observations.duration import Duration
from nof1_causal_lab.artifacts.observations.spec import ResolvedObservationSpec
from nof1_causal_lab.artifacts.reports.checks import NotEvaluated, PredictiveAssessment
from nof1_causal_lab.artifacts.reports.predictive_provenance import PredictiveLawProvenance
from nof1_causal_lab.artifacts.scenarios import (
    CausalEffectResult,
    InterventionSpec,
    StateAssignment,
)


class SimulationSpec(Value):
    """Generate from a calendar day over a horizon, with interventions placed after the start.

    Authored initial states apply at the start. A retained trajectory law with a
    calendar origin places the start on that law's model-day axis.
    """

    start: date = Field(description="Calendar day the window starts, at 00:00 UTC.")
    horizon: Duration = Field(description="How long the window lasts, such as 9w or 61d.")
    interventions: tuple[InterventionSpec, ...] = ()

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        """Require interventions within the horizon and at most one assignment per state and time."""
        if any(
            event.after is not None and event.after.seconds >= self.horizon.seconds
            for event in self.interventions
        ):
            raise ValueError("Interventions must occur before the end of the horizon")
        offsets = {
            (event.target, event.after.seconds if event.after is not None else 0)
            for event in self.interventions
        }
        if len(offsets) != len(self.interventions):
            raise ValueError("A state can have only one intervention at each time")
        return self

    @property
    def start_instant(self) -> datetime:
        """Requested simulation start at midnight UTC."""
        return datetime.combine(self.start, time(), tzinfo=UTC)

    def start_day(self, origin: datetime) -> float:
        """The start in model days after the law's bound origin."""
        from nof1_causal_lab.artifacts.observations.time import ObservationInstant

        return ObservationInstant(self.start_instant).relative_to(ObservationInstant(origin)).days

    def end_day(self, origin: datetime) -> float:
        """Express the simulation horizon's endpoint as model days after the supplied origin."""
        return self.start_day(origin) + self.horizon.days

    def assignments(self, origin: datetime) -> tuple[StateAssignment, ...]:
        """Place every intervention in model days after the law's bound origin."""
        start = self.start_day(origin)
        return tuple(
            StateAssignment(
                target=event.target,
                time=start + event.after.days if event.after is not None else start,
                value=event.value,
            )
            for event in self.interventions
        )


class SimulationObservationLayout(Value):
    """Saved observation semantics and coordinates; generation truths remain separate."""

    variables: tuple[ResolvedObservationSpec, ...]
    support_start_times: NumericalArray
    support_end_times: NumericalArray
    mask: NumericalArray

    @property
    def indicator_ids(self) -> tuple[IndicatorId, ...]:
        """The observation axis is owned by the ordered variable definitions."""
        return tuple(item.id for item in self.variables)

    @model_validator(mode="after")
    def resolved_variables(self) -> Self:
        """Reject simulation layouts with repeated observation identities."""
        if len({item.id for item in self.variables}) != len(self.variables):
            raise ValueError("Simulation variables must have unique IDs")
        return self


class CategoryProbabilitySummary(Value):
    """Predictive probabilities for each declared level; unobserved anchors are null."""

    probabilities: Mapping[str, tuple[FiniteFloat | None, ...]]
    n_draws: tuple[int, ...]


type FitReliability = Literal["not_fitted", "converged", "unconverged", "unknown"]


class SimulationArm(Value):
    """One arm's exact state and observation draws."""

    latent_paths: NumericalArray
    observations: NumericalArray


class SingleArmSimulation(Value):
    """Natural-course generation without an intervention comparison."""

    kind: Literal["single"] = "single"
    action: SimulationArm


class PairedArmSimulation(Value):
    """Intervention and natural-course histories with matched draw indices."""

    kind: Literal["paired"] = "paired"
    action: SimulationArm
    reference: SimulationArm
    causal: CausalEffectResult | NotEvaluated[Literal["causal_effect"]]


type SimulationArms = Annotated[
    SingleArmSimulation | PairedArmSimulation, Field(discriminator="kind")
]


class SimulationSummary(Value):
    """Full-draw reductions retained once, independently of a viewer's draw selection."""

    state_frames: Mapping[ConstructId, tuple[FiniteFloat, FiniteFloat]]
    indicator_frames: Mapping[IndicatorId, tuple[FiniteFloat, FiniteFloat]]
    action_category_probabilities: Mapping[IndicatorId, CategoryProbabilitySummary]
    reference_category_probabilities: Mapping[IndicatorId, CategoryProbabilitySummary]


class SimulationEvidence(Value):
    """Exact generated histories with their production coordinates."""

    assignments: tuple[StateAssignment, ...]
    time_origin: AwareDatetime = Field(description="Calendar instant of model day zero.")
    times: tuple[FiniteFloat, ...] = Field(min_length=2)
    draws: int = Field(ge=1)
    seed: int = Field(ge=0)
    state_ids: tuple[ConstructId, ...]
    parameter_draws: Mapping[str, NumericalArray]
    arms: SimulationArms
    observation_layout: SimulationObservationLayout

    @model_validator(mode="after")
    def increasing_times(self) -> Self:
        """Own the generated histories' strictly increasing time axis."""
        if any(right <= left for left, right in pairwise(self.times)):
            raise ValueError("Simulation times must be strictly increasing")
        return self


class SimulationReport(Value):
    """Findings evaluated during simulation and retained with its exact evidence."""

    evidence: SimulationEvidence
    summary: SimulationSummary
    law: PredictiveLawProvenance
    findings: tuple[PredictiveAssessment, ...] = ()
    fit_reliability: FitReliability
