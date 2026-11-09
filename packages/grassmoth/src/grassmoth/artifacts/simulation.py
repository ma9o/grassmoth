"""Forward generation from a model's current laws and its recorded evidence."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import AwareDatetime, Field, FiniteFloat, model_validator

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import ConstructId, IndicatorId
from grassmoth.artifacts.model.time import ModelTimeSpec, TimeGridFailure, TimeWindowSpec
from grassmoth.artifacts.observations.spec import ResolvedObservationSpec
from grassmoth.artifacts.reports.checks import NotEvaluated, PredictiveAssessment
from grassmoth.artifacts.reports.predictive_provenance import PredictiveLawProvenance
from grassmoth.artifacts.scenarios import (
    CausalEffectResult,
    InterventionSpec,
    StateAssignment,
)

if TYPE_CHECKING:
    from datetime import datetime


class SimulationSpec(Value):
    """Generate from a calendar day over a horizon, with interventions placed after the start.

    Initial states apply at the model window's start. Query windows use the
    model's fixed grid and may extend beyond its fitting window.
    """

    window: TimeWindowSpec
    interventions: tuple[InterventionSpec, ...] = ()

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        """Require interventions within the horizon and at most one assignment per state and time."""
        if any(
            event.after is not None and event.after.seconds >= self.window.horizon.seconds
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

    def grid(self, time: ModelTimeSpec) -> tuple[float, ...] | TimeGridFailure:
        """Admit a simulation on the model's fixed grid, preserving exact event times."""
        if any(
            event.after is not None and event.after.seconds % time.step.seconds
            for event in self.interventions
        ):
            return TimeGridFailure(
                reason="Intervention offsets must align with the model time step"
            )
        return time.query_grid(self.window)

    def assignments(self, origin: datetime) -> tuple[StateAssignment, ...]:
        """Place every intervention in model days after the law's bound origin."""
        start = self.window.start_day(origin)
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
        layout = self.observation_layout
        support_shape = (len(self.times), len(layout.variables))
        observation_shape = (self.draws, *support_shape)
        for bounds in (layout.support_start_times, layout.support_end_times):
            shape, dtype = bounds.layout
            if shape != support_shape or dtype.kind != "f":
                raise ValueError(
                    "Simulation support must be floating arrays on the observation layout"
                )
        shape, dtype = layout.mask.layout
        if shape != observation_shape or dtype.kind != "b":
            raise ValueError("Simulation observation mask must be boolean and align with draws")
        arms = (
            (self.arms.action, self.arms.reference)
            if isinstance(self.arms, PairedArmSimulation)
            else (self.arms.action,)
        )
        for arm in arms:
            shape, dtype = arm.observations.layout
            if shape != observation_shape or dtype.kind != "f":
                raise ValueError(
                    "Simulation observations must be floating arrays on the recorded draws and design"
                )
        return self


class SimulationReport(Value):
    """Findings evaluated during simulation and retained with its exact evidence."""

    evidence: SimulationEvidence
    summary: SimulationSummary
    law: PredictiveLawProvenance
    findings: tuple[PredictiveAssessment, ...] = ()
