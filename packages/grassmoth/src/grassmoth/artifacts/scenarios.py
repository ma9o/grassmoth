"""Interventions, their model-time assignments, and certified readouts of generated histories."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ConfigDict, Field, FiniteFloat

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import ConstructId, IndicatorId
from grassmoth.artifacts.observations.duration import Duration
from grassmoth.artifacts.reports.effects import EffectSummary


class InterventionSpec(Value):
    """Set a state some time after the design's start, then let its dynamics resume."""

    target: ConstructId
    after: Duration | None = Field(
        default=None, description="Offset from the design's start; omitted means at the start."
    )
    value: FiniteFloat


class StateAssignment(Value):
    """A state set at one model time: a resolved intervention or a deterministic input point."""

    target: ConstructId
    time: FiniteFloat = Field(description="Absolute time in model days.")
    value: FiniteFloat


class CausalEffectResult(Value):
    """Causal effects and realized trajectories under the enclosing report's design."""

    model_config = ConfigDict(allow_inf_nan=False)

    outcome: ConstructId
    labels: Mapping[ConstructId, str]
    differences: NumericalArray = Field(description="Paired outcome contrasts, [draw, time].")
    frame: tuple[FiniteFloat, FiniteFloat]
    summary: EffectSummary
    reference_mean: FiniteFloat = Field(description="Mean reference outcome at the final time.")
    manifest_effects: Mapping[IndicatorId, FiniteFloat] = Field(
        description="Mean final-time indicator contrasts where every paired draw is finite."
    )
    warnings: tuple[str, ...] = Field(default_factory=tuple)
