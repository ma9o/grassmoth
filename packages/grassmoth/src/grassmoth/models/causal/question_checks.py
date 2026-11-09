"""Check the study question against the model and the prepared record, without simulating."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np

from grassmoth.artifacts.arrays import vector_values
from grassmoth.artifacts.identity import ConstructRef
from grassmoth.artifacts.model.construct import Role, TemporalStatus
from grassmoth.artifacts.observations.time import ModelTime, ObservationInstant
from grassmoth.artifacts.reports.checks import (
    Evaluated,
    NotEvaluated,
    OutcomeSubject,
    QueryTargetSubject,
    QueryWindowSubject,
)
from grassmoth.models.causal.graphs import get_all_treatments
from grassmoth.models.ssm.preparation.binding import reading_level
from grassmoth.models.structure import (
    StructuralSelection,
    selected_state_ids,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from grassmoth.artifacts.identity import ConstructId
    from grassmoth.artifacts.model.construct import ConstructSpec
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth.artifacts.observations.dataset import ObservationDataset
    from grassmoth.artifacts.reports.model_checks import QuestionAssessment
    from grassmoth.artifacts.simulation import SimulationSpec


def question_findings(
    selection: StructuralSelection,
    *,
    history: ObservationDataset | None,
) -> tuple[QuestionAssessment, ...]:
    """Each check names what the model or the record still lacks for one query."""
    dynamical_model_spec = selection.dynamical_model_spec
    constructs = {construct.id: construct for construct in dynamical_model_spec.constructs}
    query = dynamical_model_spec.core_query
    targets = sorted({event.target for event in query.interventions})
    if history is not None:
        return (
            _window_finding(query, history),
            *(_range_finding(target, query, constructs.get(target), history) for target in targets),
        )
    states = frozenset(selected_state_ids(selection))
    return (
        _outcome_finding(dynamical_model_spec),
        *(_target_finding(target, constructs.get(target), states) for target in targets),
        *(_identification_finding(target, selection) for target in targets),
    )


def _outcome_finding(model: DynamicalModelSpec) -> QuestionAssessment:
    subject = OutcomeSubject(outcome=ConstructRef(id=model.outcome))
    construct = model.get_construct(model.outcome)
    problems = [
        problem
        for failed, problem in (
            (
                construct.temporal_status == TemporalStatus.TIME_INVARIANT,
                "it is time-invariant, so it has no course to contrast",
            ),
            (not construct.indicators, "it has no indicators, so the model can't follow it"),
        )
        if failed
    ]
    return Evaluated(
        code="outcome",
        subject=subject,
        outcome="failed" if problems else "passed",
        evidence=f"{construct.name}: " + "; ".join(problems)
        if problems
        else f"{construct.name} is a measured, modeled course.",
    )


def _target_finding(
    target: ConstructId, construct: ConstructSpec | None, states: frozenset[ConstructId]
) -> QuestionAssessment:
    subject = QueryTargetSubject(target=ConstructRef(id=target))
    if construct is None:
        return NotEvaluated(
            code="target",
            subject=subject,
            reason="CONSTRUCT_UNDEFINED",
            detail="The model does not define this intervention target yet.",
        )
    if target not in states:
        return Evaluated(
            code="target",
            subject=subject,
            outcome="failed",
            evidence=f"{construct.name} has no indicators, so it is not a state a simulation can set.",
        )
    return Evaluated(
        code="target",
        subject=subject,
        outcome="passed",
        evidence=f"{construct.name} is a state a simulation can set.",
    )


def _identification_finding(
    target: ConstructId, selection: StructuralSelection
) -> QuestionAssessment:
    dynamical_model_spec, identification = selection.dynamical_model_spec, selection.identification
    constructs = {construct.id: construct for construct in dynamical_model_spec.constructs}
    states = frozenset(selected_state_ids(selection))
    subject = QueryTargetSubject(target=ConstructRef(id=target))
    construct = constructs.get(target)
    if construct is None:
        return NotEvaluated(
            code="identification",
            subject=subject,
            reason="CONSTRUCT_UNDEFINED",
            detail="Identification waits until the model defines this target.",
        )
    assert identification.outcome is not None
    effect = f"The effect of {construct.name} on {constructs[identification.outcome].name}"
    status = identification.treatments.get(target)
    if status is None:
        ancestors = get_all_treatments(
            dynamical_model_spec.constructs, dynamical_model_spec.edges, identification.outcome
        )
        return Evaluated(
            code="identification",
            subject=subject,
            outcome="failed",
            evidence=f"{effect} is not checked: "
            + (
                f"{construct.name} has no directed path to the outcome."
                if construct.name not in ancestors
                else f"{construct.name} is unmeasured."
                if target not in states
                else "the outcome is unmeasured."
            ),
        )
    if status.status == "identified":
        return Evaluated(
            code="identification",
            subject=subject,
            outcome="passed",
            evidence=f"{effect} is identified: {status.estimand}",
        )
    confounders = ", ".join(constructs[identity].name for identity in status.confounders)
    return Evaluated(
        code="identification",
        subject=subject,
        outcome="failed",
        evidence=f"{effect} is not identified: the outcome is unobserved."
        if identification.outcome in status.confounders
        else f"{effect} is not identified"
        + (f"; it is confounded by {confounders}." if confounders else "."),
    )


def _window_finding(design: SimulationSpec, history: ObservationDataset) -> QuestionAssessment:
    subject = QueryWindowSubject()
    time_origin = history.time_origin
    ends = [
        float(end)
        for series in history.series.values()
        for end in vector_values(series.history.support_end)
        if np.isfinite(end)
    ]
    if not ends:
        return NotEvaluated(
            code="window",
            subject=subject,
            reason="NO_OBSERVATIONS",
            detail="The record has no known measurement support.",
        )
    record_end = ModelTime(max(ends)).at(ObservationInstant(time_origin)).value
    window_end = design.window.start_instant + timedelta(seconds=design.window.horizon.seconds)
    return Evaluated(
        code="window",
        subject=subject,
        outcome="passed",
        evidence="It lies inside the record, so its reference is the recorded course."
        if window_end <= record_end
        else f"It runs past the record's end on {record_end.date()}; from there, inputs hold "
        "their last reading in both arms."
        if design.window.start_instant <= record_end
        else f"It starts after the record ends on {record_end.date()}, so the model forecasts "
        "the gap while inputs hold their last reading.",
    )


def _range_finding(
    target: ConstructId,
    design: SimulationSpec,
    construct: ConstructSpec | None,
    history: ObservationDataset,
) -> QuestionAssessment:
    subject = QueryTargetSubject(target=ConstructRef(id=target))
    if construct is None:
        return NotEvaluated(
            code="range",
            subject=subject,
            reason="CONSTRUCT_UNDEFINED",
            detail="The model does not define this intervention target yet.",
        )
    if construct.role != Role.EXOGENOUS:
        return NotEvaluated(
            code="range",
            subject=subject,
            reason="STATE_NOT_RECORDED",
            detail="The record measures this state only through its indicators.",
        )
    levels = tuple(_recorded_levels(construct, history))
    if not levels:
        return Evaluated(
            code="range",
            subject=subject,
            outcome="failed",
            evidence=f"{construct.name} has no readings in the record.",
        )
    low, high = min(levels), max(levels)
    recorded = (
        f"the record holds it at {low:g} throughout"
        if low == high
        else f"the recorded range is {low:g} to {high:g}"
    )
    outside = sorted(
        {
            event.value
            for event in design.interventions
            if event.target == target and not low <= event.value <= high
        }
    )
    if outside:
        *rest, last = (f"{value:g}" for value in outside)
        values = f"{', '.join(rest)} and {last}" if rest else last
        return Evaluated(
            code="range",
            subject=subject,
            outcome="failed",
            evidence=f"It sets {construct.name} to {values}, but {recorded}; "
            "the answer there rests on the model's assumptions.",
        )
    return Evaluated(
        code="range",
        subject=subject,
        outcome="passed",
        evidence=f"Every value is inside the record: {recorded}.",
    )


def _recorded_levels(construct: ConstructSpec, history: ObservationDataset) -> Iterator[float]:
    for indicator in construct.indicators:
        series = history.series.get(indicator.observation.id)
        if series is None:
            continue
        values = vector_values(series.history.values)
        starts = vector_values(series.history.support_start)
        ends = vector_values(series.history.support_end)
        for value, start, end in zip(values, starts, ends, strict=True):
            if not np.isnan(value):
                yield reading_level(
                    float(value), float(start), float(end), indicator.observation.summary_operator
                )
