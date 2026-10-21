"""Check the study question against the model and the prepared record, without simulating."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import polars as pl

from nof1_causal_lab.artifacts.identity import ConstructRef
from nof1_causal_lab.artifacts.model.construct import Role, TemporalStatus
from nof1_causal_lab.artifacts.observations.time import ObservationInstant
from nof1_causal_lab.artifacts.reports.checks import (
    Evaluated,
    NotEvaluated,
    OutcomeSubject,
    QueryTargetSubject,
    QueryWindowSubject,
)
from nof1_causal_lab.models.causal.graphs import get_all_treatments
from nof1_causal_lab.models.ssm.preparation.binding import reading_level
from nof1_causal_lab.models.structure import (
    StructuralSelection,
    StructuralSelectionError,
    selected_state_ids,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime

    from nof1_causal_lab.artifacts.identity import ConstructId
    from nof1_causal_lab.artifacts.model.construct import ConstructSpec
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from nof1_causal_lab.artifacts.model.question import QuestionSpec
    from nof1_causal_lab.artifacts.observations.dataset import ObservationDataset
    from nof1_causal_lab.artifacts.reports.identification import IdentificationReport
    from nof1_causal_lab.artifacts.reports.model_checks import QuestionAssessment
    from nof1_causal_lab.artifacts.simulation import SimulationSpec


def question_findings(
    question: QuestionSpec,
    selection: StructuralSelection,
    *,
    history: ObservationDataset | None,
) -> tuple[QuestionAssessment, ...]:
    """Each check names what the model or the record still lacks for one query."""
    dynamical_model_spec = selection.dynamical_model_spec
    constructs = {construct.id: construct for construct in dynamical_model_spec.constructs}
    if history is not None:
        panel = history.recorded.frame
        return tuple(
            finding
            for name, query in question.queries.items()
            for finding in (
                _window_finding(name, query, panel, history.time_origin),
                *(
                    _range_finding(name, target, query, constructs.get(target), panel)
                    for target in sorted({event.target for event in query.interventions})
                ),
            )
        )
    identification = selection.identification
    states = frozenset(selected_state_ids(selection))
    outcome = _outcome_finding(question, constructs)
    findings: list[QuestionAssessment] = [outcome]
    for name, query in question.queries.items():
        targets = sorted({event.target for event in query.interventions})
        findings.extend(
            _target_finding(name, target, constructs.get(target), states) for target in targets
        )
        findings.extend(
            _identification_finding(
                name, target, outcome, constructs, dynamical_model_spec, identification, states
            )
            for target in targets
        )
    return tuple(findings)


def _outcome_finding(
    question: QuestionSpec, constructs: dict[ConstructId, ConstructSpec]
) -> QuestionAssessment:
    subject = OutcomeSubject(outcome=ConstructRef(id=question.outcome))
    construct = constructs.get(question.outcome)
    if construct is None:
        return NotEvaluated(
            code="outcome",
            subject=subject,
            reason="CONSTRUCT_UNDEFINED",
            detail="The model does not define the question's outcome yet.",
        )
    problems = [
        problem
        for failed, problem in (
            (
                construct.role != Role.ENDOGENOUS,
                "it is exogenous, supplied by a deterministic trajectory",
            ),
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
    query: str, target: ConstructId, construct: ConstructSpec | None, states: frozenset[ConstructId]
) -> QuestionAssessment:
    subject = QueryTargetSubject(query=query, target=ConstructRef(id=target))
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
    query: str,
    target: ConstructId,
    outcome: QuestionAssessment,
    constructs: dict[ConstructId, ConstructSpec],
    dynamical_model_spec: DynamicalModelSpec,
    identification: IdentificationReport,
    states: frozenset[ConstructId],
) -> QuestionAssessment:
    subject = QueryTargetSubject(query=query, target=ConstructRef(id=target))
    construct = constructs.get(target)
    if construct is None or not isinstance(outcome, Evaluated):
        return NotEvaluated(
            code="identification",
            subject=subject,
            reason="CONSTRUCT_UNDEFINED",
            detail="Identification waits until the model defines the outcome and this target.",
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
        evidence=f"{effect} is not identified"
        + (f"; it is confounded by {confounders}" if confounders else "")
        + (f". {status.notes}" if status.notes else "."),
    )


def _window_finding(
    query: str, design: SimulationSpec, panel: pl.DataFrame, time_origin: datetime
) -> QuestionAssessment:
    subject = QueryWindowSubject(query=query)
    if design.start_instant < time_origin:
        return Evaluated(
            code="window",
            subject=subject,
            outcome="failed",
            evidence=f"It starts on {design.start}, before the record begins on "
            f"{time_origin.date()}; the initial-state law applies only from then.",
        )
    record_end = ObservationInstant(panel.select(pl.col("support_end").max()).item()).value
    window_end = design.start_instant + timedelta(seconds=design.horizon.seconds)
    return Evaluated(
        code="window",
        subject=subject,
        outcome="passed",
        evidence="It lies inside the record, so its reference is the recorded course."
        if window_end <= record_end
        else f"It runs past the record's end on {record_end.date()}; from there, inputs hold "
        "their last reading in both arms."
        if design.start_instant <= record_end
        else f"It starts after the record ends on {record_end.date()}, so the model forecasts "
        "the gap while inputs hold their last reading.",
    )


def _range_finding(
    query: str,
    target: ConstructId,
    design: SimulationSpec,
    construct: ConstructSpec | None,
    panel: pl.DataFrame,
) -> QuestionAssessment:
    subject = QueryTargetSubject(query=query, target=ConstructRef(id=target))
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
    levels = tuple(_recorded_levels(construct, panel))
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


def _recorded_levels(construct: ConstructSpec, panel: pl.DataFrame) -> Iterator[float]:
    for indicator in construct.indicators:
        rows = panel.filter(pl.col("indicator_id") == indicator.observation.id).drop_nulls("value")
        for row in rows.iter_rows(named=True):
            yield reading_level(
                float(row["value"]),
                row["support_start"],
                row["support_end"],
                indicator.observation.summary_operator,
            )


def question_edit_reason(
    dynamical_model_spec: DynamicalModelSpec, question: QuestionSpec
) -> str | None:
    """Reject edits that cannot answer the question or violate scoped parameter anchors."""
    from nof1_causal_lab.artifacts.model.construct import Role
    from nof1_causal_lab.models.causal.graphs import get_all_treatments

    if missing := sorted(
        identity
        for identity in {question.outcome, *question.targets}
        if identity not in dynamical_model_spec._constructs
    ):
        return "The model must define the question's constructs: " + ", ".join(missing)
    outcome_construct = dynamical_model_spec.get_construct(question.outcome)
    if outcome_construct.role != Role.ENDOGENOUS:
        return "The question's outcome must reference an endogenous construct"
    ancestors = frozenset(
        get_all_treatments(
            dynamical_model_spec.constructs, dynamical_model_spec.edges, question.outcome
        )
    )
    outside = sorted(
        f"{construct.name!r} ({construct.id})"
        for construct in dynamical_model_spec.constructs
        if construct.id != question.outcome and construct.name not in ancestors
    )
    if outside:
        return (
            "Every construct must have a directed path to the question's outcome "
            f"{outcome_construct.name!r} ({outcome_construct.id}). "
            "No directed path: " + ", ".join(outside)
        )
    try:
        StructuralSelection(dynamical_model_spec, question.outcome)
    except StructuralSelectionError as exc:
        return str(exc)
    return None
