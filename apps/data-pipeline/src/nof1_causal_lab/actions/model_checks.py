"""Evaluate only the scientific findings owned by the executing action."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from nof1_causal_lab.actions.checks import check_model_data, check_specification
from nof1_causal_lab.artifacts.reports.inference import FitCheckReport
from nof1_causal_lab.artifacts.reports.model_checks import ModelCheckReport, QuestionCheckReport
from nof1_causal_lab.models.causal.question_checks import question_findings
from nof1_causal_lab.models.ssm.compile.inputs import compile_fit_inputs, compile_model
from nof1_causal_lab.models.structure import StructuralSelection
from nof1_causal_lab.observations.quality.profile import validate_extraction
from nof1_causal_lab.study.persistence.store import ArtifactStore, read_model, read_question
from nof1_causal_lab.study.queries.data import read_data_history
from nof1_causal_lab.study.state import apply_effects

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.reports.identification import IdentificationReport
    from nof1_causal_lab.study.records import Applied
    from nof1_causal_lab.study.state import StudyState


@overload
def read_model_checks(
    workspace_id: str, state: StudyState, *, action: Literal["edit_model"]
) -> tuple[ModelCheckReport, IdentificationReport]: ...


@overload
def read_model_checks(
    workspace_id: str, state: StudyState, *, action: Literal["fit"]
) -> FitCheckReport: ...


def read_model_checks(
    workspace_id: str,
    state: StudyState,
    *,
    action: Literal["edit_model", "fit"],
) -> tuple[ModelCheckReport, IdentificationReport] | FitCheckReport:
    """Retain model findings for an edit and record-dependent findings for a fit."""
    from nof1_causal_lab.bootstrap import initialize_numerical_runtime

    initialize_numerical_runtime()
    store = ArtifactStore(workspace_id)
    dynamical_model_spec = read_model(store, state.current["model"].revision)
    question = read_question(store, state.current["question"].revision)
    selection = StructuralSelection.for_question(dynamical_model_spec, question)
    compiled = compile_model(selection)
    inputs = compile_fit_inputs(compiled, selection)
    if action == "fit":
        assert state.data is not None, "A completed fit owns one selected history"
        history = read_data_history(store, state.data)
        return FitCheckReport(
            data=validate_extraction(dynamical_model_spec, [history.observations.recorded.frame]),
            preflight=check_model_data(
                inputs, history.observations, time_origin=history.time_origin
            ),
            question=QuestionCheckReport(
                findings=question_findings(question, selection, history=history.observations),
            ),
        )
    return (
        ModelCheckReport(
            specification=check_specification(compiled, inputs),
            question=QuestionCheckReport(
                findings=question_findings(question, selection, history=None),
            ),
        ),
        selection.identification,
    )


@overload
def evaluate_model_checks[ResultT](
    workspace_id: str,
    state: StudyState,
    applied: Applied[ResultT],
    *,
    action: Literal["edit_model"],
) -> tuple[ModelCheckReport, IdentificationReport]: ...


@overload
def evaluate_model_checks[ResultT](
    workspace_id: str, state: StudyState, applied: Applied[ResultT], *, action: Literal["fit"]
) -> FitCheckReport: ...


def evaluate_model_checks[ResultT](
    workspace_id: str,
    state: StudyState,
    applied: Applied[ResultT],
    *,
    action: Literal["edit_model", "fit"],
) -> tuple[ModelCheckReport, IdentificationReport] | FitCheckReport:
    """Check the edited model, or the selected inputs that a fit actually consumed."""
    return read_model_checks(
        workspace_id,
        state
        if action == "fit"
        else apply_effects(state, applied.effects.produced, applied.effects.retracted),
        action=action,
    )
