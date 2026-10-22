"""Apply a model edit to supplied scientific values and retain its findings."""

from pydantic import ValidationError

from grassmoth.actions.checks import model_check_report
from grassmoth.actions.contracts.bodies import EditModelOutput
from grassmoth.artifacts.base import Value
from grassmoth.artifacts.model.dynamical_model_spec import (
    DynamicalModelSpec,
    apply_model_edit,
)
from grassmoth.artifacts.model.question import QuestionSpec
from grassmoth.models.causal.question_checks import question_edit_reason
from grassmoth.models.structure import StructuralSelection


class ModelEditFailure(Value):
    """An authored edit cannot supply a valid model for the question."""

    detail: str


def edit_model(
    question: QuestionSpec, parent: DynamicalModelSpec, edit: DynamicalModelSpec
) -> EditModelOutput | ModelEditFailure:
    """Merge, prune, validate and evaluate an edit without reading or writing study state."""
    try:
        edited = apply_model_edit(parent, edit, question.outcome)
    except ValidationError as exc:
        return ModelEditFailure(detail=str(exc))
    if (reason := question_edit_reason(edited.dynamical_model_spec, question)) is not None:
        return ModelEditFailure(detail=reason)
    selection = StructuralSelection.for_question(edited.dynamical_model_spec, question)
    return EditModelOutput(
        dynamical_model_spec=edited.dynamical_model_spec,
        checks=model_check_report(selection, question),
        identification=selection.identification,
        pruning=edited.pruning,
    )
