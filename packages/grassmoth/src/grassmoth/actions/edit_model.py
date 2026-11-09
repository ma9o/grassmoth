"""Create or revise a self-contained model and retain its scientific findings."""

from pydantic import ValidationError

from grassmoth.actions.checks import model_check_report
from grassmoth.actions.contracts.io import EditModelOutput
from grassmoth.artifacts.base import Value
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec, apply_model_edit
from grassmoth.models.structure import StructuralSelection, StructuralSelectionError


class ModelEditFailure(Value):
    """An authored document cannot supply a valid model."""

    detail: str


def edit_model(
    parent: DynamicalModelSpec | None, edit: DynamicalModelSpec
) -> EditModelOutput | ModelEditFailure:
    """Merge supplied fields, admit the complete model, and evaluate its readiness."""
    try:
        edited = apply_model_edit(DynamicalModelSpec() if parent is None else parent, edit)
        selection = StructuralSelection.for_model(edited.dynamical_model_spec)
    except (ValidationError, StructuralSelectionError) as exc:
        return ModelEditFailure(detail=str(exc))
    return EditModelOutput(
        dynamical_model_spec=edited.dynamical_model_spec,
        checks=model_check_report(selection),
        identification=selection.identification,
        pruning=edited.pruning,
    )
