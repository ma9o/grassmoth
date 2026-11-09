"""Small initial model results for isolated transport and persistence tests."""

from grassmoth.actions.contracts.io import EditModelOutput
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec, ModelEditResult
from grassmoth.artifacts.reports.identification import IdentificationReport
from grassmoth.artifacts.reports.model_checks import ModelCheckReport, QuestionCheckReport


def model_output(model: DynamicalModelSpec) -> EditModelOutput:
    return EditModelOutput(
        dynamical_model_spec=model,
        checks=ModelCheckReport(specification=(), question=QuestionCheckReport(findings=())),
        identification=IdentificationReport(outcome=model.outcome, treatments={}),
        pruning=ModelEditResult(),
    )
