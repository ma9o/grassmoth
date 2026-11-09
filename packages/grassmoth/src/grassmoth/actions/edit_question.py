"""Establish the question and its outcome-only initial model."""

from grassmoth.actions.checks import model_check_report
from grassmoth.actions.contracts.io import EditQuestionOutput
from grassmoth.artifacts.identity import ConstructId
from grassmoth.artifacts.model.construct import ConstructSpec
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.question import QuestionSpec
from grassmoth.models.structure import StructuralSelection


def edit_question(question: QuestionSpec[ConstructSpec]) -> EditQuestionOutput:
    """Save the question's identity, initial model and scientific readiness findings."""
    saved = QuestionSpec[ConstructId](
        text=question.text, outcome=question.outcome.id, queries=question.queries
    )
    model = DynamicalModelSpec.from_entities(constructs=(question.outcome,))
    selection = StructuralSelection.for_question(model, saved)
    return EditQuestionOutput(
        question=saved,
        dynamical_model_spec=model,
        checks=model_check_report(selection, saved),
        identification=selection.identification,
    )
