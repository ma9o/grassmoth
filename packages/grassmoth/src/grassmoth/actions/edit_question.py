"""Retain the authored question as a complete scientific action result."""

from grassmoth.actions.contracts.bodies import EditQuestionOutput
from grassmoth.artifacts.model.question import QuestionSpec


def edit_question(question: QuestionSpec) -> EditQuestionOutput:
    """Return the question supplied to the action."""
    return EditQuestionOutput(question=question)
