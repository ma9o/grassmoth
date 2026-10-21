"""Retain the authored question as a complete scientific action result."""

from nof1_causal_lab.actions.contracts.bodies import EditQuestionOutput
from nof1_causal_lab.artifacts.model.question import QuestionSpec


def edit_question(question: QuestionSpec) -> EditQuestionOutput:
    """Return the question supplied to the action."""
    return EditQuestionOutput(question=question)
