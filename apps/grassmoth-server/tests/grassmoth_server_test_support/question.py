"""Small authored questions and initial results for isolated contract tests."""

from grassmoth.actions.contracts.io import EditQuestionOutput
from grassmoth.artifacts.identity import ConstructId
from grassmoth.artifacts.model.construct import ConstructSpec
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.question import QuestionSpec


def authored_question(question: QuestionSpec[ConstructId]) -> QuestionSpec[ConstructSpec]:
    return QuestionSpec[ConstructSpec].model_validate(
        {
            "text": question.text,
            "queries": question.queries,
            "outcome": {
                "id": question.outcome,
                "name": str(question.outcome).removeprefix("construct:"),
                "reasoning": {"text": question.text},
                "role": "endogenous",
                "temporal_status": "time_varying",
            },
        }
    )


def question_output(question: QuestionSpec[ConstructId]) -> EditQuestionOutput:
    return EditQuestionOutput(
        question=question,
        dynamical_model_spec=DynamicalModelSpec.from_entities(
            constructs=(authored_question(question).outcome,)
        ),
        checks={"specification": (), "question": {"findings": ()}},
        identification={"outcome": question.outcome, "treatments": {}},
    )
