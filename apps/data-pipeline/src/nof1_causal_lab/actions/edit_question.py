"""Record the study question as the first artifact of its lineage."""

from nof1_causal_lab.actions.contracts.effects import ActionEffects
from nof1_causal_lab.actions.contracts.requests import EditQuestionRequest
from nof1_causal_lab.study.artifact_files import json_filename
from nof1_causal_lab.study.persistence.store import ArtifactStore
from nof1_causal_lab.study.records import Applied


def edit_question(workspace_id: str, request: EditQuestionRequest) -> Applied[None]:
    """Persist the authored question and return the artifact publication effect."""
    info = ArtifactStore(workspace_id).write_artifact(
        "question",
        derived_from={},
        produced_by="edit_question",
        json_files={
            json_filename("question", "question"): request.input.question.model_dump(mode="json")
        },
    )
    return Applied(result=None, effects=ActionEffects(produced=(info,)))
