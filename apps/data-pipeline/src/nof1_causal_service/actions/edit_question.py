"""Record the study question as the first artifact of its lineage."""

from nof1_causal_lab.actions.edit_question import edit_question as question_result
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_service.actions.contracts.effects import ActionEffects
from nof1_causal_service.actions.contracts.requests import EditQuestionRequest
from nof1_causal_service.study.artifact_files import json_filename
from nof1_causal_service.study.persistence.store import ArtifactStore
from nof1_causal_service.study.records import Applied


def edit_question(workspace_id: str, request: EditQuestionRequest) -> Applied[GitOid]:
    """Persist the authored question and return the artifact publication effect."""
    store = ArtifactStore(workspace_id)
    result = question_result(request.input.question)
    info = store.write_artifact(
        "question",
        derived_from={},
        produced_by="edit_question",
        json_files={json_filename("question", "question"): result.question.model_dump(mode="json")},
    )
    return Applied(result=store.write_result(result), effects=ActionEffects(produced=(info,)))
