"""Record the study question as the first artifact of its lineage."""

from grassmoth.public import GitOid
from grassmoth.public import edit_question as question_result
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.actions.contracts.requests import EditQuestionRequest
from grassmoth_server.study.artifact_files import json_filename
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import Applied


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
