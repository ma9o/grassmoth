"""Record the study question as the first artifact of its lineage."""

from grassmoth.public import GitOid
from grassmoth.public import edit_question as question_result
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.actions.contracts.requests import EditQuestionRequest, call_identity
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import Applied


def edit_question(workspace_id: str, request: EditQuestionRequest) -> Applied[GitOid]:
    """Persist the authored question and return the artifact publication effect."""
    store = ArtifactStore(workspace_id)
    result = question_result(request.input.question)
    saved = store.write_result(result, call_id=call_identity(request))
    info = store.result_record("question", saved, derived_from={}, produced_by="edit_question")
    return Applied(result=saved, effects=ActionEffects(produced=(info,)))
