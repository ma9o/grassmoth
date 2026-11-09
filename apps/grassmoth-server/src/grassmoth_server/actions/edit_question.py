"""Root the study with its question and initial outcome model."""

from grassmoth.public import GitOid
from grassmoth.public import edit_question as question_result
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.actions.contracts.requests import EditQuestionRequest, call_identity
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import Applied


def edit_question(workspace_id: str, request: EditQuestionRequest) -> Applied[GitOid]:
    """Publish the question and its initial model together from their complete result."""
    from grassmoth_server.bootstrap import initialize_numerical_runtime

    initialize_numerical_runtime()
    store = ArtifactStore(workspace_id)
    result = question_result(request.input.question)
    saved = store.write_result(result, call_id=call_identity(request))
    info = store.result_record("question", saved, derived_from={}, produced_by="edit_question")
    model = store.result_record(
        "model", saved, derived_from={"question": info.revision}, produced_by="edit_question"
    )
    return Applied(result=saved, effects=ActionEffects(produced=(info, model)))
