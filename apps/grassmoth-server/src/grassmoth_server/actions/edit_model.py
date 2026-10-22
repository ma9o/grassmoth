"""Resolve an edit's pinned values, execute it, and stage its complete result."""

from grassmoth.public import DynamicalModelSpec, GitOid, ModelEditFailure
from grassmoth.public import edit_model as edit_model_values
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.actions.contracts.requests import EditModelRequest
from grassmoth_server.actions.input_resolution import model_edit_state
from grassmoth_server.study.persistence.store import (
    ArtifactStore,
    model_values,
    read_model,
    read_question,
)
from grassmoth_server.study.persistence.writes import write_model_revision
from grassmoth_server.study.records import Applied, Rejected


def edit_model(workspace_id: str, request: EditModelRequest[GitOid]) -> Applied[GitOid] | Rejected:
    """Keep reference resolution and publication outside the scientific edit."""
    from grassmoth_server.bootstrap import initialize_numerical_runtime

    initialize_numerical_runtime()
    store = ArtifactStore(workspace_id)
    state = model_edit_state(store, request.input.parent_ref)
    question = read_question(store, state.current["question"].revision)
    parent = state.get("model")
    base = read_model(store, parent.revision) if parent is not None else DynamicalModelSpec()
    result = edit_model_values(
        question, model_values(store, base), model_values(store, request.input.dynamical_model_spec)
    )
    if isinstance(result, ModelEditFailure):
        return Rejected(code="MODEL_SPEC_INVALID", subject="model", detail=result.detail)
    info = write_model_revision(
        store,
        result.dynamical_model_spec,
        derived_from={kind: record.revision for kind, record in state.current.items()},
        produced_by="edit_model",
    )
    return Applied(result=store.write_result(result), effects=ActionEffects(produced=(info,)))
