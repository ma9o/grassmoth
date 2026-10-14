"""Stage one explicit model edit before action-owned checks and atomic publication."""

from functools import cache

from pydantic import ValidationError

from nof1_causal_lab.actions.contracts.effects import ActionEffects
from nof1_causal_lab.actions.contracts.requests import EditModelRequest
from nof1_causal_lab.actions.input_resolution import model_edit_state
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model.dynamical_model_spec import (
    DynamicalModelSpec,
    ModelEditResult,
    apply_model_edit,
)
from nof1_causal_lab.models.causal.question_checks import question_edit_reason
from nof1_causal_lab.study.persistence.store import ArtifactStore, read_model, read_question
from nof1_causal_lab.study.persistence.writes import write_model_revision
from nof1_causal_lab.study.records import Applied, Rejected


def edit_model(
    workspace_id: str, request: EditModelRequest[GitOid]
) -> Applied[ModelEditResult] | Rejected:
    """Merge into the selected parent, prune unrelated structure, and validate before writing."""
    store = ArtifactStore(workspace_id)
    state = model_edit_state(store, request.input.parent_ref)
    question = read_question(store, state.current["question"].revision)
    parent = state.get("model")
    base = read_model(store, parent.revision) if parent is not None else DynamicalModelSpec()
    try:
        edited = apply_model_edit(
            base,
            request.input.dynamical_model_spec,
            question.outcome,
            context={"distribution_array_loader": cache(store.read_array)},
        )
    except ValidationError as exc:
        return Rejected(code="MODEL_SPEC_INVALID", subject="model", detail=str(exc))
    if (reason := question_edit_reason(edited.dynamical_model_spec, question)) is not None:
        return Rejected(code="MODEL_SPEC_INVALID", subject="model", detail=reason)
    info = write_model_revision(
        store,
        edited.dynamical_model_spec,
        derived_from={kind: record.revision for kind, record in state.current.items()},
        produced_by="edit_model",
    )
    return Applied(
        result=edited.pruning,
        effects=ActionEffects(produced=(info,)),
    )
