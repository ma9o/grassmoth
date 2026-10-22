"""Read-only selection and comparison contracts for immutable scientific inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.public import DynamicalModelSpec
from nof1_causal_lab.public import model_diff as compare_models

if TYPE_CHECKING:
    from nof1_causal_lab.public import GitOid, ModelDiffOutput


def _model_revision(workspace_id: str, revision: GitOid) -> DynamicalModelSpec:
    """Use the edit-parent contract: a question supplies an empty spec, a model its saved spec."""
    from nof1_causal_service.actions.input_resolution import model_edit_state
    from nof1_causal_service.study.persistence.store import ArtifactStore, model_values, read_model

    store = ArtifactStore(workspace_id)
    model_record = model_edit_state(store, revision).get("model")
    return (
        DynamicalModelSpec.from_entities()
        if model_record is None
        else model_values(store, read_model(store, model_record.revision))
    )


def model_diff(workspace_id: str, before_id: GitOid, after_id: GitOid) -> ModelDiffOutput:
    """Compare saved spec documents without compiling or evaluating scientific findings."""
    before = _model_revision(workspace_id, before_id)
    after = _model_revision(workspace_id, after_id)
    return compare_models(before, after)
