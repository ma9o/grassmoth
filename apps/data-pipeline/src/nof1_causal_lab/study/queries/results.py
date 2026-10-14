"""The complete result of a successful action is its durable cache entry."""

from nof1_causal_lab.actions.contracts.bodies import (
    DataDiffOutput,
    EditModelOutput,
    EditQuestionOutput,
    FitOutput,
    ModelDiffOutput,
    PrepareDataOutput,
    SimulateOutput,
)
from nof1_causal_lab.artifacts.identity import ActionId, GitOid
from nof1_causal_lab.study.persistence.store import ArtifactStore

type ActionOutput = (
    EditQuestionOutput
    | EditModelOutput
    | PrepareDataOutput
    | FitOutput
    | SimulateOutput
    | DataDiffOutput
    | ModelDiffOutput
)


def read_result(store: ArtifactStore, action: ActionId, revision: GitOid) -> ActionOutput:
    """Decode the saved result with the schema owned by its recorded action."""
    match action:
        case "edit_question":
            return store.read_result(revision, EditQuestionOutput)
        case "edit_model":
            return store.read_result(revision, EditModelOutput)
        case "prepare_data":
            return store.read_result(revision, PrepareDataOutput)
        case "fit":
            return store.read_result(revision, FitOutput)
        case "simulate":
            return store.read_result(revision, SimulateOutput)
        case "data_diff":
            return store.read_result(revision, DataDiffOutput)
        case "model_diff":
            return store.read_result(revision, ModelDiffOutput)
