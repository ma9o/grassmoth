"""HTTP workspaces boundary."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import (
    APIRouter,
    Response,
)

from nof1_causal_lab.api.policy import actions_enabled
from nof1_causal_lab.storage import backend as storage
from nof1_causal_lab.storage import paths as data_module
from nof1_causal_lab.study.persistence.history import StudyRepository
from nof1_causal_lab.study.persistence.store import ArtifactStore, read_question
from nof1_causal_lab.study.records import (
    Applied,
)

workspaces_router = APIRouter(prefix="/api")


@workspaces_router.get("/workspaces", response_model=Mapping[str, str | None])
def list_workspaces(response: Response) -> Mapping[str, str | None]:
    """Available workspaces and their immutable study questions.

    X-Actions-Enabled reports whether the facade accepts new calls.
    """
    response.headers["X-Actions-Enabled"] = "true" if actions_enabled() else "false"
    workspaces = {}
    for entry in sorted(storage.listdir(data_module.data_root())):
        workspace_id = entry.rstrip("/").rsplit("/", 1)[-1]
        if not workspace_id or workspace_id.startswith("."):
            continue
        repository = StudyRepository(workspace_id)
        records = repository.attempts()
        question = next(
            (
                item
                for revision in records
                if revision.record.attempt.action == "edit_question"
                and isinstance(revision.record.attempt.outcome, Applied)
                for item in revision.record.attempt.outcome.effects.produced
                if item.artifact_id == "question"
            ),
            None,
        )
        workspaces[workspace_id] = (
            read_question(ArtifactStore(workspace_id), question.revision).text
            if question is not None
            else None
        )
    return workspaces
