"""Public workspace discovery and input-file staging."""

from collections.abc import Mapping

from grassmoth_server.actions.contracts.polling import StudyError
from grassmoth_server.storage import backend as storage
from grassmoth_server.storage import paths as data_module
from grassmoth_server.study.calls import actions_enabled, parse_workspace_id
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore, read_model


def list_workspaces() -> Mapping[str, str | None]:
    """Read available workspace names and their retained study questions."""
    workspaces = {}
    for entry in sorted(storage.listdir(data_module.data_root())):
        workspace_id = entry.rstrip("/").rsplit("/", 1)[-1]
        if not workspace_id or workspace_id.startswith("."):
            continue
        model = StudyRepository(workspace_id).state().get("model")
        workspaces[workspace_id] = (
            read_model(ArtifactStore(workspace_id), model.revision).question
            if model is not None
            else None
        )
    return workspaces


def stage_input(workspace_id: str, filename: str, payload: bytes) -> str | StudyError:
    """Stage an input file; prepare_data captures its bytes into the action identity."""
    if not actions_enabled():
        return StudyError(code="read_only", detail="This read-only facade answers saved calls only")
    parsed = parse_workspace_id(workspace_id)
    if isinstance(parsed, StudyError):
        return parsed
    filename = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if not filename or filename in {".", ".."}:
        return StudyError(code="invalid_request", detail="Invalid file name")
    directory = data_module.input_dir(parsed)
    storage.makedirs(directory)
    with storage.open_file(storage.join(directory, filename), "wb") as handle:
        handle.write(payload)
    return f"{parsed}/tmp/input/{filename}"
