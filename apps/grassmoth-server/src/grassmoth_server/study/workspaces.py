"""Public workspace discovery and input-file staging."""

from grassmoth_server.actions.contracts.polling import StudyError
from grassmoth_server.storage import backend as storage
from grassmoth_server.storage import paths as data_module
from grassmoth_server.study.calls import actions_enabled, parse_workspace_id


def list_workspaces() -> tuple[str, ...]:
    """List workspace directories without opening their study histories."""
    entries = storage.listdir(data_module.data_root(), directories_only=True)
    names = (entry.rstrip("/").rsplit("/", 1)[-1] for entry in entries)
    return tuple(sorted(name for name in names if not name.startswith(".")))


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
