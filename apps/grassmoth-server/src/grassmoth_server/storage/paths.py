"""Durable study history and call-owned temporary execution files."""

from grassmoth_server.storage.backend import get_base_uri, join

_DATA_URI = get_base_uri()


def data_root() -> str:
    """Return the configured root containing study workspaces."""
    return _DATA_URI


def workspace_dir(workspace_id: str) -> str:
    """Locate one workspace under the configured data root."""
    return join(_DATA_URI, workspace_id)


def history_dir(workspace_id: str) -> str:
    """Locate the complete durable study, including its storage format config."""
    return join(workspace_dir(workspace_id), "history.git")


def tmp_dir(workspace_id: str) -> str:
    """Locate temporary uploads and unfinished call state."""
    return join(workspace_dir(workspace_id), "tmp")


def input_dir(workspace_id: str) -> str:
    """Locate uploads waiting to be captured by prepare_data."""
    return join(tmp_dir(workspace_id), "input")


def call_dir(workspace_id: str, call_id: str) -> str:
    """Locate the temporary files owned by one content-named call."""
    return join(tmp_dir(workspace_id), call_id)
