"""Workspace storage paths; observation transformations live in observation_rows."""

from nof1_causal_lab.storage.backend import get_base_uri, join

# Remote-aware base URI (``/abs/path/to/data`` locally, ``s3://bucket/prefix`` on R2).
# Consumers choose a lifecycle through the builders below rather than joining
# paths against this root directly.
_DATA_URI = get_base_uri()

# ---------------------------------------------------------------------------
# Workspace storage tiers
#
# Every path under data/{workspace_id}/ belongs to exactly one lifecycle tier,
# and all workspace paths must be built through these functions so choosing a
# tier is explicit:
#
#   ledger  (input/ + study/ + store/)  durable and committable; must be
#           referentially closed — ledger content never points outside the
#           ledger (sole sanctioned exception: AttemptRecord.resume, a
#           typed retention pointer into scratch)
#   cache/  content-addressed or regenerable sidecars — safe to evict anytime
#   scratch/  run-scoped Temporal execution state and UI telemetry — collected
#           by machine.sweep once the run is off the resume path
# ---------------------------------------------------------------------------


def data_root() -> str:
    """Return the configured root URI containing study workspaces."""
    return _DATA_URI


def workspace_dir(workspace_id: str) -> str:
    """Resolve a workspace directory beneath the configured local or remote data root."""
    return join(_DATA_URI, workspace_id)


def input_dir(workspace_id: str) -> str:
    """Ledger tier: user-provided input bundle at ``data/{workspace_id}/input/``."""
    return join(_DATA_URI, workspace_id, "input")


def store_dir(workspace_id: str) -> str:
    """Ledger tier: versioned artifact store at ``data/{workspace_id}/store/``."""
    return join(_DATA_URI, workspace_id, "store")


def study_dir(workspace_id: str) -> str:
    """Ledger tier: local Git snapshots and commit-local logs at ``data/{workspace_id}/study/``."""
    return join(_DATA_URI, workspace_id, "study")


def cache_dir(workspace_id: str) -> str:
    """Cache tier: evictable regenerable sidecars at ``data/{workspace_id}/cache/``."""
    return join(_DATA_URI, workspace_id, "cache")


def scratch_dir(workspace_id: str) -> str:
    """Scratch tier root at ``data/{workspace_id}/scratch/``."""
    return join(_DATA_URI, workspace_id, "scratch")


def scratch_events_dir(workspace_id: str) -> str:
    """Scratch tier: UI telemetry event stream (one JSON file per event)."""
    return join(_DATA_URI, workspace_id, "scratch", "events")


def scratch_runs_dir(workspace_id: str) -> str:
    """Scratch tier: parent of all per-run execution state."""
    return join(_DATA_URI, workspace_id, "scratch", "runs")


def scratch_run_dir(workspace_id: str, run_id: str) -> str:
    """Scratch tier: ALL execution state of one run (= one journal seq) lives here.

    One run is one GC unit: machine.sweep deletes the whole directory once the
    run's seq is journaled and off the model-spec resume path.
    """
    return join(_DATA_URI, workspace_id, "scratch", "runs", run_id)
