"""Bind complete action results to their produced artifact manifests before publication."""

from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import ActionAttempt, Applied


def complete_attempt(workspace_id: str, attempt: ActionAttempt) -> ActionAttempt:
    """Publish references to the already computed result without recomputing or rereading it."""
    if not isinstance(attempt.outcome, Applied):
        return attempt
    store = ArtifactStore(workspace_id)
    result = attempt.outcome.result
    produced = tuple(
        store.result_artifact(info, result) for info in attempt.outcome.effects.produced
    )
    return attempt.revised(
        outcome=attempt.outcome.revised(
            effects=attempt.outcome.effects.revised(produced=produced, reports={}),
        )
    )
