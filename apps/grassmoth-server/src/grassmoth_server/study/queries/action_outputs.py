"""Read the complete result saved by a completed action, without scientific execution."""

from grassmoth_server.actions.call_logs import read_call_log
from grassmoth_server.actions.contracts.polling import FailedPoll
from grassmoth_server.actions.contracts.requests import call_identity
from grassmoth_server.study.errors import StudyLookupError
from grassmoth_server.study.persistence.codec import pack_envelope, pack_result
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.records import Applied, StudyRevision


def completed_call_msgpack(workspace_id: str, revision: StudyRevision) -> bytes:
    """Return the saved body bytes inside the polling envelope, without rebuilding projections."""
    attempt = revision.record.attempt
    if attempt.request is None:
        raise StudyLookupError("This historical attempt has no retained call arguments")
    repository = StudyRepository(workspace_id)
    identity = call_identity(attempt.request)
    log = read_call_log(repository, revision.commit_id)
    if not isinstance(attempt.outcome, Applied):
        return pack_result(
            FailedPoll(
                call_id=identity,
                action=attempt.action,
                commit_id=revision.commit_id,
                messages=log.messages,
            )
        )
    metadata = {
        "call_id": identity,
        "action": attempt.action,
        "status": "success",
        "commit_id": revision.commit_id,
        "messages": log.model_dump(mode="json")["messages"],
    }
    from grassmoth_server.study.persistence.store import ArtifactStore

    store = ArtifactStore(workspace_id)
    if attempt.action == "fit":
        from grassmoth.public import FitOutput
        from grassmoth_server.compute.remote_receipts import published_fit_report
        from grassmoth_server.study.persistence.codec import pack_payload, result_payload

        saved = published_fit_report(store, attempt.outcome.result)
        if isinstance(saved, FitOutput):
            return pack_envelope(metadata, pack_payload(result_payload(saved, thin_fit_draws=True)))
    # Remote transfer bodies are already capped. Serve their verified bytes unchanged.
    return pack_envelope(metadata, store.read_result_bytes(attempt.outcome.result))
