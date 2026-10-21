"""Read the complete result saved by a completed action, without scientific execution."""

from nof1_causal_service.actions.call_logs import read_call_log
from nof1_causal_service.actions.contracts.polling import FailedPoll
from nof1_causal_service.actions.contracts.requests import call_identity
from nof1_causal_service.study.errors import StudyLookupError
from nof1_causal_service.study.persistence.codec import pack_envelope, pack_result
from nof1_causal_service.study.persistence.history import StudyRepository
from nof1_causal_service.study.records import Applied, StudyRevision


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
    body = repository.read_file(revision.commit_id, "result.msgpack")
    return pack_envelope(metadata, body)
