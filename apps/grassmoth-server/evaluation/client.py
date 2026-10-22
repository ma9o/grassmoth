"""HTTP-only action client; it never invokes an inference or generation engine."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from grassmoth_server.public.client import StudyClient
from grassmoth_server.public.contracts import (
    ActionInput,
    ActionPoll,
    ActionResult,
    ActionSuccess,
    ContinueFitTarget,
    FailedPoll,
    FitRequest,
    NewFitTarget,
    RunningPoll,
    StudyError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from evaluation.budget import BudgetLedger
    from grassmoth_server.public.contracts import (
        ActionId,
        ArtifactId,
        CallId,
        GitOid,
        TimelineResponse,
    )


class ActionClient:
    """Persist every returned envelope so failures and reused calls remain reviewable."""

    def __init__(
        self, endpoint: str, workspace: str, output: Path, *, budget: BudgetLedger | None = None
    ):
        self.transport = StudyClient(endpoint, workspace)
        self.endpoint = self.transport.endpoint
        self.workspace = workspace
        self.output = output
        self.reused: dict[str, bool] = {}
        self.budget = budget
        output.mkdir(parents=True, exist_ok=True)

    def _retain(self, result: ActionResult | StudyError) -> ActionPoll:
        if isinstance(result, StudyError):
            raise ValueError(result.detail)
        poll = result.decode()
        (self.output / (poll.call_id + ".msgpack")).write_bytes(result.payload)
        return poll

    def saved(self, action: ActionId, call_id: CallId) -> ActionSuccess:
        """Read a saved call without executing or retrying it."""
        poll = self._retain(self.transport.poll(action, call_id))
        if isinstance(poll, (RunningPoll, FailedPoll)):
            raise ValueError(f"Call {call_id} is {poll.status}; inspect its retained envelope")
        return poll

    def call(self, request: ActionInput) -> ActionSuccess:
        """Submit one typed action and poll its pinned identity until publication."""
        reused = False
        if isinstance(request, FitRequest):
            identity = request.call_id
            existing = self.transport.poll(request.action, identity)
            if isinstance(existing, StudyError):
                if existing.code != "unknown_call":
                    raise ValueError(existing.detail)
            else:
                self._retain(existing)
                reused = True
            # A POST may be accepted even if its response is lost. Retain the
            # pinned identity before submission so recovery polls this call
            # without allocating another fit or inventing a scientific failure.
            (self.output / (identity + ".request.json")).write_text(
                request.model_dump_json(round_trip=True)
            )
            self.reused[identity] = reused
            if (
                not reused
                and self.budget is not None
                and request.input.settings.hardware != "local"
            ):
                self.budget.reserve(self.endpoint, self.workspace, identity, request.input.settings)
        poll = self._retain(self.transport.submit(request))
        self.reused[poll.call_id] = reused
        while isinstance(poll, RunningPoll):
            time.sleep(1)
            poll = self._retain(self.transport.poll(poll.action, poll.call_id))
        if isinstance(request, FitRequest) and self.budget is not None:
            self.budget.finish(self.endpoint, self.workspace, poll.call_id, poll.status)
        if isinstance(poll, FailedPoll):
            raise ValueError(f"{request.action} failed; inspect {poll.call_id}")
        return poll

    def _timeline(self) -> TimelineResponse:
        timeline = self.transport.timeline()
        if isinstance(timeline, StudyError):
            raise ValueError(timeline.detail)
        (self.output / "timeline.json").write_text(timeline.model_dump_json())
        return timeline

    def artifact_revision(self, call_id: str, artifact: ArtifactId) -> GitOid:
        """Select a produced artifact tree; an action commit is a different revision."""
        from grassmoth_server.public.contracts import Applied

        item = next(item for item in self._timeline().attempts if item.call_id == call_id)
        outcome = item.record.attempt.outcome
        if not isinstance(outcome, Applied):
            raise ValueError("Only successful actions produce case artifacts")
        return next(
            info.revision for info in outcome.effects.produced if info.artifact_id == artifact
        )

    def fit_source(self, call_id: str) -> NewFitTarget[GitOid]:
        """Check a saved reference's original inputs against its public journal lineage."""
        timeline = self._timeline()
        by_revision = {item.commit_id: item for item in timeline.attempts}
        item = next(item for item in timeline.attempts if item.call_id == call_id)
        while True:
            request = item.record.attempt.request
            if not isinstance(request, FitRequest):
                raise ValueError("Reference lineage must contain fit requests")
            target = request.input.target
            if not isinstance(target, ContinueFitTarget):
                return target
            item = by_revision[target.fit_ref]

    def upload(self, filename: str, payload: bytes) -> None:
        """Stage source bytes through the framework's public client."""
        error = self.transport.upload(filename, payload)
        if error is not None:
            raise ValueError(error.detail)
