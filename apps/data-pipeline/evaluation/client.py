"""HTTP-only action client; it never invokes an inference or generation engine."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from pydantic import TypeAdapter

from nof1_causal_lab.actions.contracts.bodies import ContinueFitTarget, NewFitTarget
from nof1_causal_lab.actions.contracts.polling import (
    ActionPoll,
    ActionSuccess,
    FailedPoll,
    RunningPoll,
)
from nof1_causal_lab.actions.contracts.requests import ActionInput, FitRequest, call_identity
from nof1_causal_lab.study.persistence.codec import unpack_result

if TYPE_CHECKING:
    from pathlib import Path

    from nof1_causal_lab.api.studies import TimelineResponse
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid

_POLL = TypeAdapter(ActionPoll)


class ActionClient:
    """Persist every returned envelope so failures and reused calls remain reviewable."""

    def __init__(self, endpoint: str, workspace: str, output: Path):
        self.endpoint = endpoint.rstrip("/")
        self.workspace = workspace
        self.output = output
        self.reused: dict[str, bool] = {}
        output.mkdir(parents=True, exist_ok=True)

    def _read(self, request: Request) -> ActionPoll:
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        poll = _POLL.validate_python(unpack_result(payload))
        (self.output / (poll.call_id + ".msgpack")).write_bytes(payload)
        return poll

    def saved(self, action: str, call_id: str) -> ActionSuccess:
        """Read a saved call without executing or retrying it."""
        poll = self._read(
            Request(f"{self.endpoint}/api/studies/{self.workspace}/{action}/{call_id}")
        )
        if isinstance(poll, (RunningPoll, FailedPoll)):
            raise ValueError(f"Call {call_id} is {poll.status}; inspect its retained envelope")
        return poll

    def call(self, request: ActionInput) -> ActionSuccess:
        """Submit one typed action and poll its pinned identity until publication."""
        url = f"{self.endpoint}/api/studies/{self.workspace}/{request.action}"
        reused = False
        if isinstance(request, FitRequest):
            identity = call_identity(request)
            try:
                self._read(Request(url + "/" + identity))
            except HTTPError as exc:
                if exc.code != 404:
                    raise
            else:
                reused = True
            # A POST may be accepted even if its response is lost. Retain the
            # pinned identity before submission so recovery polls this call
            # without allocating another fit or inventing a scientific failure.
            (self.output / (identity + ".request.json")).write_text(
                request.model_dump_json(round_trip=True)
            )
            self.reused[identity] = reused
        poll = self._read(
            Request(
                url,
                data=request.model_dump_json(round_trip=True).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        )
        self.reused[poll.call_id] = reused
        while isinstance(poll, RunningPoll):
            time.sleep(1)
            poll = self._read(Request(url + "/" + poll.call_id))
        if isinstance(poll, FailedPoll):
            raise ValueError(f"{request.action} failed; inspect {poll.call_id}")
        return poll

    def _timeline(self) -> TimelineResponse:
        from nof1_causal_lab.api.studies import TimelineResponse

        with urlopen(
            f"{self.endpoint}/api/studies/{self.workspace}/timeline", timeout=60
        ) as response:
            payload = response.read()
        timeline = TimelineResponse.model_validate_json(payload)
        (self.output / "timeline.json").write_bytes(payload)
        return timeline

    def require_idle(self) -> None:
        """A live study must never be copied as a benchmark preparation snapshot."""
        if self._timeline().running is not None:
            raise ValueError("Anchor workspace still has a running action")

    def artifact_revision(self, call_id: str, artifact: ArtifactId) -> GitOid:
        """Select a produced artifact tree; an action commit is a different revision."""
        from nof1_causal_lab.study.records import Applied

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
        """Stage source bytes through the same public upload endpoint as study clients."""
        boundary = "nof1-benchmark-" + uuid4().hex
        body = (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="workspaceId"\r\n\r\n{self.workspace}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
            + payload
            + f"\r\n--{boundary}--\r\n".encode()
        )
        with urlopen(
            Request(
                self.endpoint + "/api/upload",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            ),
            timeout=60,
        ) as response:
            response.read()
