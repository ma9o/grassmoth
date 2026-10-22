"""HTTP client for study actions, saved results and source uploads."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from pydantic import TypeAdapter

from grassmoth.public import Value
from grassmoth_server.public.contracts import ActionResult, StudyError, TimelineResponse

if TYPE_CHECKING:
    from grassmoth_server.public.contracts import ActionId, ActionInput, CallId


class _ErrorBody(Value):
    """The HTTP adapter's expected-error payload."""

    detail: str


class StudyClient:
    """Submit actions and read their retained results without owning execution policy."""

    def __init__(self, endpoint: str, workspace: str) -> None:
        """Bind the transport to one public facade and workspace."""
        self.endpoint = endpoint.rstrip("/")
        self.workspace = workspace

    def _request(self, request: Request) -> bytes | StudyError:
        try:
            with urlopen(request, timeout=60) as response:
                return cast("bytes", response.read())
        except HTTPError as exc:
            codes = {
                400: "invalid_request",
                403: "read_only",
                404: "unknown_call",
                422: "invalid_inputs",
            }
            if exc.code not in codes:
                raise
            body = _ErrorBody.model_validate_json(exc.read())
            return StudyError.model_validate({"code": codes[exc.code], "detail": body.detail})

    def poll(self, action: ActionId, call_id: CallId) -> ActionResult | StudyError:
        """Read an existing action without starting or retrying work."""
        result = self._request(
            Request(f"{self.endpoint}/api/studies/{self.workspace}/{action}/{call_id}")
        )
        return result if isinstance(result, StudyError) else ActionResult(payload=result)

    def submit(self, request: ActionInput) -> ActionResult | StudyError:
        """Submit one complete request to the framework-owned action lifecycle."""
        result = self._request(
            Request(
                f"{self.endpoint}/api/studies/{self.workspace}/{request.action}",
                data=request.model_dump_json(round_trip=True).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        )
        return result if isinstance(result, StudyError) else ActionResult(payload=result)

    def timeline(self) -> TimelineResponse | StudyError:
        """Read the slim journal and the active action without reading scientific results."""
        result = self._request(Request(f"{self.endpoint}/api/studies/{self.workspace}/timeline"))
        return (
            result
            if isinstance(result, StudyError)
            else TimelineResponse.model_validate_json(result)
        )

    def upload(self, filename: str, payload: bytes) -> StudyError | None:
        """Stage a source file through the public upload endpoint."""
        boundary = "grassmoth-upload-" + uuid4().hex
        body = (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="workspaceId"\r\n\r\n{self.workspace}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
            + payload
            + f"\r\n--{boundary}--\r\n".encode()
        )
        result = self._request(
            Request(
                self.endpoint + "/api/upload",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
        )
        if isinstance(result, StudyError):
            return result
        TypeAdapter(str).validate_json(result)
        return None
