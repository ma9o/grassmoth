"""Read pinned files and saved model results through the public study client."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from evaluation.declarations import CaseFile
from nof1_causal_lab.public.client import StudyClient
from nof1_causal_lab.public.contracts import DynamicalModelSpec
from nof1_causal_lab.public.studies import StudyError

if TYPE_CHECKING:
    from pathlib import Path

    from evaluation.declarations import ModelSource


def read_source(source: ModelSource, root: Path) -> bytes:
    """Verify the exact captured bytes; missing inputs and changed content are errors."""
    if isinstance(source, CaseFile):
        payload = (root / source.path).read_bytes()
    else:
        result = StudyClient(source.endpoint, source.workspace).poll("edit_model", source.call_id)
        if isinstance(result, StudyError):
            raise ValueError(result.detail)
        poll = result.decode()
        if poll.status != "success" or poll.action != "edit_model":
            raise ValueError("Model source must name a successful edit_model call")
        payload = poll.body.dynamical_model_spec.model_dump_json(round_trip=True).encode()
    if hashlib.sha256(payload).hexdigest() != source.sha256:
        raise ValueError(f"Case source hash mismatch: {source}")
    return payload


def model_document(payload: bytes) -> DynamicalModelSpec:
    """Decode the pinned document using the production model and payload contracts."""
    return DynamicalModelSpec.model_validate_json(payload)
