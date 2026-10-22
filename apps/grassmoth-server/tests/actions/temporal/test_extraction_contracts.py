"""Extraction outcomes reject fields belonging to a different alternative."""

import pytest
from pydantic import TypeAdapter, ValidationError

from grassmoth_server.actions.temporal.extraction.messages import ExtractionChunkResult

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "completed", "worker_id": 0, "n_windows": 1, "n_extractions": 1},
        {
            "status": "failed",
            "worker_id": 0,
            "n_windows": 1,
            "n_extractions": 0,
            "error": "failed",
            "result_ref": "out.json",
        },
    ],
)
def test_extraction_outcomes_reject_impossible_fields(payload):
    with pytest.raises(ValidationError):
        TypeAdapter(ExtractionChunkResult).validate_python(payload)
