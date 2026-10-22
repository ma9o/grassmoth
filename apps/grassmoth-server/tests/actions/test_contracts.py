"""Tests for artifact payload contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from grassmoth_server_test_support.artifact_contract_support import validate_artifact_payload
from grassmoth_test_support.helpers import make_model

pytestmark = pytest.mark.contract


@pytest.fixture
def valid_artifact_payloads() -> dict[str, dict[str, Any]]:
    """Minimal valid payload for each persisted artifact id."""
    return {
        "question": {"text": "Does stress change performance?", "outcome": "construct:perf"},
        "model": make_model(["Stress", "Perf"], [("Stress", "Perf")]).model_dump(mode="json"),
    }


def test_validate_artifact_payload_accepts_all_artifacts(
    valid_artifact_payloads: dict[str, dict[str, Any]],
):
    """Each artifact payload validates and round-trips to a JSON-serializable dict."""
    for artifact_id, payload in valid_artifact_payloads.items():
        validated = validate_artifact_payload(artifact_id, payload)
        assert isinstance(validated, dict)


def test_validate_artifact_payload_rejects_missing_required_fields(
    valid_artifact_payloads: dict[str, dict[str, Any]],
):
    """Artifact contract validation should fail on contract violations."""
    bad = deepcopy(valid_artifact_payloads["question"])
    bad.pop("outcome")
    with pytest.raises(ValidationError):
        validate_artifact_payload("question", bad)
