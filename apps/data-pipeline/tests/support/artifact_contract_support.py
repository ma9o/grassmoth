"""Test validation against the production artifact catalog."""

from typing import Any

from pydantic import TypeAdapter
from scripts.artifact_contracts import ARTIFACT_CONTRACTS


def validate_artifact_payload(artifact_id: str, data: dict[str, Any]) -> dict[str, Any]:
    if artifact_id not in ARTIFACT_CONTRACTS:
        known = ", ".join(sorted(ARTIFACT_CONTRACTS))
        raise ValueError(f"Unknown artifact_id '{artifact_id}'. Expected one of: {known}")
    adapter = TypeAdapter(ARTIFACT_CONTRACTS[artifact_id])
    return adapter.dump_python(adapter.validate_python(data), mode="json")
