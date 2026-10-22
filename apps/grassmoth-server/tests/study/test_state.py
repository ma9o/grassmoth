"""Immutable scientific facts are installed atomically and keep their input lineage."""

import pytest
from pydantic import ValidationError

from grassmoth_server.study.state import (
    ArtifactRecord,
    RetractedArtifact,
    StudyState,
    apply_effects,
    is_stale,
)
from tests.support.git import git_oid

pytestmark = pytest.mark.contract


def test_install_supersede_and_retract_facts_without_mutating_the_prior_state():
    first = ArtifactRecord(artifact_id="raw_data", revision=git_oid(1))
    original = apply_effects(StudyState(), (first,))
    second = first.revised(revision=git_oid(2))
    current = apply_effects(original, (second,))
    assert original.current["raw_data"] == first
    assert current.current["raw_data"] == second
    removed = apply_effects(
        current, (), (RetractedArtifact(artifact_id="raw_data", reason_ref="input.removed"),)
    )
    assert not removed.has("raw_data")
    assert current.has("raw_data")


def test_fact_freshness_follows_exact_execution_inputs_and_their_ancestors():
    raw = ArtifactRecord(artifact_id="raw_data", revision=git_oid(1))
    panel = ArtifactRecord(
        artifact_id="panel", revision=git_oid(2), derived_from={"raw_data": raw.revision}
    )
    model = ArtifactRecord(
        artifact_id="model",
        revision=git_oid(3),
        produced_by="fit",
        derived_from={"panel": panel.revision},
    )
    state = StudyState().with_artifacts((raw, panel, model))
    assert not is_stale(state, "panel")
    assert not is_stale(state, "model")
    assert state.matches_inputs("panel", "raw_data")
    revised = state.with_artifacts((raw.revised(revision=git_oid(4)),))
    assert is_stale(revised, "panel")
    assert not is_stale(revised, "model")
    assert is_stale(state.without(["raw_data"]), "panel")
    assert not is_stale(StudyState(), "panel")


def test_history_shapes_accept_facts_and_reject_stored_findings_and_fingerprints():
    for payload in (
        {"artifact_id": "validation_report", "revision": str(git_oid(1))},
        {"artifact_id": "model", "revision": str(git_oid(1)), "model_inputs": {}},
    ):
        with pytest.raises(ValidationError):
            ArtifactRecord.model_validate(payload)
    with pytest.raises(ValidationError):
        StudyState.model_validate({"checks": {}})
