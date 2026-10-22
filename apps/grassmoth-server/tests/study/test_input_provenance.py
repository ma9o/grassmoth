"""Stored deterministic input laws retain authored provenance."""

import pytest

from grassmoth_server.storage import paths as data
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.queries.lineage import law_provenance
from grassmoth_test_support.input_trajectories import deterministic_input_model

pytestmark = pytest.mark.contract


def test_stored_deterministic_inputs_have_authored_provenance(tmp_path, monkeypatch):
    dynamical_model_spec = deterministic_input_model()
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("INPUTLAW")
    record = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": dynamical_model_spec.model_dump(mode="json")},
    )
    assert law_provenance(store, record, dynamical_model_spec, None).kind == "authored"
