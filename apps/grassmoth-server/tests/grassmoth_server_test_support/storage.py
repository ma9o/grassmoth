"""Seed complete result-backed records for tests that isolate their consumers."""

from pathlib import Path
from uuid import uuid4

from grassmoth_server.storage import paths
from grassmoth_server.study.persistence.call import SavedResult
from grassmoth_test_support.helpers import observation_dataset


def result_blob(store, result, *, call_id=None):
    identity = call_id or f"fixture-{uuid4()}"
    revision = store.write_result(result, call_id=identity)
    saved = SavedResult.model_validate_json(
        (Path(paths.call_dir(store.workspace_id, identity)) / "result.json").read_bytes()
    )
    # Unpublished fixture records must also resolve their law buffers.
    for array in saved.law_arrays:
        ref = f"refs/arrays/{array}"
        if ref not in store.repo.references:
            store.repo.references.create(ref, revision)
    return revision


def stage_result(store, revision, request):
    from grassmoth_server.actions.contracts.requests import call_identity
    from grassmoth_server.study.persistence.arrays import owned_arrays
    from grassmoth_server.study.persistence.codec import unpack_result

    identity = call_identity(request)
    directory = Path(paths.call_dir(store.workspace_id, identity))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "result.msgpack"
    payload = store.read_result_bytes(revision)
    path.write_bytes(payload)
    return store.import_result(
        call_id=identity,
        path=path,
        law_arrays=tuple(owned_arrays(unpack_result(payload).get("dynamical_model_spec"))),
        full_fit=None,
    )


def seed_record(
    store, artifact_id, *, derived_from, produced_by, json_files=None, parquet_files=None
):
    from grassmoth.actions.contracts.io import EditModelOutput
    from grassmoth.actions.prepare_data.action import prepared_output
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth.artifacts.observations.preparation import (
        DataPreparationResult,
        PreparedDataMetadata,
    )

    documents = json_files or {}
    if artifact_id == "raw_data":
        return store.raw_record(f"fixture-{uuid4()}")
    if artifact_id == "model":
        output = EditModelOutput(
            dynamical_model_spec=DynamicalModelSpec.model_validate(documents["model.json"]),
            checks={"specification": (), "question": {"findings": ()}},
            identification={"outcome": None, "treatments": {}},
            pruning={},
        )
    else:
        assert parquet_files is not None
        metadata = PreparedDataMetadata.model_validate(documents["metadata.json"])
        output = prepared_output(
            observation_dataset(
                parquet_files["panel.parquet"], metadata.variables, time_origin=metadata.time_origin
            ),
            metadata,
            DataPreparationResult(),
        )
    return store.result_record(
        artifact_id,
        result_blob(store, output),
        derived_from=derived_from,
        produced_by=produced_by,
    )


def seed_model_record(store, payload, *, derived_from, produced_by):
    return seed_record(
        store,
        "model",
        derived_from=derived_from,
        produced_by=produced_by,
        json_files={
            "model.json": payload.model_dump(mode="python", context={"binary_arrays": True})
        },
    )


def seed_source_table(store, frame, call_id="seq-000001"):
    directory = Path(paths.call_dir(store.workspace_id, call_id))
    sources = directory / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / "raw.parquet")
    frame.write_parquet(sources / "fixture.parquet")
    return store.raw_record(call_id)


def routing_record(store, kind, *, derived_from):
    """A routing-only fixture needs record ownership but does not read its body."""
    from pydantic import RootModel

    return store.result_record(
        kind,
        result_blob(store, RootModel({})),
        derived_from=derived_from,
        produced_by="prepare_data",
    )
