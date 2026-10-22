"""A published result is the response and remains reusable without scientific reads."""

import io

import msgpack
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from grassmoth.actions.contracts.bodies import EditQuestionOutput
from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.model.question import QuestionSpec
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.actions.contracts.inputs import EditQuestionInput
from grassmoth_server.actions.contracts.requests import EditQuestionRequest, call_identity
from grassmoth_server.storage import paths as data_module
from grassmoth_server.study.persistence.codec import pack_result, unpack_result
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore, read_question
from grassmoth_server.study.queries.action_outputs import completed_call_msgpack
from grassmoth_server.study.records import Applied, AttemptRecord, EditQuestionAttempt
from tests.support.helpers import fixture_entity_id

pytestmark = pytest.mark.contract


def _array_value(values):
    payload = NumericalArray.from_numpy(values).npy
    return NumericalArray(npy=payload)


def test_poll_and_repeated_post_return_the_saved_result_without_building(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    from grassmoth_server.actions import output_builder
    from grassmoth_server.api import studies as study_api

    workspace = "RESULTS"
    store = ArtifactStore(workspace)
    repository = StudyRepository(workspace)
    question = QuestionSpec(
        text="What changes the outcome?", outcome=fixture_entity_id("construct", "outcome")
    )
    request = EditQuestionRequest(input=EditQuestionInput(question=question))
    result = EditQuestionOutput(question=question)
    ref = store.write_result(result)
    staged = store.write_artifact(
        "question",
        derived_from={},
        produced_by="edit_question",
        json_files={"question.json": question.model_dump(mode="json")},
    )
    artifact = store.result_artifact(staged, ref)
    revision = repository.append(
        AttemptRecord(
            seq=1,
            ts="2026-01-01T00:00:00Z",
            attempt=EditQuestionAttempt(
                action="edit_question",
                request=request,
                outcome=Applied(result=ref, effects=ActionEffects(produced=(artifact,))),
            ),
        )
    )
    expected = msgpack.unpackb(completed_call_msgpack(workspace, revision))
    assert expected["body"] == msgpack.unpackb(
        repository.read_file(revision.commit_id, "result.msgpack")
    )
    assert read_question(store, artifact.revision) == question

    def cannot_build(*args, **kwargs):
        raise AssertionError("A saved call must never construct another result")

    monkeypatch.setattr(output_builder, "complete_attempt", cannot_build)
    monkeypatch.setenv("READ_ONLY_FACADE", "0")
    app = FastAPI()

    class UnusedClients:
        async def get(self):
            raise AssertionError("Cached calls do not contact Temporal")

    app.state.study_clients = UnusedClients()
    app.include_router(study_api.router)
    with TestClient(app) as client:
        for _ in range(2):
            polled = client.get(f"/api/studies/{workspace}/edit_question/{call_identity(request)}")
            assert polled.status_code == 200, polled.text
            assert polled.headers["content-type"] == "application/msgpack"
            assert msgpack.unpackb(polled.content) == expected
            posted = client.post(
                f"/api/studies/{workspace}/edit_question", json=request.model_dump(mode="json")
            )
            assert posted.status_code == 200, posted.text
            assert msgpack.unpackb(posted.content) == expected
    assert repository.latest_seq() == 1


def test_numerical_result_preserves_dtype_shape_missingness_and_vector_selection():
    from grassmoth.artifacts.arrays import ArrayVector, resolve_vector

    values = np.asarray([[1, np.nan, np.inf], [2, -np.inf, 4]], dtype=np.float64)
    saved = _array_value(values)
    loaded = NumericalArray.model_validate(
        msgpack.unpackb(msgpack.packb(saved.model_dump(mode="python")))
    )
    np.testing.assert_array_equal(loaded.values, values)
    assert loaded.values.shape == (2, 3)
    assert loaded.values.dtype == "float64"
    assert saved.content_digest() == loaded.content_digest()
    assert (
        saved.content_digest()
        != NumericalArray.from_numpy(values.astype(np.float32)).content_digest()
    )
    selected = ArrayVector(
        array=loaded,
        indices=(None, 0),
        mask=ArrayVector(array=_array_value(np.asarray([True, False])), indices=(None,)),
    )
    assert resolve_vector(selected) == (1.0, None)
    empty = np.zeros((0, 3), dtype=np.int32)
    np.testing.assert_array_equal(_array_value(empty).values, empty)


@pytest.mark.parametrize(
    "values",
    [np.arange(6, dtype=">f8"), np.asfortranarray(np.arange(6).reshape(2, 3))],
    ids=["big_endian", "fortran_order"],
)
def test_numerical_owner_normalizes_numpy_values_and_rejects_nonportable_wire_buffers(values):
    owned = NumericalArray.from_numpy(values)
    np.testing.assert_array_equal(owned.values, values)
    assert owned.values.flags.c_contiguous
    assert owned.values.dtype.str[0] in "<|"
    original = io.BytesIO()
    np.save(original, values, allow_pickle=False)
    with pytest.raises(ValueError, match="little-endian, row-major"):
        NumericalArray(npy=original.getvalue())


def test_prepared_observations_round_trip_through_the_published_body(monkeypatch, tmp_path):
    import polars as pl
    from polars.testing import assert_frame_equal

    from grassmoth.artifacts.identity import GitOid
    from grassmoth.artifacts.observations.preparation import DataPreparationResult
    from grassmoth.artifacts.observations.reference import DataRef
    from grassmoth.observations.quality.profile import profile_data
    from grassmoth_server.actions.contracts.inputs import PrepareDataInput
    from grassmoth_server.actions.contracts.preparation import FileSourceRef
    from grassmoth_server.actions.contracts.requests import PrepareDataRequest
    from grassmoth_server.actions.output_builder import complete_attempt
    from grassmoth_server.study.queries.data import read_data_history
    from grassmoth_server.study.records import applied_attempt
    from grassmoth_server.study.state import ArtifactResult
    from tests.support.action import question_root
    from tests.support.runner_fixtures import panel_frame, panel_metadata, seed_model

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    workspace = "PREPARED"
    question_root(workspace)
    store, repository = ArtifactStore(workspace), StudyRepository(workspace)
    model = seed_model(store)
    metadata = panel_metadata()
    frame = panel_frame(n_days=2).with_columns(
        pl.col("anchor_time", "support_start", "support_end").str.to_datetime(),
        pl.when(pl.col("value") == 1.0).then(None).otherwise(pl.col("value")).alias("value"),
    )
    panel = store.write_artifact(
        "panel",
        produced_by="prepare_data",
        derived_from={},
        json_files={"metadata.json": metadata.model_dump(mode="json")},
        parquet_files={"panel.parquet": frame},
    )
    expected = read_data_history(
        store, DataRef[GitOid, int](revision=panel.revision, replicate_index=0)
    ).observations.recorded.frame
    profile = profile_data(expected, metadata=metadata)
    request = PrepareDataRequest[GitOid, FileSourceRef](
        input=PrepareDataInput[GitOid, FileSourceRef](
            dynamical_model_spec_ref=model.revision,
            source=FileSourceRef(files=("input/test.csv",), hashes={}),
            extraction={
                item.observation.id: item.extraction for item in metadata.preparation.variables
            },
        )
    )
    from grassmoth.actions.prepare_data.action import prepare_data
    from grassmoth.artifacts.observations.dataset import ObservationDataset

    prepared = prepare_data(
        ObservationDataset.from_frame(
            expected, metadata.variables, time_origin=metadata.time_origin
        ),
        metadata,
        DataPreparationResult(),
    )
    attempt = complete_attempt(
        workspace,
        applied_attempt(
            request,
            Applied(
                result=store.write_result(prepared),
                effects=ActionEffects(
                    produced=(panel,), reports={"data-profile": store.write_result(profile)}
                ),
            ),
        ),
    )
    revision = repository.append(AttemptRecord(seq=2, ts="2026-01-01T00:00:00Z", attempt=attempt))
    assert isinstance(attempt.outcome, Applied)
    published = attempt.outcome.effects.produced[0]
    assert isinstance(published.source, ArtifactResult)
    assert published.source.result == attempt.outcome.result
    assert published.revision != panel.revision
    actual = read_data_history(
        store, DataRef[GitOid, int](revision=revision.commit_id, replicate_index=0)
    ).observations.recorded.frame
    assert_frame_equal(actual, expected)
    body = msgpack.unpackb(completed_call_msgpack(workspace, revision))["body"]
    assert set(body) == {"data", "metadata", "profile", "extraction"}
    assert body["metadata"] == metadata.model_dump(mode="json")
    assert body["profile"] == profile.model_dump(mode="json")
    assert body == msgpack.unpackb(repository.read_file(revision.commit_id, "result.msgpack"))


def test_model_comparisons_resolve_published_artifact_trees_as_specs(monkeypatch, tmp_path):
    from grassmoth.artifacts.identity import GitOid
    from grassmoth_server.actions.contracts.inputs import EditModelInput
    from grassmoth_server.actions.contracts.requests import EditModelRequest
    from grassmoth_server.actions.model_diff import model_diff
    from grassmoth_server.actions.output_builder import complete_attempt
    from grassmoth_server.study.records import applied_attempt
    from tests.support.action import edit_and_check, question_root
    from tests.support.helpers import make_model

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    workspace = "MODEL_REFS"
    question_root(
        workspace, QuestionSpec(text="What changes Y?", outcome=fixture_entity_id("construct", "Y"))
    )
    store, repository = ArtifactStore(workspace), StudyRepository(workspace)
    dynamical_model_spec = make_model(["X", "Y"], [("X", "Y")])
    request = EditModelRequest[GitOid](
        input=EditModelInput[GitOid](
            parent_ref=repository.question().revision,
            dynamical_model_spec=dynamical_model_spec,
        )
    )
    staged = edit_and_check(workspace, request, repository.state())
    assert isinstance(staged, Applied)
    attempt = complete_attempt(workspace, applied_attempt(request, staged))
    repository.append(AttemptRecord(seq=2, ts="2026-01-01T00:00:00Z", attempt=attempt))
    assert isinstance(attempt.outcome, Applied)
    tree = attempt.outcome.effects.produced[0].revision
    comparison = model_diff(workspace, tree, tree)
    assert comparison.model_dump(mode="json") == {"changes": {}}
    ref = store.dynamical_model_spec_ref(tree)
    assert ref.path == "result.msgpack"
    assert msgpack.unpackb(repository.read_file(tree, ref.path))[
        "dynamical_model_spec"
    ] == dynamical_model_spec.model_dump(mode="json")


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([[[1.0, np.nan]], [[np.inf, -np.inf]]], dtype=np.float64),
        np.asarray([-(2**63), 2**63 - 1], dtype=np.int64),
        np.asarray([0, 2**64 - 1], dtype=np.uint64),
        np.asarray([[True, False], [False, True]]),
        np.asfortranarray(np.arange(12, dtype=">f8").reshape(3, 4)),
        np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2],
    ],
    ids=["nonfinite", "int64", "uint64", "boolean", "big_endian_fortran", "strided"],
)
def test_published_arrays_remain_readable_after_execution_buffers_are_removed(
    monkeypatch, tmp_path, values
):
    from grassmoth.actions.contracts.bodies import EditModelOutput
    from grassmoth.artifacts.identity import GitOid
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth_server.actions.contracts.inputs import EditModelInput
    from grassmoth_server.actions.contracts.requests import EditModelRequest
    from grassmoth_server.study.records import EditAttempt
    from tests.support.git import git_oid

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, repository = ArtifactStore("BUFFERS"), StudyRepository("BUFFERS")
    identity = store.write_array(values)
    result = EditModelOutput(
        dynamical_model_spec=DynamicalModelSpec.model_validate(
            {
                "distributions": {
                    "distribution:stored": {
                        "distribution": "Delta",
                        "params": {
                            "v": {
                                "array_ref": identity,
                                "shape": list(values.shape),
                                "dtype": str(values.dtype.newbyteorder("<")),
                                "index": [],
                            }
                        },
                    }
                }
            }
        ),
        checks={"specification": (), "question": {"findings": ()}},
        identification={"outcome": None, "treatments": {}},
        pruning={},
    )
    ref = store.write_result(result)
    request = EditModelRequest[GitOid](
        input=EditModelInput[GitOid](
            parent_ref=git_oid(1), dynamical_model_spec=DynamicalModelSpec()
        )
    )
    publication = AttemptRecord(
        seq=1,
        ts="2026-01-01T00:00:00Z",
        attempt=EditAttempt(
            action="edit_model",
            request=request,
            outcome=Applied(result=ref, effects=ActionEffects()),
        ),
    )
    from grassmoth_server.storage import arrays

    read_staging = arrays.read_array_bytes

    def publish_during_staging_read(directory, selected):
        repository.append(publication)
        return read_staging(directory, selected)

    with monkeypatch.context() as concurrent:
        concurrent.setattr(arrays, "read_array_bytes", publish_during_staging_read)
        np.testing.assert_array_equal(store.read_array(identity), values)
    assert not (tmp_path / "BUFFERS/store/arrays" / f"{identity}.npy").exists()
    np.testing.assert_array_equal(ArtifactStore("BUFFERS").read_array(identity), values)
    payload = unpack_result(repository.read_file(repository.head(), "result.msgpack"))
    assert payload == {
        "dynamical_model_spec": {
            "distributions": {
                "distribution:stored": {
                    "distribution": "Delta",
                    "params": {"v": _array_value(values).model_dump(mode="python")},
                }
            }
        },
        "checks": {"specification": [], "question": {"findings": []}},
        "identification": {"outcome": None, "treatments": {}},
        "pruning": {"constructs": [], "edges": [], "parameters": [], "distributions": []},
    }


def test_domain_views_share_one_wire_buffer_and_decode_without_a_store():
    from grassmoth.artifacts.arrays import ArrayVector, resolve_vector
    from grassmoth.artifacts.base import Value

    class NumericalViews(Value):
        owner: NumericalArray
        view: ArrayVector

    array = NumericalArray.from_numpy(np.arange(12, dtype=np.float64).reshape(3, 4))
    result = NumericalViews(owner=array, view=ArrayVector(array=array, indices=(None, 2)))
    encoded = pack_result(result)
    assert encoded.count(array.npy) == 1
    payload = unpack_result(encoded)
    restored = NumericalViews.model_validate(payload)
    assert restored.owner.npy is restored.view.array.npy
    assert np.shares_memory(restored.owner.values, restored.view.array.values)
    assert not restored.owner.values.flags.writeable
    assert resolve_vector(restored.view) == (2.0, 6.0, 10.0)
    assert result.content_digest() == restored.content_digest()
    with pytest.raises(ValueError, match="earlier"):
        unpack_result(msgpack.packb({"npy": msgpack.ExtType(42, (0).to_bytes(8, "big"))}))


def test_restored_distribution_keeps_its_owned_buffer_when_published_again(monkeypatch):
    from grassmoth.artifacts.base import Value
    from grassmoth.artifacts.probability.serialization import NumPyroDistribution
    from grassmoth_server.study.persistence.arrays import owned_arrays
    from grassmoth_server.study.persistence.codec import result_payload

    class Law(Value):
        distribution: NumPyroDistribution

    array = NumericalArray.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4))
    law = Law.model_validate(
        {
            "distribution": {
                "distribution": "Delta",
                "params": {"v": {"npy": array.npy}, "event_dim": 1},
            }
        }
    )
    restored = Law.model_validate(unpack_result(pack_result(law)))
    owned = owned_arrays(result_payload(restored))

    def forbid_reencoding(_values):
        raise AssertionError("An owned binary distribution must not be re-encoded")

    monkeypatch.setattr(NumericalArray, "from_numpy", forbid_reencoding)
    published = result_payload(restored)
    assert owned_arrays(published)[array.identity].npy is owned[array.identity].npy
    assert unpack_result(pack_result(restored)) == published
    assert restored.content_digest() == law.content_digest()
