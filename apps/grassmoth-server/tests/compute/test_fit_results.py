"""Remote publication stays compact while explicit reads retain complete scientific evidence."""

from contextlib import nullcontext
from unittest.mock import Mock

import modal
import numpy as np
import pytest

from grassmoth.public import DataRef, FitOutput, NumericalArray
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.actions.contracts.inputs import FitInput, NewFitTarget
from grassmoth_server.actions.contracts.requests import FitRequest, call_identity
from grassmoth_server.compute.analysis_cache import AnalysisCall, cache_fit_result
from grassmoth_server.compute.fit_results import RemoteFitResult, download_fit_result, load_full_fit
from grassmoth_server.storage import paths
from grassmoth_server.study.persistence.arrays import owned_arrays
from grassmoth_server.study.persistence.codec import pack_result, unpack_result
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore, read_model
from grassmoth_server.study.queries.action_outputs import completed_call_msgpack
from grassmoth_server.study.records import Applied
from grassmoth_test_support.git import git_oid

pytestmark = pytest.mark.contract


@pytest.fixture
def retained_case(tmp_path, monkeypatch):
    from grassmoth.artifacts.reports.data_quality import DataProfileReport
    from grassmoth.artifacts.reports.inference import FitCheckReport
    from grassmoth.artifacts.reports.model_checks import QuestionCheckReport
    from grassmoth_test_support.inference import _report
    from grassmoth_test_support.model import x_y_model

    monkeypatch.setattr(paths, "_DATA_URI", str(tmp_path / "study"))
    model = x_y_model()
    report = _report(model)
    values = NumericalArray.from_numpy(np.arange(2_100_000, dtype=np.float32))
    result = FitOutput(
        dynamical_model_spec=model,
        checks=FitCheckReport(
            data=DataProfileReport(indicators={}, findings=()),
            preflight=(),
            question=QuestionCheckReport(findings=()),
        ),
        inference=report.revised(
            evidence=report.evidence.revised(
                phase_extra_fields={"initialization": {"position": values}}
            )
        ),
    )
    request = FitRequest(
        input=FitInput(
            target=NewFitTarget(
                dynamical_model_spec_ref=git_oid(1),
                data_ref=DataRef(revision=git_oid(2), replicate_index=0),
            )
        )
    )
    call = AnalysisCall(workspace_id="FIT", call_id=call_identity(request))
    body = pack_result(result)
    entry = cache_fit_result(tmp_path / "volume" / "retained", call, unpack_result(body))
    completion = RemoteFitResult(
        full_fit=entry,
        transfer=entry,
        law_arrays=tuple(owned_arrays(unpack_result(body)["dynamical_model_spec"])),
    )
    return result, body, completion, request


def test_published_body_is_local_and_full_pointer_is_storage_only(
    retained_case, monkeypatch, tmp_path
):
    from grassmoth_server.study.records import AttemptRecord, action_attempt

    result, body, completion, request = retained_case
    volume = Mock(
        read_file_into_fileobj=Mock(side_effect=AssertionError("Saved GET contacted Modal"))
    )
    monkeypatch.setattr(modal.Volume, "from_name", lambda _name: volume)
    store = ArtifactStore(completion.full_fit.call.workspace_id)
    path = tmp_path / "transfer.msgpack"
    path.write_bytes(body)
    reference = store.import_result(
        call_id=call_identity(request),
        path=path,
        law_arrays=completion.law_arrays,
        full_fit=completion.full_fit,
    )
    artifact = store.result_record("model", reference, derived_from={}, produced_by="fit")
    applied = Applied(result=reference, effects=ActionEffects(produced=(artifact,)))
    repository = StudyRepository(store.workspace_id)
    revision = repository.append(
        AttemptRecord(seq=1, ts="2026-01-01T00:00:00Z", attempt=action_attempt(request, applied))
    )
    from grassmoth_server.study.persistence.call import SavedApplied

    saved = repository.saved_record(revision.commit_id).outcome
    assert isinstance(saved, SavedApplied)
    assert saved.result.full_fit == completion.full_fit
    assert repository.read_file(revision.commit_id, "result.msgpack") == body
    assert store.read_result_bytes(reference) == body
    assert store.read_result(reference, FitOutput) == result
    assert read_model(store, artifact.revision) == result.dynamical_model_spec
    assert unpack_result(completed_call_msgpack(store.workspace_id, revision))[
        "body"
    ] == unpack_result(body)
    assert {
        ref.removeprefix("refs/arrays/")
        for ref in store.repo.references
        if ref.startswith("refs/arrays/")
    } == set(completion.law_arrays)
    volume.read_file_into_fileobj.assert_not_called()


@pytest.mark.parametrize("failure", ["checksum", "truncated", "interrupted"])
def test_corrupt_or_partial_download_is_never_installed(
    retained_case, monkeypatch, tmp_path, failure
):
    _result, body, completion, _request = retained_case

    def download(_path, stream):
        if failure == "checksum":
            stream.write(bytes([body[0] ^ 1]) + body[1:])
        else:
            stream.write(body[:100])
            if failure == "interrupted":
                raise RuntimeError("Interrupted download")

    monkeypatch.setattr(
        modal.Volume, "from_name", lambda _name: Mock(read_file_into_fileobj=download)
    )
    root = tmp_path / "download"
    with pytest.raises((ValueError, RuntimeError), match=r"identity|Interrupted"):
        download_fit_result(root, completion.transfer)
    assert not tuple((root / "results").iterdir())


@pytest.mark.parametrize("thinned", [False, True])
def test_gpu_commits_full_and_transfer_bodies_before_returning_pointers(
    retained_case, monkeypatch, tmp_path, thinned
):
    import jax

    from grassmoth_server.actions.fit import remote
    from grassmoth_server.compute import compilation_cache

    result, body, completion, _request = retained_case
    if thinned:
        from datetime import UTC, datetime

        import jax.numpy as jnp

        from grassmoth.actions.fit.conditioning import condition_model
        from grassmoth.artifacts import fit_transfer
        from grassmoth.artifacts.probability.serialization import empirical_atoms
        from grassmoth.artifacts.reports.inference import FitRuntime, FitSegment
        from grassmoth.artifacts.sampling import SamplerSpec
        from grassmoth.models.ssm.inference.types import JointPosteriorDraws
        from grassmoth_test_support.inference import (
            compile_model_fixture,
            parameter_draws,
            particle_posterior,
        )

        model = result.dynamical_model_spec
        conditioned, identity = condition_model(
            model,
            compile_model_fixture(model),
            particle_posterior(
                JointPosteriorDraws(parameter_draws(model, 52), jnp.zeros((52, 2, 2)))
            ),
            times=jnp.array([0.0, 1.0]),
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
        )
        metadata = result.inference.core.inference_metadata.revised(
            distribution=identity,
            num_samples_total=52,
            draws_sampled_per_chain=13,
            draws_kept_per_chain=13,
            segments=(
                FitSegment(
                    start=0,
                    stop=13,
                    settings=SamplerSpec(num_samples_per_chain=13),
                    runtime=FitRuntime(source_digest="test", versions={}, compiler={}, devices=()),
                    phases=(),
                    diagnostics=None,
                    convergence=result.inference.core.convergence,
                ),
            ),
        )
        result = result.revised(
            dynamical_model_spec=conditioned,
            inference=result.inference.revised(
                core=result.inference.core.revised(inference_metadata=metadata)
            ),
        )
        body = pack_result(result)
        monkeypatch.setattr(
            fit_transfer,
            "MAX_TRANSFER_DRAW_VALUES",
            12 * empirical_atoms(conditioned.distributions[identity]).shape[1],
        )
    monkeypatch.setenv(remote.FIT_CACHE_VOLUME_ENV, "test-compilation-cache")
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(remote, "ANALYSIS_CACHE_ROOT", tmp_path / "worker")
    monkeypatch.setattr(remote, "execute_fit_compute", lambda _payload: result)
    monkeypatch.setattr(
        compilation_cache, "local_compilation_cache", lambda *_args, **_kwargs: nullcontext()
    )
    volume = Mock()
    monkeypatch.setattr(modal.Volume, "from_name", lambda _name: volume)
    chunks = remote._fit_gpu(Mock(analysis_call=completion.full_fit.call), Mock())
    decoded = remote.decode_fit_result(b"".join(chunks))
    returned = RemoteFitResult.model_validate(unpack_result(decoded.output))
    if not thinned:
        assert returned == completion
    assert (
        tmp_path / "worker" / "retained" / "results" / (returned.full_fit.sha256 + ".msgpack")
    ).read_bytes() == body
    from grassmoth_server.compute.analysis_cache import retained_fit_result

    assert (
        retained_fit_result(tmp_path / "worker" / "retained", returned.full_fit).read_bytes()
        == body
    )
    volume.commit.assert_called_once()
    if thinned:
        monkeypatch.setattr(
            volume,
            "read_file_into_fileobj",
            lambda path, stream: stream.write((tmp_path / "worker" / path).read_bytes()),
        )
        transfer_body = download_fit_result(tmp_path / "local", returned.transfer).read_bytes()
        downloaded = FitOutput.model_validate(unpack_result(transfer_body))
        assert downloaded.inference.core.inference_metadata.draws_kept_per_chain == 3
        assert downloaded.inference.core.inference_metadata.thinning_factor == 5
        assert downloaded.inference.core.inference_metadata.draws_sampled_per_chain == 13
        assert returned.full_fit != returned.transfer
        restored = load_full_fit(tmp_path / "local", returned.full_fit)
        assert restored.inference.core.inference_metadata.draws_kept_per_chain == 13
        assert restored.inference.core.inference_metadata.thinning_factor == 1
        assert pack_result(restored) == body


def test_dispatch_rejects_pointers_for_another_call(retained_case, monkeypatch):
    from grassmoth.public import FitInputs, SamplerSpec
    from grassmoth_server.actions.contracts.identity import CallId
    from grassmoth_server.actions.fit import remote
    from grassmoth_server_test_support.fit_cache import fixture

    _result, _body, completion, _request = retained_case
    question, selection, data = fixture()
    monkeypatch.setattr(
        remote, "_dispatch_fit", Mock(return_value=remote.FitComputeResult(pack_result(completion)))
    )
    with pytest.raises(ValueError, match="different call"):
        remote.fit_on_modal(
            FitInputs(
                selection=selection,
                question=question,
                data_for_model=data,
                sampler=SamplerSpec(hardware="L40S"),
                compute_loo_diagnostics=False,
            ),
            analysis_call=completion.full_fit.call.revised(call_id=CallId("call:" + "f" * 64)),
        )
