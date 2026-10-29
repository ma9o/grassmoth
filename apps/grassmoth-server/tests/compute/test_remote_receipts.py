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
from grassmoth_server.compute.remote_receipts import (
    RemoteFitReceipt,
    complete_remote_body,
    published_fit_report,
    receipt_for,
)
from grassmoth_server.storage import paths
from grassmoth_server.study.persistence.arrays import owned_arrays
from grassmoth_server.study.persistence.codec import pack_result, unpack_result
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore, read_model
from grassmoth_server.study.queries.action_outputs import completed_call_msgpack
from grassmoth_server.study.records import Applied
from grassmoth_server_test_support.action import applied_record
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
    call = AnalysisCall(workspace_id="RECEIPT", call_id=call_identity(request))
    body = pack_result(result)
    entry = cache_fit_result(tmp_path / "volume" / "retained", call, body)
    receipt = receipt_for(result, entry, tuple(owned_arrays(unpack_result(body))))
    return result, body, receipt, request


def test_publication_needs_no_download_and_full_reads_preserve_body(retained_case, monkeypatch):
    result, body, receipt, request = retained_case
    from grassmoth_server.actions.temporal.study.activities import (
        _journal_attempt,
        evaluate_model_checks_activity,
    )
    from grassmoth_server.actions.temporal.study.messages import (
        AttemptPublication,
        EvaluateChecksInput,
    )
    from grassmoth_server.study.state import StudyState
    from grassmoth_server_test_support.helpers import run_async

    volume = Mock(
        read_file_into_fileobj=Mock(side_effect=AssertionError("Publication downloaded evidence"))
    )
    monkeypatch.setattr(modal.Volume, "from_name", lambda _name: volume)
    store = ArtifactStore(receipt.cache.call.workspace_id)
    reference = store.write_result(receipt)
    artifact = store.result_artifact(
        store.write_artifact("model", derived_from={}, produced_by="fit"), reference
    )
    applied = Applied(result=reference, effects=ActionEffects(produced=(artifact,)))
    run_async(
        evaluate_model_checks_activity(
            EvaluateChecksInput(
                workspace_id=store.workspace_id,
                state=StudyState(),
                applied=applied,
                request=request,
            )
        )
    )
    record = applied_record(store.workspace_id, applied, seq=1, request=request)
    revision = _journal_attempt(AttemptPublication(workspace_id=store.workspace_id, record=record))
    assert published_fit_report(store, reference) == receipt
    assert len(pack_result(receipt)) < 100_000 < len(body)
    assert not receipt.inference.evidence.phase_extra_fields
    assert receipt.checks == result.checks
    assert receipt.inference.core.convergence == result.inference.core.convergence
    assert (
        len(StudyRepository(store.workspace_id).read_file(revision.commit_id, "result.msgpack"))
        < 100_000
    )
    volume.read_file_into_fileobj.assert_not_called()

    monkeypatch.setattr(
        volume, "read_file_into_fileobj", Mock(side_effect=lambda _path, stream: stream.write(body))
    )
    assert store.read_result_bytes(reference) == body
    assert store.read_result(reference, FitOutput) == result
    assert read_model(store, artifact.revision) == result.dynamical_model_spec
    identity = receipt.array_ids[0]
    assert store.read_array_bytes(identity) == owned_arrays(unpack_result(body))[identity].npy
    assert unpack_result(completed_call_msgpack(store.workspace_id, revision))[
        "body"
    ] == unpack_result(body)
    volume.read_file_into_fileobj.assert_called_once()


@pytest.mark.parametrize("failure", ["checksum", "truncated", "interrupted"])
def test_corrupt_or_partial_download_is_never_installed(
    retained_case, monkeypatch, tmp_path, failure
):
    _result, body, receipt, _request = retained_case

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
        complete_remote_body(pack_result(receipt), root)
    assert not tuple((root / "results").iterdir())


def test_gpu_commits_complete_body_before_returning_only_receipt(
    retained_case, monkeypatch, tmp_path
):
    import jax

    from grassmoth_server.actions.fit import remote
    from grassmoth_server.compute import compilation_cache

    result, body, receipt, _request = retained_case
    monkeypatch.setenv(remote.FIT_CACHE_VOLUME_ENV, "test-compilation-cache")
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(remote, "ANALYSIS_CACHE_ROOT", tmp_path / "worker")
    monkeypatch.setattr(remote, "execute_fit_compute", lambda _payload: result)
    monkeypatch.setattr(
        compilation_cache, "local_compilation_cache", lambda *_args, **_kwargs: nullcontext()
    )
    volume = Mock()
    monkeypatch.setattr(modal.Volume, "from_name", lambda _name: volume)
    chunks = remote._fit_gpu(Mock(analysis_call=receipt.cache.call), Mock())
    decoded = remote.decode_fit_result(b"".join(chunks))
    returned = RemoteFitReceipt.model_validate(unpack_result(decoded.output))
    assert returned == receipt
    assert (
        tmp_path / "worker" / "retained" / "results" / (receipt.cache.sha256 + ".msgpack")
    ).read_bytes() == body
    volume.commit.assert_called_once()


def test_dispatch_rejects_receipt_for_another_call(retained_case, monkeypatch):
    from grassmoth.public import FitInputs, SamplerSpec
    from grassmoth_server.actions.contracts.identity import CallId
    from grassmoth_server.actions.fit import remote
    from grassmoth_server_test_support.fit_cache import fixture

    _result, _body, receipt, _request = retained_case
    question, selection, data = fixture()
    monkeypatch.setattr(
        remote, "_dispatch_fit", Mock(return_value=remote.FitComputeResult(pack_result(receipt)))
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
            analysis_call=receipt.cache.call.revised(call_id=CallId("call:" + "f" * 64)),
        )
