"""HTTP schema and full native response retain an execution's stopping evidence."""

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from grassmoth.artifacts.model.question import QuestionSpec
from grassmoth.artifacts.observations.dataset import ObservationDataset
from grassmoth.artifacts.reports.checks import Evaluated, NumericCriterionEvidence
from grassmoth.artifacts.reports.posterior_diagnostics import ConvergenceReport
from grassmoth.artifacts.sampling import (
    ConvergenceStoppingSpec,
    MarginalParticleGibbsSpec,
    NutsDASpec,
    NutsTrajectorySpec,
    SamplerSpec,
    TwistedParticleAGradSpec,
)
from grassmoth.models.ssm.inference.diagnostics import convergence
from grassmoth.models.structure import StructuralSelection
from grassmoth_server.actions.contracts.identity import CallId
from grassmoth_server.actions.contracts.polling import ActionPoll, ActionResult, SuccessfulPoll
from grassmoth_server.actions.contracts.requests import ActionInput, FitRequest
from grassmoth_server.actions.fit.action import run_fit
from grassmoth_server.api.app import app
from grassmoth_server.study import calls
from grassmoth_server.study.persistence.codec import pack_result, unpack_result
from grassmoth_test_support.git import git_oid
from grassmoth_test_support.inference import nonlinear_fit_fixture
from grassmoth_test_support.model import load_model_fixture


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(
    "algorithm",
    [
        MarginalParticleGibbsSpec(path_update=TwistedParticleAGradSpec(num_particles=8)),
        NutsDASpec(trajectory=NutsTrajectorySpec(max_tree_depth=2)),
    ],
)
def test_http_stopping_request_and_native_response_round_trip(algorithm, monkeypatch):
    model = load_model_fixture(
        "ecosystem_consumer/nonlinear_mixed_missing_irregular_particle_fit_and_exact_diagnostics_nonlinear_model.json"
    )
    _, panel = nonlinear_fit_fixture()
    data = ObservationDataset.from_frame(
        pl.DataFrame(panel.rows),
        tuple(item.observation for item in panel.compiled_dynamical_model.observations),
        time_origin=panel.time_origin,
    )
    passing = ConvergenceReport(
        findings=(
            Evaluated(
                code="test_gate",
                subject="recorded_parameter_chains",
                outcome="passed",
                evidence=NumericCriterionEvidence(criterion="test", value=1.0),
            ),
        )
    )
    monkeypatch.setattr(convergence, "chain_convergence", lambda _chains: passing)
    policy = ConvergenceStoppingSpec(checkpoint_draws=4, min_draws=8)
    settings = SamplerSpec(
        algorithm=algorithm,
        num_warmup=40,
        num_samples_per_chain=12,
        num_chains=2,
        seed=19,
        hardware="local",
        stop_when_converged=policy,
    )
    output = run_fit(
        question=QuestionSpec(text="Stopping response", outcome=model.constructs[0].id),
        selection=StructuralSelection(model, None),
        data_for_model=data,
        time_origin=panel.time_origin,
        sampler=settings,
        compute_loo_diagnostics=False,
    )
    call_id = CallId("call:" + "a" * 64)
    poll = SuccessfulPoll(
        call_id=call_id, action="fit", commit_id=git_oid(1), body=output, messages=()
    )
    request_body = {
        "action": "fit",
        "input": {
            "target": {
                "kind": "new",
                "dynamical_model_spec_ref": "latest",
                "data_ref": {"revision": "latest", "replicate_index": 0},
            },
            "settings": settings.model_dump(mode="json"),
        },
    }
    parsed = TypeAdapter(ActionInput).validate_python(request_body)
    assert isinstance(parsed, FitRequest)
    assert parsed.input.settings.stop_when_converged == policy
    assert TypeAdapter(ActionInput).validate_json(parsed.model_dump_json()) == parsed

    async def submit(_workspace, request, _provider):
        assert isinstance(request, FitRequest)
        assert request.input.settings.stop_when_converged == policy
        return ActionResult(payload=pack_result(poll))

    monkeypatch.setattr(calls, "submit_action", submit)
    response = TestClient(app).post(
        "/api/studies/TEST/fit", json=request_body, headers={"Accept": "application/msgpack"}
    )
    assert response.status_code == 200, response.text
    restored = TypeAdapter(ActionPoll).validate_python(unpack_result(response.content))
    assert isinstance(restored, SuccessfulPoll)
    assert restored.action == "fit"
    metadata = restored.body.inference.core.inference_metadata
    assert metadata.num_samples_total == 16
    (segment,) = metadata.segments
    assert (segment.start, segment.stop) == (0, 8)
    assert segment.settings.num_samples_per_chain == 12
    assert segment.convergence_stopping is not None
    assert segment.convergence_stopping.retained_draws == 8
    assert segment.convergence_stopping.checkpoint is not None
    assert segment.convergence_stopping.checkpoint.draws == 8
    assert segment.convergence_stopping.checkpoint.report == passing
    assert restored.body.inference.evidence.checkpoint is not None
    assert np.isfinite(restored.body.inference.evidence.checkpoint.parameters.values).all()
