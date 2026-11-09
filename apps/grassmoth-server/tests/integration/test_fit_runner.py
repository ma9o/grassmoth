"""Persistence and refit lifecycle with mocked inference, never numerical fitting."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock
from uuid import UUID

import numpy as np
import pytest

from grassmoth.artifacts.identity import ConstructId, GitOid
from grassmoth.artifacts.observations.reference import DataRef
from grassmoth.public import SamplerSpec
from grassmoth_server.actions.call_logs import collect_call_log
from grassmoth_server.actions.contracts.call_state import RunningCall
from grassmoth_server.actions.contracts.inputs import FitInput, NewFitTarget
from grassmoth_server.actions.contracts.requests import FitRequest
from grassmoth_server.actions.contracts.settings import FitSettingsSpec
from grassmoth_server.actions.fit.capacity import FitAdmission
from grassmoth_server.actions.runners import run_action_locally
from grassmoth_server.study.persistence.store import read_model
from grassmoth_server.study.state import apply_effects
from grassmoth_server_test_support import runner_fixtures as fx
from grassmoth_server_test_support.git import artifact_revisions
from grassmoth_server_test_support.helpers import run_async, write_question
from grassmoth_test_support.inference import (
    bind_panel_fixture,
    compile_model_fixture,
    model_draws,
    parameter_draws,
    particle_posterior,
)
from grassmoth_test_support.values import revise_fixture

pytestmark = pytest.mark.contract

if TYPE_CHECKING:
    from grassmoth_server.actions.contracts.identity import ArtifactId
    from grassmoth_server.study.persistence.store import ArtifactStore


def _admit_without_execution(launch, *_args, **_kwargs):
    return FitAdmission(
        sampler=launch.resolve(2),
        detail="Sampling demand fits the configured host and device budgets.",
    )


def _capacity_messages(workspace_id, request):
    log = collect_call_log(
        workspace_id,
        RunningCall(attempt_id=UUID(int=1), seq=1, request=request),
        at=datetime.now(UTC),
    )
    return tuple(
        message
        for message in log.messages
        if message.kind == "log" and message.code == "FIT_CAPACITY_ESTIMATE"
    )


def test_inference_advances_model_and_uses_the_selected_input(
    integration_workspace: str,
    artifact_store: ArtifactStore,
    monkeypatch,
) -> None:
    import jax.numpy as jnp

    from grassmoth.actions.fit import execution as stage5_fit
    from grassmoth.actions.fit.execution import FittedModelResult
    from grassmoth.models.ssm import compiled_access as numerics
    from grassmoth.models.ssm.inference.types import JointPosteriorDraws
    from grassmoth_server.actions.fit import capacity

    monkeypatch.setattr(capacity, "admit_sampler", _admit_without_execution)

    original = fx.seed_model(artifact_store)
    panel = fx.seed_panel(artifact_store, model_revision=original.revision)
    authored = read_model(artifact_store, original.revision)
    parameters = parameter_draws(authored, 4)
    states = tuple(numerics.state_ids(compile_model_fixture(authored)))
    paths = jnp.arange(4 * 2 * len(states), dtype=jnp.float32).reshape(4, 2, len(states))
    fitted_inputs = []

    def fake_fit_model(inputs, *_args: Any, **_kwargs: Any) -> FittedModelResult:
        model = inputs.selection.dynamical_model_spec
        fitted_inputs.append(model.model_dump(mode="json"))
        return FittedModelResult.from_run(
            duration_seconds=0.01,
            result=particle_posterior(
                draws=JointPosteriorDraws(
                    parameters=parameters, latent_paths=paths, state_ids=states
                )
            ),
            panel=bind_panel_fixture(
                compile_model_fixture(model),
                jnp.array([[jnp.nan, jnp.nan], [0.0, 0.0]]),
                jnp.array([0.0, 1.0], dtype=jnp.float32),
            ),
        )

    monkeypatch.setattr(stage5_fit, "fit_model", fake_fit_model)
    state = fx.state_from(original, panel)
    revisions = [original.revision]
    # Refit the selected authored revision; conditioned joint laws cannot be fit inputs.
    request = FitRequest[GitOid](
        input=FitInput[GitOid](
            target=NewFitTarget(
                dynamical_model_spec_ref=original.revision,
                data_ref=DataRef(revision=panel.revision, replicate_index=0),
            ),
            settings=FitSettingsSpec(num_chains=1, num_samples_per_chain=4),
        )
    )
    from grassmoth.artifacts.model.question import QuestionSpec
    from grassmoth_test_support.model import construct_named

    question = write_question(
        artifact_store,
        QuestionSpec[ConstructId](
            text="What does the model imply about sleep?",
            outcome=construct_named(authored, "Sleep").id,
        ),
    )
    pins: dict[ArtifactId, GitOid] = {
        "model": original.revision,
        "panel": panel.revision,
        "question": question.revision,
    }
    from grassmoth.actions.contracts.io import FitOutput
    from grassmoth_server.study.queries import data
    from grassmoth_server_test_support.action import model_findings

    assert isinstance(request.input.target, NewFitTarget)
    expected_checks = model_findings(
        integration_workspace,
        revise_fixture(state.with_artifacts([question]), data=request.input.target.data_ref),
        action="fit",
    )
    history_reader = Mock(wraps=data.read_data_history)
    monkeypatch.setattr(data, "read_data_history", history_reader)
    for index in range(2):
        assert isinstance(request.input.target, NewFitTarget)
        applied = run_async(run_action_locally(integration_workspace, request, pins))
        assert history_reader.call_count == index + 1
        assert artifact_store.read_result(applied.result, FitOutput).checks == expected_checks
        info = next(info for info in applied.effects.produced if info.artifact_id == "model")
        assert info.revision not in revisions
        revisions.append(info.revision)
        # Both provenance and computation follow the selected model revision.
        assert info.derived_from == pins
        assert {
            "model": request.input.target.dynamical_model_spec_ref,
            "panel": request.input.target.data_ref.revision,
            "question": question.revision,
        } == info.derived_from

        report = artifact_store.read_result(applied.result, FitOutput).inference
        assert report.core.inference_diagnostics is not None
        assert report.core.inference_diagnostics.num_chains == 1
        assert report.core.inference_diagnostics.num_samples_per_chain == 4
        assert report.core.inference_metadata.num_samples_total == 4
        assert (
            report.core.inference_metadata.distribution
            in read_model(artifact_store, info.revision).law_layouts
        )
        conditioned = read_model(artifact_store, info.revision)
        assert type(conditioned) is type(authored)
        assert conditioned.distributions
        retained = model_draws(compile_model_fixture(conditioned))
        np.testing.assert_array_equal(retained.latent_paths, paths)
        for name, values in parameters.items():
            np.testing.assert_array_equal(retained.parameters[name], values)
        state = apply_effects(state, applied.effects.produced, applied.effects.retracted)
        assert state.current["model"].produced_by == "fit"
    assert fitted_inputs == [
        read_model(artifact_store, original.revision).model_dump(mode="json") for _ in range(2)
    ]
    assert read_model(artifact_store, original.revision).model_dump(mode="json") == fitted_inputs[0]
    assert artifact_revisions(artifact_store, "model") == revisions


def test_library_fit_failure_stops_before_publication(
    integration_workspace, artifact_store, monkeypatch
):
    from grassmoth.actions.fit.execution import ModelFitFailure
    from grassmoth_server.actions.errors import ModelFitError
    from grassmoth_server.actions.fit import action as flow
    from grassmoth_server.actions.fit import capacity

    monkeypatch.setattr(capacity, "admit_sampler", _admit_without_execution)

    model = fx.seed_model(artifact_store)
    panel = fx.seed_panel(artifact_store, model_revision=model.revision)
    question = write_question(artifact_store)
    compute = Mock(
        return_value=ModelFitFailure.model_validate(
            {"error": "unsupported", "duration_seconds": 0.01}
        )
    )
    write_result = Mock()
    result_record = Mock()
    monkeypatch.setattr(flow, "fit", compute)
    monkeypatch.setattr(type(artifact_store), "write_result", write_result)
    monkeypatch.setattr(type(artifact_store), "result_record", result_record)
    request = FitRequest[GitOid](
        input=FitInput[GitOid](
            target=NewFitTarget(
                dynamical_model_spec_ref=model.revision,
                data_ref=DataRef(revision=panel.revision, replicate_index=0),
            ),
        )
    )
    with pytest.raises(ModelFitError, match="unsupported") as caught:
        run_async(
            run_action_locally(
                integration_workspace,
                request,
                {"model": model.revision, "question": question.revision},
            )
        )
    assert caught.value.diagnostics == {"duration_seconds": 0.01}
    compute.assert_called_once()
    write_result.assert_not_called()
    result_record.assert_not_called()


def test_capacity_rejection_stops_before_remote_dispatch(
    integration_workspace, artifact_store, monkeypatch
):
    from grassmoth_server.actions.fit import capacity, remote
    from grassmoth_server.study.records import Rejected

    model = fx.seed_model(artifact_store)
    panel = fx.seed_panel(artifact_store, model_revision=model.revision)
    question = write_question(artifact_store)
    rejection = Rejected(code="FIT_CAPACITY_EXCEEDED", subject="fit", detail="Insufficient memory")
    admission, dispatch = Mock(return_value=rejection), Mock()
    monkeypatch.setattr(capacity, "admit_sampler", admission)
    monkeypatch.setattr(remote, "fit_on_modal", dispatch)
    request = FitRequest[GitOid](
        input=FitInput[GitOid](
            target=NewFitTarget(
                dynamical_model_spec_ref=model.revision,
                data_ref=DataRef(revision=panel.revision, replicate_index=0),
            ),
            settings=FitSettingsSpec(hardware="L40S", num_chains="auto"),
        )
    )
    result = run_async(
        run_action_locally(
            integration_workspace, request, {"model": model.revision, "question": question.revision}
        )
    )
    assert result == rejection
    admission.assert_called_once()
    dispatch.assert_not_called()
    assert not _capacity_messages(integration_workspace, request)


def test_capacity_estimate_is_persisted_before_remote_dispatch(
    integration_workspace, artifact_store, monkeypatch
):
    from grassmoth_server.actions.fit import capacity, remote

    model = fx.seed_model(artifact_store)
    panel = fx.seed_panel(artifact_store, model_revision=model.revision)
    question = write_question(artifact_store)
    admitted = FitAdmission(
        sampler=SamplerSpec(hardware="L40S", num_chains=3),
        detail=(
            "L40S: N=64, M=8, C=3. Estimated sampling memory: "
            "host=1 KiB / 4 KiB (25.00%); device=2 KiB / 8 KiB (25.00%)."
        ),
    )
    monkeypatch.setattr(capacity, "admit_sampler", Mock(return_value=admitted))
    request = FitRequest[GitOid](
        input=FitInput[GitOid](
            target=NewFitTarget(
                dynamical_model_spec_ref=model.revision,
                data_ref=DataRef(revision=panel.revision, replicate_index=0),
            ),
            settings=FitSettingsSpec(hardware="L40S", num_chains="auto"),
        )
    )

    def inspect_dispatch(inputs, **_kwargs):
        assert inputs.sampler == admitted.sampler
        (message,) = _capacity_messages(integration_workspace, request)
        assert message.severity == "info"
        assert message.subject == "fit"
        assert message.detail == admitted.detail
        raise RuntimeError("Reached mocked remote dispatch")

    dispatch = Mock(side_effect=inspect_dispatch)
    monkeypatch.setattr(remote, "fit_on_modal", dispatch)
    with pytest.raises(RuntimeError, match="Reached mocked remote dispatch"):
        run_async(
            run_action_locally(
                integration_workspace,
                request,
                {"model": model.revision, "question": question.revision},
            )
        )
    dispatch.assert_called_once()
    assert len(_capacity_messages(integration_workspace, request)) == 1
