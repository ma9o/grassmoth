"""Persistence and refit lifecycle with mocked inference, never numerical fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import numpy as np
import pytest

from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.observations.reference import DataRef
from grassmoth_server.actions.contracts.inputs import FitInput, NewFitTarget
from grassmoth_server.actions.contracts.requests import FitRequest
from grassmoth_server.actions.contracts.settings import FitSettingsSpec
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

pytestmark = pytest.mark.contract

if TYPE_CHECKING:
    from grassmoth_server.actions.contracts.identity import ArtifactId
    from grassmoth_server.study.persistence.store import ArtifactStore


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
        QuestionSpec(
            text="What does the model imply about sleep?",
            outcome=construct_named(authored, "Sleep").id,
        ),
    )
    pins: dict[ArtifactId, GitOid] = {
        "model": original.revision,
        "panel": panel.revision,
        "question": question.revision,
    }
    from grassmoth.actions.contracts.bodies import FitOutput
    from grassmoth_server.study.queries import data
    from grassmoth_server_test_support.action import model_findings

    assert isinstance(request.input.target, NewFitTarget)
    expected_checks = model_findings(
        integration_workspace,
        state.with_artifacts([question]).revised(data=request.input.target.data_ref),
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

    model = fx.seed_model(artifact_store)
    panel = fx.seed_panel(artifact_store, model_revision=model.revision)
    question = write_question(artifact_store)
    compute = Mock(
        return_value=ModelFitFailure.model_validate(
            {"error": "unsupported", "duration_seconds": 0.01}
        )
    )
    write_result = Mock()
    write_artifact = Mock()
    monkeypatch.setattr(flow, "fit", compute)
    monkeypatch.setattr(type(artifact_store), "write_result", write_result)
    monkeypatch.setattr(type(artifact_store), "write_artifact", write_artifact)
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
    write_artifact.assert_not_called()
