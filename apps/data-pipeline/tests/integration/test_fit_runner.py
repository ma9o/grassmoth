"""Persistence and refit lifecycle with mocked inference, never numerical fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import numpy as np
import pytest

from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.observations.reference import DataRef
from nof1_causal_service.actions.contracts.inputs import FitInput, NewFitTarget
from nof1_causal_service.actions.contracts.requests import FitRequest
from nof1_causal_service.actions.runners import run_action_locally
from nof1_causal_service.study.persistence.store import read_model
from nof1_causal_service.study.state import apply_effects
from tests.integration import runner_fixtures as fx
from tests.support.git import artifact_revisions
from tests.support.helpers import run_async, write_question
from tests.support.inference import (
    bind_panel_fixture,
    compile_model_fixture,
    model_draws,
    parameter_draws,
    particle_posterior,
)

pytestmark = pytest.mark.contract

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId
    from nof1_causal_service.study.persistence.store import ArtifactStore


def test_inference_advances_model_and_uses_the_selected_input(
    integration_workspace: str,
    artifact_store: ArtifactStore,
    monkeypatch,
) -> None:
    import jax.numpy as jnp

    from nof1_causal_lab.models.ssm import compiled_access as numerics
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
    from nof1_causal_service.actions.fit import execution as stage5_fit

    original = fx.seed_model(artifact_store)
    panel = fx.seed_panel(artifact_store, model_revision=original.revision)
    authored = read_model(artifact_store, original.revision)
    parameters = parameter_draws(authored, 4)
    states = tuple(numerics.state_ids(compile_model_fixture(authored)))
    paths = jnp.arange(4 * 2 * len(states), dtype=jnp.float32).reshape(4, 2, len(states))
    fitted_inputs = []

    def fake_fit_model(selection, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        model = selection.dynamical_model_spec
        fitted_inputs.append(model.model_dump(mode="json"))
        return {
            "fitted": True,
            "duration_seconds": 0.01,
            "result": particle_posterior(
                draws=JointPosteriorDraws(
                    parameters=parameters, latent_paths=paths, state_ids=states
                )
            ),
            "panel": bind_panel_fixture(
                compile_model_fixture(model),
                jnp.array([[jnp.nan, jnp.nan], [0.0, 0.0]]),
                jnp.array([0.0, 1.0], dtype=jnp.float32),
            ),
        }

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
        )
    )
    from nof1_causal_lab.artifacts.model.question import QuestionSpec
    from tests.support.model import construct_named

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
    from nof1_causal_lab.actions import checks
    from nof1_causal_lab.actions.contracts.bodies import FitOutput
    from nof1_causal_service.study.queries import data
    from tests.support.action import model_findings

    assert isinstance(request.input.target, NewFitTarget)
    expected_checks = model_findings(
        integration_workspace,
        state.with_artifacts([question]).revised(data=request.input.target.data_ref),
        action="fit",
    )
    history_reader = Mock(wraps=data.read_data_history)
    preflight = Mock(wraps=checks.check_model_data)
    monkeypatch.setattr(data, "read_data_history", history_reader)
    monkeypatch.setattr(checks, "check_model_data", preflight)
    for index in range(2):
        assert isinstance(request.input.target, NewFitTarget)
        applied = run_async(run_action_locally(integration_workspace, request, pins))
        assert history_reader.call_count == preflight.call_count == index + 1
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


@pytest.mark.parametrize("evaluated", [False, True])
def test_failed_preflight_still_stops_before_compute_or_reports(
    integration_workspace, artifact_store, monkeypatch, evaluated
):
    from nof1_causal_lab.actions import checks
    from nof1_causal_lab.actions.errors import ModelFitError
    from nof1_causal_lab.artifacts.reports.checks import Evaluated, NotEvaluated
    from nof1_causal_service.actions.fit import action as flow

    model = fx.seed_model(artifact_store)
    panel = fx.seed_panel(artifact_store, model_revision=model.revision)
    question = write_question(artifact_store)
    finding = (
        Evaluated(code="fit_preflight", subject="model", outcome="failed", evidence="unsupported")
        if evaluated
        else NotEvaluated(
            code="fit_preflight", subject="model", reason="MODEL_INCOMPLETE", detail="incomplete"
        )
    )
    monkeypatch.setattr(checks, "check_model_data", Mock(return_value=(finding,)))
    compute = Mock()
    write_report = Mock()
    monkeypatch.setattr(flow, "fit", compute)
    monkeypatch.setattr(type(artifact_store), "write_report", write_report)
    request = FitRequest[GitOid](
        input=FitInput[GitOid](
            target=NewFitTarget(
                dynamical_model_spec_ref=model.revision,
                data_ref=DataRef(revision=panel.revision, replicate_index=0),
            ),
        )
    )
    with pytest.raises(ModelFitError, match="preflight rejected") as caught:
        run_async(
            run_action_locally(
                integration_workspace,
                request,
                {"model": model.revision, "question": question.revision},
            )
        )
    assert caught.value.diagnostics == {"findings": [finding.model_dump(mode="json")]}
    compute.assert_not_called()
    write_report.assert_not_called()
