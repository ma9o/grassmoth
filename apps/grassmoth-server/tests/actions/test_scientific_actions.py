"""Scientific actions depend on selected inputs, independently of authoring recipes."""

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.actions.contracts.bodies import SimulateOutput
from grassmoth.actions.fit.conditioning import condition_model
from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.observations.reference import DataRef
from grassmoth.artifacts.simulation import SimulationSpec
from grassmoth.models.ssm.inference.types import JointPosteriorDraws
from grassmoth.models.ssm.predictive.parameters import sample_model_laws
from grassmoth.models.ssm.preparation.preflight import ObservationPreflightFailure
from grassmoth.models.structure import StructuralSelection
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.actions.contracts.inputs import SimulateInput
from grassmoth_server.actions.contracts.requests import SimulateRequest
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import Applied
from grassmoth_server.study.state import StudyState
from grassmoth_server_test_support.action import applied_record, empty_simulation_summary
from grassmoth_server_test_support.git import artifact_revision, commit_id
from grassmoth_server_test_support.helpers import write_question
from grassmoth_server_test_support.model import stress_sleep_causal_model, stress_sleep_model
from grassmoth_server_test_support.runner_fixtures import panel_metadata
from grassmoth_test_support.git import git_oid
from grassmoth_test_support.inference import (
    compile_model_fixture,
    parameter_draws,
    particle_posterior,
)
from grassmoth_test_support.model import x_model, x_y_model

if TYPE_CHECKING:
    from grassmoth_server.actions.contracts.identity import ArtifactId


@pytest.mark.inference(concern="predictive")
def test_current_law_sampling_preserves_joint_parameter_atoms():
    dynamical_model_spec = x_y_model()
    original = parameter_draws(dynamical_model_spec, 3)
    # Each atom has distinct coordinated values; independently resampling marginals
    # would produce combinations absent from the joint law.
    original = {
        name: value * jnp.arange(1, 4).reshape((3,) + (1,) * (value.ndim - 1))
        for name, value in original.items()
    }
    conditioned, _ = condition_model(
        dynamical_model_spec,
        compile_model_fixture(dynamical_model_spec),
        particle_posterior(
            JointPosteriorDraws(
                original,
                jnp.zeros((3, 2, 2)),
            )
        ),
        times=jnp.array([0.0, 1.0]),
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
    )
    draws = sample_model_laws(
        compile_model_fixture(conditioned), draws=24, key=jax.random.PRNGKey(4)
    ).parameters
    for draw in range(24):
        assert any(
            all(np.allclose(draws[name][draw], values[atom]) for name, values in original.items())
            for atom in range(3)
        )
    authored = sample_model_laws(
        compile_model_fixture(dynamical_model_spec), draws=5, key=jax.random.PRNGKey(4)
    ).parameters
    assert all(value.shape[0] == 5 and np.isfinite(value).all() for value in authored.values())


@pytest.mark.inference(concern="simulation")
@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize(
    ("fitted_laws", "scientific_model_payload"),
    [
        pytest.param(
            False,
            stress_sleep_causal_model,
            id="False",
        ),
        pytest.param(
            True,
            stress_sleep_causal_model,
            id="True",
        ),
    ],
)
def test_durable_replication_preserves_current_laws_without_comparison(
    tmp_path, monkeypatch, fitted_laws, scientific_model_payload
):
    from grassmoth.artifacts.reports.inference import InferenceEvidence
    from grassmoth_server.actions.runners import run_action_locally
    from grassmoth_server.storage import paths as data_module
    from grassmoth_server.study.persistence.history import StudyRepository
    from grassmoth_server_test_support.helpers import run_async
    from grassmoth_server_test_support.inference import inference_metadata
    from grassmoth_server_test_support.runner_fixtures import panel_frame

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, journal = ArtifactStore("TEST"), StudyRepository("TEST")
    dynamical_model_spec = scientific_model_payload()
    definition = store.write_artifact(
        "model",
        produced_by=None,
        derived_from={},
        json_files={"model.json": dynamical_model_spec.model_dump(mode="json")},
    )
    from grassmoth_server.study.persistence.store import read_model

    dynamical_model_spec = read_model(store, definition.revision)
    from grassmoth.artifacts.model.question import QuestionSpec
    from grassmoth_test_support.model import construct_named

    question = write_question(
        store,
        QuestionSpec(
            text="How does sleep change?", outcome=construct_named(dynamical_model_spec, "Sleep").id
        ),
    )
    produced = [definition, question]
    pins: dict[ArtifactId, GitOid] = {
        "model": artifact_revision("TEST", "model", 1),
        "question": question.revision,
    }
    produced.append(
        store.write_artifact(
            "panel",
            produced_by="prepare_data",
            derived_from={},
            json_files={"metadata.json": panel_metadata().model_dump(mode="json")},
            parquet_files={"panel.parquet": panel_frame(n_days=4)},
        )
    )
    if fitted_laws:
        from grassmoth.actions.fit.conditioning import condition_model
        from grassmoth.models.ssm.inference.types import JointPosteriorDraws
        from grassmoth_test_support.inference import parameter_draws

        dynamical_model_spec, _ = condition_model(
            dynamical_model_spec,
            compile_model_fixture(dynamical_model_spec),
            particle_posterior(
                JointPosteriorDraws(parameter_draws(dynamical_model_spec, 3), jnp.zeros((3, 5, 2)))
            ),
            times=jnp.arange(-1.0, 4.0),
            time_origin=panel_metadata().time_origin,
        )
        from grassmoth_server.study.persistence.writes import write_model_revision

        fitted = write_model_revision(
            store,
            dynamical_model_spec,
            produced_by="fit",
            derived_from={"model": definition.revision, "panel": produced[-1].revision},
        )
        pins["model"] = fitted.revision
        produced.append(fitted)
        from grassmoth.actions.fit.action import inference_report
        from grassmoth_server.study.persistence.store import read_model

        evidence = InferenceEvidence()
        retained = evidence
        prepared = Applied(
            result=retained,
            effects=ActionEffects(
                produced=tuple(produced),
                reports={
                    "inference": store.write_result(
                        inference_report(
                            read_model(store, fitted.revision),
                            read_model(
                                store,
                                store.read_meta("model", fitted.revision).derived_from["model"],
                            ),
                            retained,
                            inference_metadata(dynamical_model_spec),
                        )
                    )
                },
            ),
        )
    else:
        prepared = Applied(result=None, effects=ActionEffects(produced=tuple(produced)))
    from grassmoth_server.actions.contracts.inputs import FitInput, NewFitTarget
    from grassmoth_server.actions.contracts.requests import FitRequest

    request = (
        FitRequest(
            input=FitInput(
                target=NewFitTarget(
                    dynamical_model_spec_ref=definition.revision,
                    data_ref=DataRef(revision=produced[-2].revision, replicate_index=0),
                ),
            )
        )
        if fitted_laws
        else None
    )
    journal.append(
        applied_record("TEST", prepared, request=request, seq=1, ts="2026-09-15T12:00:00Z")
    )
    applied = run_async(
        run_action_locally(
            "TEST",
            SimulateRequest[GitOid](
                input=SimulateInput[GitOid](
                    simulation=SimulationSpec(
                        start=date(2023, 12, 31) if fitted_laws else date(2026, 1, 1), horizon="4d"
                    ),
                    dynamical_model_spec_ref=pins["model"],
                )
            ),
            pins,
        )
    )
    report = store.read_result(applied.result, SimulateOutput).report
    evidence = report.evidence
    assert evidence.arms.action.latent_paths.values.shape == (evidence.draws, 5, 2)
    assert evidence.arms.action.observations.values.shape == (evidence.draws, 5, 2)
    assert set(pins) == {"model", "question"}
    assert report.law is not None
    assert report.law.kind == ("fitted" if fitted_laws else "authored")
    if fitted_laws:
        assert report.law.interpretation == "posterior_predictive"
        assert report.law.fitted_data.revision == produced[2].revision
    assert not applied.effects.produced
    assert any(finding.code.startswith("C5c") for finding in report.findings)
    assert not any(finding.code.startswith("C5d") for finding in report.findings)
    simulation = journal.append(
        applied_record(
            "TEST",
            applied,
            request=SimulateRequest(
                input=SimulateInput(
                    dynamical_model_spec_ref=pins["model"],
                    simulation=SimulationSpec(
                        start=date(2023, 12, 31) if fitted_laws else date(2026, 1, 1), horizon="4d"
                    ),
                )
            ),
            seq=2,
            ts="2026-09-15T12:01:00Z",
        )
    )
    assert simulation.record.attempt.outcome.status == "applied"
    result_ref = simulation.record.attempt.outcome.result
    current = store.read_result(result_ref, SimulateOutput).report
    assert current == report
    edited = store.write_artifact(
        "model",
        produced_by=None,
        derived_from={},
        json_files={
            "model.json": dynamical_model_spec.with_entities(
                edges=(
                    dynamical_model_spec.edges[0].revised(description="Revised justification"),
                    *dynamical_model_spec.edges[1:],
                )
            ).model_dump(mode="json")
        },
    )
    journal.append(
        applied_record(
            "TEST",
            Applied(result=None, effects=ActionEffects(produced=[edited])),
            seq=3,
            ts="2026-09-15T12:02:00Z",
            trace_ids=[],
        )
    )
    retained = ArtifactStore("TEST").read_result(result_ref, SimulateOutput).report
    assert retained == report

    # A retained generator result is read directly without running it again.
    from grassmoth_server.study.queries.data import read_data_history

    selected = read_data_history(
        store, DataRef[GitOid, int](revision=commit_id("TEST", 2), replicate_index=1)
    )
    panel = selected.observations.recorded.frame
    from grassmoth.models.ssm.preparation.binding import project_observation_data

    projected = project_observation_data(
        panel,
        compiled_dynamical_model=compile_model_fixture(dynamical_model_spec),
        time_origin=report.evidence.time_origin,
    )
    assert not isinstance(projected, ObservationPreflightFailure)
    (wide, _) = projected
    np.testing.assert_allclose(
        wide.select(
            [
                dynamical_model_spec.indicator(identity).observation.name
                for identity in evidence.observation_layout.indicator_ids
            ]
        ).to_numpy(),
        evidence.arms.action.observations.values[1],
        equal_nan=True,
    )


@pytest.mark.inference(concern="simulation")
@pytest.mark.parametrize(
    ("start_time", "origin", "complete_test_model_payload"),
    [
        pytest.param(1.0, datetime(2026, 1, 1, tzinfo=UTC), x_y_model, id="1.0"),
        pytest.param(0.0, datetime(2026, 1, 2, tzinfo=UTC), x_y_model, id="0.0"),
        pytest.param(0.25, datetime(2026, 1, 1, 18, tzinfo=UTC), x_y_model, id="0.25"),
    ],
)
def test_retained_forecast_and_timed_intervention_preserve_joint_starts(
    tmp_path, monkeypatch, start_time, origin, complete_test_model_payload
):
    from grassmoth.actions.simulate.action import _simulate as simulate
    from grassmoth.artifacts.scenarios import InterventionSpec
    from grassmoth_server.compute.resources import read_compute_budget
    from grassmoth_server.storage import paths as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    dynamical_model_spec = x_model()
    from grassmoth.models.ssm import compiled_access as numeric

    states = numeric.state_ids(compile_model_fixture(dynamical_model_spec))
    paths = jnp.arange(1.0, 1 + 4 * len(states)).reshape(2, 2, len(states))
    dynamical_model_spec, _ = condition_model(
        dynamical_model_spec,
        compile_model_fixture(dynamical_model_spec),
        particle_posterior(JointPosteriorDraws(parameter_draws(dynamical_model_spec, 2), paths)),
        times=jnp.array([0.0, 1.0]),
        time_origin=origin,
    )
    ArtifactStore("TEST")
    design = SimulationSpec(
        start=date(2026, 1, 2),
        horizon="1d",
        interventions=(InterventionSpec(target=states[0], value=2.0, after="12h"),),
    )
    assert design.start_day(origin) == start_time
    report = simulate(
        StructuralSelection(dynamical_model_spec, None),
        design,
        seed=0,
        num_replicates=100,
        budget=read_compute_budget(),
        fitted=None,
    )
    assert not isinstance(report, ObservationPreflightFailure)
    action = report.arms.action.latent_paths.values
    assert report.arms.kind == "paired"
    reference = report.arms.reference.latent_paths.values
    np.testing.assert_allclose(action[:, 0], reference[:, 0])
    if start_time in (1.0, 0.0):
        for initial in action[:, 0]:
            assert any(np.allclose(initial, atom) for atom in np.asarray(paths[:, int(start_time)]))
    np.testing.assert_allclose(action[:, 1, 0], 2.0, atol=1e-5)
    assert not np.allclose(action[:, -1, 0], 2.0)
    assert np.isfinite(action).all()
    emissions = report.arms.action.observations.values
    assert np.isnan(emissions[:, :-1]).all()
    assert np.isfinite(emissions[:, -1]).all()


@pytest.mark.inference(concern="simulation")
def test_causal_action_uses_common_generator_and_requires_matching_engine_evidence(
    tmp_path, monkeypatch
):
    from grassmoth.actions.simulate.action import _simulate as simulate
    from grassmoth.actions.simulate.effects import summarize_causal_simulation
    from grassmoth.artifacts.identity import GitRef
    from grassmoth.artifacts.scenarios import InterventionSpec
    from grassmoth.models.ssm import compiled_access as numeric
    from grassmoth_server.compute.resources import read_compute_budget
    from grassmoth_server.storage import paths as data_module
    from grassmoth_server_test_support.inference import inference_log

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    dynamical_model_spec = x_y_model()
    states = numeric.state_ids(compile_model_fixture(dynamical_model_spec))
    dynamical_model_spec, _ = condition_model(
        dynamical_model_spec,
        compile_model_fixture(dynamical_model_spec),
        particle_posterior(
            JointPosteriorDraws(parameter_draws(dynamical_model_spec, 2), jnp.zeros((2, 2, 2)))
        ),
        times=jnp.array([0.0, 1.0]),
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
    )
    design = SimulationSpec(
        start=date(2024, 1, 2),
        horizon="1d",
        interventions=(InterventionSpec(target=states[0], value=1.0),),
    )
    inference_log(dynamical_model_spec)
    ArtifactStore("TEST")
    evidence = simulate(
        StructuralSelection(dynamical_model_spec, states[1]),
        design,
        seed=0,
        num_replicates=100,
        budget=read_compute_budget(),
        fitted=None,
    )
    assert not isinstance(evidence, ObservationPreflightFailure)
    from grassmoth.artifacts.reports.predictive_provenance import AuthoredLawProvenance
    from grassmoth.artifacts.simulation import SimulationReport
    from grassmoth_test_support.inference import _report

    generated = SimulationReport(
        summary=empty_simulation_summary(),
        evidence=evidence,
        law=AuthoredLawProvenance(),
        findings=(),
        fit_reliability="converged",
    )
    fitted = (
        GitRef(workspace_id="TEST", revision=git_oid(2), path="model.json"),
        _report(dynamical_model_spec).core,
    )
    assert evidence.arms.kind == "paired"
    arms = evidence.arms

    def summarize():
        return summarize_causal_simulation(
            StructuralSelection(dynamical_model_spec, states[1]),
            design,
            fitted=fitted,
            action=arms.action,
            reference=arms.reference,
            state_ids=evidence.state_ids,
            indicator_ids=evidence.observation_layout.indicator_ids,
        )

    result = summarize()
    from grassmoth.artifacts.reports.checks import NotEvaluated
    from grassmoth.artifacts.scenarios import CausalEffectResult

    assert isinstance(result, CausalEffectResult)
    outcome_index = evidence.state_ids.index(result.outcome)
    reference = arms.reference.latent_paths.values
    expected = arms.action.latent_paths.values[:, :, outcome_index] - reference[:, :, outcome_index]
    np.testing.assert_array_equal(result.differences.values, expected)
    assert result.summary.mean == pytest.approx(float(expected[:, -1].mean()))
    assert result.reference_mean == pytest.approx(float(reference[:, -1, outcome_index].mean()))
    report = generated.revised(evidence=evidence.revised(arms=arms.revised(causal=result)))
    from grassmoth_server.study.persistence.codec import pack_result, unpack_result

    assert SimulationReport.model_validate(unpack_result(pack_result(report))) == report
    from grassmoth.models.ssm.preparation.binding import project_observation_data
    from grassmoth_server.study.queries.data import read_simulation_observations

    panel = read_simulation_observations(evidence, 0)
    projected = project_observation_data(
        panel,
        compiled_dynamical_model=compile_model_fixture(dynamical_model_spec),
        time_origin=evidence.time_origin,
    )
    assert not isinstance(projected, ObservationPreflightFailure)
    (wide, _) = projected
    np.testing.assert_allclose(
        wide.select(
            numeric.observation_names(compile_model_fixture(dynamical_model_spec))
        ).to_numpy(),
        evidence.arms.action.observations.values[0],
        equal_nan=True,
    )
    fitted = None
    unavailable = summarize()
    assert isinstance(unavailable, NotEvaluated)
    assert unavailable.reason == "CAUSAL_EVALUATION_FAILED"


@pytest.mark.contract
def test_data_profile_survives_model_edits(tmp_path, monkeypatch):
    from grassmoth.artifacts.observations.preparation import DataPreparationResult
    from grassmoth_server.storage import paths as data_module
    from grassmoth_server_test_support.action import prepared_profile
    from grassmoth_server_test_support.runner_fixtures import seed_panel

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("TEST")
    panel = seed_panel(store, model_revision=None)
    profile = prepared_profile(
        "TEST",
        StudyState(),
        Applied(result=DataPreparationResult(), effects=ActionEffects(produced=[panel])),
    )
    state = StudyState().with_artifacts((panel,))
    assert set(profile.indicators) == {item.id for item in panel_metadata().variables}
    assert not state.has("model")
    revised_model = stress_sleep_model()
    revised_model = revised_model.with_entities(
        edges=(
            revised_model.edges[0].revised(description="Another scientific goal"),
            *revised_model.edges[1:],
        )
    )
    revised = store.write_artifact(
        "model",
        produced_by=None,
        derived_from={},
        json_files={"model.json": revised_model.model_dump(mode="json")},
    )
    from grassmoth.artifacts.observations.dataset import SelectedObservations
    from grassmoth_server.study.queries.data import read_data_history
    from grassmoth_server_test_support.data import metadata_for_model

    history = read_data_history(
        store, DataRef[GitOid, int](revision=panel.revision, replicate_index=0)
    )
    assert isinstance(
        history.observations.select(metadata_for_model(revised_model).variables),
        SelectedObservations,
    )
    current = state.with_artifacts([revised])
    from grassmoth_server_test_support.action import data_profile

    assert (
        data_profile(
            store,
            DataRef[GitOid, int](revision=current.current["panel"].revision, replicate_index=0),
        )
        == profile
    )


@pytest.mark.inference(concern="predictive")
def test_simulation_retains_exact_histories_without_running_measurement_reducers(monkeypatch):
    from importlib import import_module

    from grassmoth.models.ssm.diagnostics import simulation as simulation_checks
    from grassmoth.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory

    action = import_module("grassmoth.actions.simulate.action")
    simulation = import_module("grassmoth.models.ssm.predictive.simulation")
    dynamical_model_spec = x_model()
    paths = jnp.zeros((2, 3, 1))
    prediction = PredictiveDraws(
        parameters={"future_parameter_site": jnp.array([1.0, 2.0])},
        trajectory=PredictiveTrajectory(
            paths, paths, paths, jnp.ones_like(paths, dtype=bool), paths
        ),
    )
    monkeypatch.setattr(
        simulation,
        "sample_model_laws",
        lambda *_a, **_k: JointPosteriorDraws(prediction.parameters),
    )
    monkeypatch.setattr(simulation, "simulate_predictive_draws", lambda *_a, **_k: prediction)
    called = []

    def no_measurements(*_args, **_kwargs):
        called.append("measurement")
        pytest.fail("Generation must retain evidence before derived reads")

    monkeypatch.setattr(simulation_checks, "measure_construct_dynamics", no_measurements)
    monkeypatch.setattr(simulation_checks, "measure_construct_measurement", no_measurements)
    from grassmoth_server.compute.resources import read_compute_budget

    evidence = action._simulate(
        StructuralSelection(dynamical_model_spec, None),
        SimulationSpec(start=date(2026, 1, 1), horizon="2d"),
        seed=0,
        num_replicates=100,
        budget=read_compute_budget(),
        fitted=None,
    )
    assert not isinstance(evidence, ObservationPreflightFailure)
    assert called == []
    assert evidence.parameter_draws.keys() == prediction.parameters.keys()
    np.testing.assert_array_equal(
        evidence.parameter_draws["future_parameter_site"].values, [1.0, 2.0]
    )
