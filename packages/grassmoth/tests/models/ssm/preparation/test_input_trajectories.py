"""Deterministic model inputs share one path across fitting, prediction and conditioning."""

from datetime import UTC, datetime, timedelta

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import polars as pl
import pytest
from pydantic import ValidationError

from grassmoth.actions.fit.conditioning import condition_model
from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.observations.dataset import ObservationDataset
from grassmoth.artifacts.scenarios import StateAssignment
from grassmoth.models.ssm.compile.inputs import IncompleteModel, compile_model
from grassmoth.models.ssm.inference.types import JointPosteriorDraws
from grassmoth.models.ssm.predictive.parameters import sample_model_laws
from grassmoth.models.ssm.predictive.simulation import (
    SimulationBatch,
    generate_simulation_batch,
)
from grassmoth.models.ssm.preparation.binding import (
    BoundPanel,
    PanelPreparationFailure,
    bind_panel,
    input_trajectory_events,
)
from grassmoth.models.ssm.preparation.preflight import ObservationPreflightFailure
from grassmoth.models.structure import StructuralSelection
from grassmoth_test_support.compute import COMPUTE_BUDGET, compile_program
from grassmoth_test_support.inference import (
    compile_model_fixture,
    parameter_draws,
    particle_posterior,
)
from grassmoth_test_support.input_trajectories import deterministic_input_model

_ORIGIN = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.contract
def test_inputs_require_deterministic_laws_and_keep_independent_coordinates():
    dynamical_model_spec = deterministic_input_model()
    compiled_dynamical_model = compile_model_fixture(dynamical_model_spec)
    assert compiled_dynamical_model.states[0].is_input
    assert (
        dynamical_model_spec.time_points == ()
    )  # Input points are not retained endogenous states.
    assert len(compiled_dynamical_model.input_trajectories) == 1
    assert all(not law.layout.constructs for law in compiled_dynamical_model.laws)
    source = dynamical_model_spec.constructs[0]
    identity = source.distribution
    assert identity is not None
    for law in (
        dist.Normal(jnp.zeros(3), 1).to_event(1),
        dist.Delta(jnp.ones(3), log_density=1, event_dim=1),
    ):
        with pytest.raises(ValidationError, match="deterministic Delta"):
            _ = dynamical_model_spec.with_entities(
                distributions={**dynamical_model_spec.distributions, identity: law}
            ).constructs
    with pytest.raises(ValidationError, match="finite values"):
        _ = dynamical_model_spec.with_entities(
            distributions={
                **dynamical_model_spec.distributions,
                identity: dist.Delta(jnp.array([1.0, np.nan, 2.0]), event_dim=1),
            }
        ).constructs
    with pytest.raises(ValidationError, match="must be constant"):
        _ = dynamical_model_spec.with_entities(
            edges=replace_constructs(
                dynamical_model_spec.edges, (source.revised(temporal_status="time_invariant"),)
            )
        ).constructs
    draft = dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (source.with_distribution(None),)),
        distributions={
            key: value
            for key, value in dynamical_model_spec.distributions.items()
            if key != identity
        },
        law_layouts={},
    )
    result = compile_model(StructuralSelection(draft, None))
    assert isinstance(result, IncompleteModel)
    assert "deterministic trajectory law" in result.message


@pytest.mark.inference(concern="sampling")
def test_fit_binds_model_path_at_every_change_and_rejects_contradictory_exact_readings():
    compiled_dynamical_model = compile_model_fixture(
        deterministic_input_model(measured=True, time_origin=_ORIGIN)
    )
    variables = tuple(item.observation for item in compiled_dynamical_model.observations)
    rows = pl.DataFrame(
        [
            {
                "indicator_id": item.id,
                "value": value,
                "support_kind": item.support_kind.value,
                "summary_operator": item.summary_operator.value,
                "anchor_policy": item.anchor_policy.value,
                "observation_window": str(item.observation_window),
                "anchor_time": _ORIGIN + timedelta(days=day),
                "support_start": _ORIGIN
                + timedelta(days=day - (1 if item.support_kind == "interval" else 0)),
                "support_end": _ORIGIN + timedelta(days=day),
            }
            for item, values in zip(variables, ([2.0, 3.0, 2.5], [0.0, 1.0, 2.0]), strict=True)
            for day, value in zip((1, 2, 3), values, strict=True)
        ]
    )
    panel = ObservationDataset.from_frame(rows, variables, time_origin=_ORIGIN)
    bound = bind_panel(
        panel,
        compiled_dynamical_model=compiled_dynamical_model,
        compiler=compile_program,
    )
    assert isinstance(bound, BoundPanel), bound
    np.testing.assert_array_equal(bound.times, [0, 0.5, 1, 2, 2.5, 3])
    np.testing.assert_array_equal(bound.input_values[:, 0], [1, 3, 3, 3, 2, 2])
    assert np.isnan(bound.observations[:, 0]).all()
    contradictory = ObservationDataset.from_frame(
        rows.with_columns(
            pl.when(pl.col("indicator_id") == variables[0].id)
            .then(pl.col("value") + 1)
            .otherwise(pl.col("value"))
            .alias("value")
        ),
        variables,
        time_origin=_ORIGIN,
    )
    failure = bind_panel(
        contradictory,
        compiled_dynamical_model=compiled_dynamical_model,
        compiler=compile_program,
    )
    assert isinstance(failure, PanelPreparationFailure)
    assert "conflict with its deterministic trajectory law" in failure.message
    # Dated input laws align by calendar instant, independently of a panel's origin.
    events = input_trajectory_events(
        compiled_dynamical_model, time_origin=_ORIGIN + timedelta(days=1), start=0, end=4
    )
    assert not isinstance(events, ObservationPreflightFailure)
    assert [(event.spec.time, event.spec.value) for event in events] == [(0.0, 3.0), (1.5, 2.0)]


@pytest.mark.inference(concern="predictive")
@pytest.mark.inference(concern="simulation")
def test_unmeasured_model_inputs_drive_both_simulation_arms_and_survive_conditioning():
    dynamical_model_spec = deterministic_input_model()
    compiled_dynamical_model = compile_model_fixture(dynamical_model_spec)
    source = dynamical_model_spec.constructs[0]
    batch = generate_simulation_batch(
        compiled_dynamical_model,
        start=0,
        end=4,
        draws=2,
        time_origin=_ORIGIN,
        assignments=(StateAssignment(target=source.id, time=1, value=7),),
        compute_budget=COMPUTE_BUDGET,
    )
    assert isinstance(batch, SimulationBatch)
    grid = np.asarray(batch.times)
    expected = np.where(grid < 0.5, 1.0, np.where(grid < 2.5, 3.0, 2.0))
    assert batch.prediction.reference is not None
    np.testing.assert_allclose(
        batch.prediction.reference.latents[:, :, 0], np.tile(expected, (2, 1))
    )
    np.testing.assert_allclose(
        batch.prediction.trajectory.latents[:, :, 0],
        np.tile(np.where(grid < 1, expected, 7.0), (2, 1)),
    )
    paths = jnp.array([[[1.0, 4.0], [3.0, 5.0]], [[1.0, 6.0], [3.0, 7.0]]])
    conditioned, _ = condition_model(
        dynamical_model_spec,
        compiled_dynamical_model,
        particle_posterior(JointPosteriorDraws(parameter_draws(dynamical_model_spec, 2), paths)),
        times=jnp.array([0.0, 1.0]),
        time_origin=_ORIGIN,
    )
    # Metadata JSON retains array references; the complete transport includes the
    # empirical atoms needed to materialize the conditioned law.
    restored = DynamicalModelSpec.model_validate(
        conditioned.model_dump(context={"binary_arrays": True})
    )
    fitted = compile_model_fixture(restored)
    assert restored.time_points == (0.0, 1.0)
    assert fitted.input_trajectories[0].layout.time_origin == _ORIGIN
    assert fitted.input_trajectories[0].values == (1.0, 3.0, 2.0)
    draws = sample_model_laws(fitted, draws=2, key=jax.random.PRNGKey(1))
    assert draws.state_ids == (dynamical_model_spec.constructs[1].id,)
    assert draws.latent_paths is not None
    assert draws.latent_paths.shape == (2, 2, 1)
    forecast = generate_simulation_batch(
        fitted, start=2, end=4, draws=2, time_origin=_ORIGIN, compute_budget=COMPUTE_BUDGET
    )
    assert isinstance(forecast, SimulationBatch)
    assert np.isfinite(forecast.prediction.trajectory.latents).all()
    np.testing.assert_array_equal(forecast.prediction.trajectory.latents[:, -1, 0], [2.0, 2.0])
