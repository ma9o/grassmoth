"""Dated state assignments and paired nonlinear histories."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest

from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.expressions import restoring_potential, state
from grassmoth.artifacts.model.likelihood import DeltaLawSpec
from grassmoth.artifacts.model.mechanism import PotentialMechanismSpec
from grassmoth.artifacts.model.parameter import SiteKind
from grassmoth.artifacts.scenarios import StateAssignment
from grassmoth.models.ssm.counterfactual.orchestration import (
    ResolvedIntervention,
    build_segment_bounds,
    vmap_simulate_interventions_from_state,
)
from grassmoth.models.ssm.dynamics.draws import DynamicsDraws
from grassmoth.models.ssm.dynamics.simulator import ProcessNoise
from grassmoth.models.ssm.preparation.preflight import ObservationPreflightFailure
from grassmoth_test_support.dynamics import DenseLinear, reference_vector_field
from grassmoth_test_support.inference import compile_model_fixture
from grassmoth_test_support.model import (
    _exact_model_model,
    construct_named,
    one_state_gaussian_model,
    parameter_for,
    without_parameters,
)


def _fully_fixed_dynamics_keep_the_explicit_draw_axis_model_fixture() -> DynamicalModelSpec:
    dynamical_model_spec = one_state_gaussian_model()
    latent_0 = construct_named(dynamical_model_spec, "latent_0")
    (latent_0_drift,) = latent_0.dynamics
    latent_0_dynamics_decay = parameter_for(
        dynamical_model_spec, SiteKind.DYNAMICS_DECAY, "latent_0"
    )
    latent_0_revised = latent_0.revised(
        dynamics=(
            PotentialMechanismSpec(
                id=latent_0_drift.id,
                expression=restoring_potential(latent_0.id, center=1.0, stiffness=0.4, quartic=0.2),
            ),
        )
    )
    parameters, distributions = without_parameters(dynamical_model_spec, latent_0_dynamics_decay)
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (latent_0_revised,)),
        parameters=parameters,
        distributions=distributions,
    )


if TYPE_CHECKING:
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth.models.ssm.dynamics.vector_field import VectorField


# var1 is driven by var0; both stable. Baseline steady state is η* = -A⁻¹c = [1, 1].
_PARAMS = ({"drift": jnp.array([[-1.0, 0.0], [0.5, -1.0]]), "cint": jnp.array([1.0, 0.5])},)
_STEADY_STATE = jnp.array([[1.0, 1.0]])
_TIME_GRID = jnp.linspace(0.0, 12.0, 13)  # daily grid, day == index


def _event(index, **payload):
    spec = StateAssignment.model_validate({"target": f"construct:state{index}", **payload})
    return ResolvedIntervention(index=index, spec=spec)


def _vf() -> VectorField:
    return reference_vector_field(n_latent=2, components=(DenseLinear(),))


def _run(interventions, initial_states=_STEADY_STATE):
    baseline, action, effect = vmap_simulate_interventions_from_state(
        DynamicsDraws(_vf(), jax.tree.map(lambda x: x[None], _PARAMS), n_draws=1),
        initial_states=initial_states,
        interventions=interventions,
        time_grid=_TIME_GRID,
    )
    return baseline[0], action[0], effect[0]  # single draw → (T, n_latent)


@pytest.mark.contract
def test_segment_bounds_split_at_exact_event_times():
    events = [_event(index=0, value=0.5, time=14.0), _event(index=1, value=2.0, time=20.0)]
    grid = jnp.arange(31.0)
    assert build_segment_bounds(grid, events) == [(0, 14), (14, 20), (20, 30)]
    assert build_segment_bounds(grid, []) == [(0, 30)]


@pytest.mark.contract
def test_model_input_paths_hold_and_interventions_override_later_points(monkeypatch):
    from datetime import UTC, datetime, timedelta, timezone

    import numpy as np
    import numpyro.distributions as dist

    from grassmoth.artifacts.model.construct import replace_constructs
    from grassmoth.artifacts.model.likelihood import LikelihoodSpec
    from grassmoth.artifacts.probability.joint_layout import JointLawLayout
    from grassmoth.models.ssm.counterfactual import orchestration
    from grassmoth.models.ssm.inference.conditioning import compile_exact_state_constraints
    from grassmoth.models.ssm.preparation.binding import (
        input_trajectory_events,
        input_trajectory_values,
    )

    dynamical_model_spec = _exact_model_model()
    dose = dynamical_model_spec.constructs[0]
    indicator = dose.indicators[0].revised(
        observation=dose.indicators[0].observation.revised(aggregation="sum"),
        likelihood=LikelihoodSpec(
            law=DeltaLawSpec(v=state(dose.id)),
            reasoning="Given dose total",
        ),
    )
    layout = JointLawLayout(parameters=(), constructs=(dose.id,), time_points=(-2, 2, 4), labels={})
    dose = dose.revised(
        role="exogenous",
        indicators=(indicator,),
        dynamics=(),
        coefficients=(),
        distribution=layout.distribution_id,
    )
    parameters = tuple(
        parameter
        for parameter in dynamical_model_spec.parameters
        if parameter.name not in {"rho_setting", "sigma_setting"}
    )
    dynamical_model_spec = dynamical_model_spec.revised(measurement_clock="1d").with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (dose,)),
        parameters=parameters,
        distributions={
            **{
                identity: law
                for identity, law in dynamical_model_spec.distributions.items()
                if identity in {parameter.distribution for parameter in parameters}
            },
            layout.distribution_id: dist.Delta(jnp.array([10.0, 8.0, 12.0]), event_dim=1),
        },
        law_layouts={layout.distribution_id: layout},
    )
    from grassmoth_test_support.inference import compile_model_fixture

    compiled_dynamical_model = compile_model_fixture(dynamical_model_spec)
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    grid = jnp.arange(8.0)
    events = input_trajectory_events(
        compiled_dynamical_model,
        time_origin=origin.astimezone(timezone(timedelta(hours=2))),
        start=0,
        end=7,
    )
    assert not isinstance(events, ObservationPreflightFailure)
    values = input_trajectory_values(compiled_dynamical_model, grid, events)
    np.testing.assert_array_equal(values[:, 0], [10, 10, 8, 8, 12, 12, 12, 12])
    constraints = compile_exact_state_constraints(
        compiled_dynamical_model, jnp.full((8, 2), jnp.nan), input_values=values
    )
    assert constraints is not None
    assert not constraints.free_mask[:, 0].any()
    failure = input_trajectory_events(compiled_dynamical_model, time_origin=origin, start=-3, end=7)
    assert isinstance(failure, ObservationPreflightFailure)
    assert "no value at the start" in failure.message
    # Isolate dated assignment control flow; no scientific solver runs in this contract.
    monkeypatch.setattr(
        orchestration,
        "simulate",
        lambda _field, _params, _intervention, initial, times, _config, **_kwargs: jnp.broadcast_to(
            initial, (len(times), len(initial))
        ),
    )
    reference, action, _ = vmap_simulate_interventions_from_state(
        DynamicsDraws(_vf(), jax.tree.map(lambda x: x[None], _PARAMS), n_draws=1),
        jnp.zeros((1, 2)),
        [_event(0, time=1.0, value=3.0), _event(0, time=5.0, value=6.0)],
        time_grid=grid,
        input_events=events,
    )
    np.testing.assert_array_equal(reference[0, :, 0], values[:, 0])
    np.testing.assert_array_equal(action[0, :, 0], [10, 3, 3, 3, 3, 6, 6, 6])


@pytest.mark.inference(concern="simulation")
def test_repeated_and_simultaneous_assignments_resume_natural_dynamics():
    baseline, action, effect = _run(
        [
            _event(index=0, time=2.0, value=3.0),
            _event(index=1, time=2.0, value=5.0),
            _event(index=0, time=4.0, value=-1.0),
            _event(index=1, time=12.0, value=7.0),
        ]
    )
    assert jnp.allclose(action[:2], baseline[:2])
    assert jnp.array_equal(action[2], jnp.array([3.0, 5.0]))
    assert jnp.allclose(action[2:4, 0], 1 + 2 * jnp.exp(-(_TIME_GRID[2:4] - 2)), atol=0.01)
    assert jnp.allclose(action[4:, 0], 1 - 2 * jnp.exp(-(_TIME_GRID[4:] - 4)), atol=0.01)
    assert action[-1, 1] == 7.0
    assert jnp.allclose(effect, action - baseline)


@pytest.mark.inference(concern="simulation")
def test_assignment_to_a_non_varying_state_persists_and_propagates():
    # X has no natural change; Y relaxes toward X/2 + 0.5.
    params = ({"drift": jnp.array([[0.0, 0.0], [0.5, -1.0]]), "cint": jnp.array([0.0, 0.5])},)
    baseline, action, _ = vmap_simulate_interventions_from_state(
        DynamicsDraws(_vf(), jax.tree.map(lambda x: x[None], params), n_draws=1),
        _STEADY_STATE,
        [_event(index=0, time=2.0, value=3.0)],
        time_grid=_TIME_GRID,
    )
    assert jnp.allclose(action[:, :2], baseline[:, :2])
    assert jnp.allclose(action[0, 2:, 0], 3.0)
    assert jnp.allclose(action[0, 2:, 1], 2 - jnp.exp(-(_TIME_GRID[2:] - 2)), atol=0.01)


@pytest.mark.inference(concern="simulation")
def test_explicit_start_evolves_from_given_state():
    initial_states = jnp.array([[0.0, 0.0]])
    baseline, action, _ = _run(
        [_event(index=0, value=2.0, time=0.0)], initial_states=initial_states
    )
    assert jnp.array_equal(baseline[0], initial_states[0])
    assert jnp.array_equal(action[0], jnp.array([2.0, 0.0]))
    assert jnp.allclose(baseline[-1], jnp.array([1.0, 1.0]), atol=0.1)


@pytest.mark.inference(concern="simulation")
def test_fully_fixed_dynamics_keep_the_explicit_draw_axis():
    from grassmoth.models.ssm.dynamics.draws import dynamics_from_samples

    dynamical_model_spec = _fully_fixed_dynamics_keep_the_explicit_draw_axis_model_fixture()
    draws = dynamics_from_samples(compile_model_fixture(dynamical_model_spec), {}, n_draws=3)
    times = jnp.array([0.0, 0.2, 0.4])
    initial = jnp.array([[-1.0], [0.0], [1.0]])
    baseline, action, effect = vmap_simulate_interventions_from_state(
        draws,
        initial,
        [],
        time_grid=times,
    )
    assert baseline.shape == (3, 3, 1)
    assert jnp.isfinite(baseline).all()
    assert jnp.array_equal(baseline, action)
    assert jnp.array_equal(effect, jnp.zeros_like(effect))
    assert jnp.array_equal(baseline[:, 0], initial)
    assert baseline[0, -1, 0] > baseline[0, 0, 0]


@pytest.mark.contract
def test_zero_dynamics_draws_return_empty_trajectories():
    empty = DynamicsDraws(reference_vector_field(n_latent=2, components=()), (), n_draws=0)
    outputs = vmap_simulate_interventions_from_state(
        empty, jnp.zeros((0, 2)), [], time_grid=_TIME_GRID
    )
    assert all(output.shape == (0, _TIME_GRID.size, 2) for output in outputs)


@pytest.mark.contract
@pytest.mark.parametrize("bad_axis", ["state", "key", "covariance"])
def test_event_simulation_rejects_misaligned_draw_inputs(bad_axis):
    draws = DynamicsDraws(reference_vector_field(n_latent=2, components=()), (), n_draws=3)
    initial = jnp.zeros((2 if bad_axis == "state" else 3, 2))
    keys = jax.random.split(jax.random.PRNGKey(0), 2 if bad_axis == "key" else 3)
    covariances = jnp.zeros((3, 1, 1)) if bad_axis == "covariance" else jnp.zeros((3, 2, 2))
    with pytest.raises(ValueError, match="must match the dynamics"):
        vmap_simulate_interventions_from_state(
            draws,
            initial,
            [],
            time_grid=_TIME_GRID,
            noise=ProcessNoise(key=keys, diffusion=jnp.linalg.cholesky(covariances)),
        )


@pytest.mark.inference(concern="simulation")
def test_stochastic_paths_share_noise_before_a_noninteger_event():
    grid = jnp.array([4.0, 4.1, 4.3, 4.7])
    intervention = _event(index=0, time=4.3, value=3.0)
    baseline, action, _ = vmap_simulate_interventions_from_state(
        DynamicsDraws(_vf(), jax.tree.map(lambda x: x[None], _PARAMS), n_draws=1),
        _STEADY_STATE,
        [intervention],
        time_grid=grid,
        noise=ProcessNoise(
            key=jax.random.split(jax.random.key(0), 1),
            diffusion=jnp.linalg.cholesky(jnp.eye(2)[None] * 0.1),
        ),
    )
    assert jnp.array_equal(baseline[:, :2], action[:, :2])
    assert action[0, 2, 0] == 3.0
    assert jnp.isfinite(action).all()
