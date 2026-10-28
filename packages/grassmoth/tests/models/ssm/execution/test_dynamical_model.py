"""Native model execution across inference, warmup, simulation, and prediction."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import TYPE_CHECKING

import dynestyx as dsx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.artifacts.model.likelihood import LinkFunction
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.models.ssm.dynamics.intervention import Intervention
from grassmoth.models.ssm.inference import problem as problem_module
from grassmoth.models.ssm.inference.parameter_transform import ParameterTransform
from grassmoth_test_support.compute import compile_program
from grassmoth_test_support.dynamics import potential_term
from grassmoth_test_support.inference import bind_panel_fixture, compile_fit_fixture
from grassmoth_test_support.model import load_model_fixture


def _runtime_model_fixture() -> DynamicalModelSpec:
    return load_model_fixture("dynamical_model/runtime_model_fixture.json")


if TYPE_CHECKING:
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec


@pytest.fixture
def runtime(monkeypatch):
    # Only parameter discovery is stubbed. The declared drift, observation model,
    # model partition, discretizer, and particle target are the actual code paths.
    times = jnp.array([2.0, 2.1, 2.35])
    from grassmoth.models.ssm.execution.parameters import assemble_model_matrices

    dynamical_model_spec = _runtime_model_fixture()
    inputs = compile_fit_fixture(dynamical_model_spec)

    def constrain(z):
        samples = {"vf_0_p0": z[0]}
        matrices, _ = assemble_model_matrices(inputs.compiled_dynamical_model, samples)
        return {**samples, **matrices}

    parameters = ParameterTransform(
        initial_position=jnp.array([0.2]),
        unravel=lambda z: {"vf_0_p0": z[0]},
        constrain=constrain,
        log_prior=lambda z: -jnp.sum(z**2),
    )
    monkeypatch.setattr(
        problem_module,
        "prepare_model_parameters",
        lambda *_, **_kwargs: (parameters, {}, {"vf_0_p0"}),
    )
    return problem_module.build_particle_problem(
        inputs.prior_runtime_bundle,
        bind_panel_fixture(
            inputs.compiled_dynamical_model,
            jnp.array([[0.2, 1.0], [jnp.nan, 2.0], [jnp.nan, jnp.nan]]),
            times,
        ),
        trace_key=jax.random.key(0),
        reparam=None,
        compiler=compile_program,
    ).runtime


@pytest.mark.inference(concern="sampling")
def test_sampler_context_is_a_dynestyx_model_pytree(runtime):
    context = runtime.context(runtime.initial_position, runtime.times)
    assert isinstance(context[0], dsx.DynamicalModel)
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(context))
    shapes = jax.eval_shape(
        jax.vmap(runtime.context, in_axes=(0, None)), jnp.zeros((2, 1)), runtime.times
    )
    assert shapes[0].initial_condition.loc.shape == (2, 1)
    assert shapes[1].shape == (2, len(runtime.times))


@pytest.mark.inference(concern="simulation")
def test_model_keeps_nonlinear_drift(runtime):
    context = runtime.context(runtime.initial_position, runtime.times)
    declared = runtime.dynamical_model(context)
    state = jnp.array([0.8])
    # Inspect a single distribution, without invoking a solver or sampler.
    with jax.disable_jit():
        law = declared.state_evolution(state, None, runtime.times[0], runtime.times[1])
    displacement = np.asarray(state) - 0.2
    dt = float(runtime.times[1] - runtime.times[0])
    expected_drift = -0.4 * displacement - 0.2 * displacement**3
    np.testing.assert_allclose(law.mean, state + dt * expected_drift, atol=1e-6)
    np.testing.assert_allclose(law.covariance_matrix, [[dt * 0.3 + 1e-8]], atol=1e-7)
    np.testing.assert_allclose(declared.initial_condition.mean, [0.7])
    np.testing.assert_allclose(declared.initial_condition.covariance_matrix, [[0.8]], atol=2e-6)


@pytest.mark.inference(concern="sampling")
def test_exact_target_and_parameter_gradient_trace_through_the_model(runtime):
    path = jnp.linspace(0.1, -0.2, runtime.times.size)[:, None]

    def evaluate(position):
        context = runtime.context(position, runtime.times)
        return runtime.path_log_prob(context, path, runtime.observations)

    density, gradient = jax.eval_shape(jax.value_and_grad(evaluate), runtime.initial_position)
    assert density.shape == ()
    assert gradient.shape == (1,)
    factors = jax.eval_shape(
        lambda position: runtime.observation_log_probs(
            runtime.context(position, runtime.times), path, runtime.observations
        ),
        runtime.initial_position,
    )
    assert factors.shape == (runtime.times.size,)


@pytest.mark.inference(concern="sampling")
def test_parameter_gradient_remains_dynamic_after_model_partition(runtime):
    previous, current = jnp.array([0.8]), jnp.array([0.9])

    def transition(position):
        context = runtime.context(position, runtime.times)
        return runtime.transition_log_prob(context, previous, current, jnp.array(1))

    # One scalar derivative with JIT disabled; no sampler or trajectory solve.
    with jax.disable_jit():
        gradient = jax.grad(transition)(runtime.initial_position)
    dt = float(runtime.times[1] - runtime.times[0])
    displacement = 0.8 - 0.2
    mean = 0.8 + dt * (-0.4 * displacement - 0.2 * displacement**3)
    mean_gradient = dt * (0.4 + 0.6 * displacement**2)
    expected = (0.9 - mean) * mean_gradient / (dt * 0.3 + 1e-8)
    np.testing.assert_allclose(gradient, [expected], rtol=1e-5, atol=1e-6)


@pytest.mark.inference(concern="sampling")
def test_observation_model_keeps_partial_and_complete_missingness(runtime):
    context = runtime.context(runtime.initial_position, runtime.times)
    observation = runtime.dynamical_model(context).observation_model(
        jnp.array([0.0]), None, runtime.times[0]
    )
    with jax.disable_jit():
        partial = observation.log_prob(jnp.array([jnp.nan, 2.0]))
        absent = observation.log_prob(jnp.array([jnp.nan, jnp.nan]))
    assert float(partial) == pytest.approx(-1.0 - np.log(2.0), abs=1e-6)
    assert float(absent) == 0.0


@pytest.mark.inference(concern="sampling")
def test_selected_observation_channels_preserve_marginal_density_and_gradients():
    from grassmoth.models.ssm.execution.contracts import MeasurementParams
    from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from grassmoth_test_support.observation import observation_laws

    covariance = jnp.array(
        [[1.2, 0.2, 0.3, 0.0], [0.2, 0.9, -0.1, 0.0], [0.3, -0.1, 1.4, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    observation = HeterogeneousObservation(
        MeasurementParams(jnp.eye(4), jnp.arange(4.0) / 10, covariance),
        observation_laws((DistributionFamily.GAUSSIAN,) * 3 + (DistributionFamily.POISSON,)),
    )
    channels = (2, 0, 3)
    selected = observation.select_channels(channels)
    values = jnp.array([[0.3, jnp.nan, 1.2, 3.0], [jnp.nan, jnp.nan, 0.5, jnp.nan]])
    state = jnp.array([0.2, -0.1, 1.4, 0.7])

    def score(model, readings, x):
        return jnp.sum(jax.vmap(lambda row: model(x, None, 0).log_prob(row))(readings))

    expected = jax.jit(jax.value_and_grad(lambda x: score(observation, values, x)))(state)
    actual = jax.jit(
        jax.value_and_grad(lambda x: score(selected, values[:, jnp.asarray(channels)], x))
    )(state)
    for measured, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(measured, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.inference(concern="sampling")
@pytest.mark.inference(concern="predictive")
def test_native_discrete_law_samples_categories_from_predictors():
    from grassmoth.models.ssm.execution.contracts import MeasurementParams
    from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from grassmoth_test_support.observation import observation_laws

    observation = HeterogeneousObservation(
        MeasurementParams(jnp.eye(2), jnp.zeros(2), jnp.eye(2)),
        observation_laws(
            (DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL),
            (LinkFunction.CUMULATIVE_LOGIT, LinkFunction.SOFTMAX),
            {
                "obs_ordered_cutpoints": jnp.array([[-1.0, 1.0], [-1.0, 1.0]]),
                "obs_cat_intercepts": jnp.zeros((2, 2)),
                "obs_cat_slopes": jnp.array([[0.0, 0.0], [-1.0, 1.0]]),
            },
            level_counts=(3, 3),
        ),
    )
    # At this predictor both declared laws concentrate on their final category.
    # Reconstructing a law from the response mean would lose that information.
    law = observation(jnp.array([100.0, 100.0]), None, jnp.array(0.0))
    for key in (jax.random.key(2), jax.random.PRNGKey(2)):
        draws = law.sample(key, sample_shape=(2, 3))
        np.testing.assert_array_equal(draws, np.full((2, 3, 2), 2))
    np.testing.assert_allclose(law.mean, [2.0, 2.0], atol=1e-6)
    assert np.isfinite(law.log_prob(jnp.array([2.0, 2.0])))


@pytest.mark.inference(concern="simulation")
def test_forward_simulation_passes_the_declared_model_to_the_ode_solver(monkeypatch):
    simulator = import_module("grassmoth.models.ssm.dynamics.simulator")
    from grassmoth.models.ssm.dynamics.spec import DynamicsSpec, compile_dynamics

    field = compile_dynamics(
        DynamicsSpec(
            1,
            (
                potential_term(
                    0,
                    center=0,
                    stiffness=0.4,
                    quartic=0.2,
                ),
            ),
        )
    ).vector_field
    params = ({},)
    grid = jnp.array([2.0, 2.2, 2.5])
    initial = jnp.array([0.8])
    paths = jnp.zeros((3, 1))

    def solve(
        model,
        *,
        initial_state,
        t0,
        path_times,
        diffeqsolve_settings,
        ctrl_times=None,
        ctrl_values=None,
    ):
        assert isinstance(model, dsx.DynamicalModel)
        assert ctrl_times is None
        assert ctrl_values is None
        np.testing.assert_allclose(
            model.state_evolution.total_drift(x=initial, u=None, t=t0),
            -0.4 * initial - 0.2 * initial**3,
        )
        np.testing.assert_array_equal(initial_state, initial)
        np.testing.assert_array_equal(path_times, grid)
        assert diffeqsolve_settings["throw"] is False
        assert diffeqsolve_settings["max_steps"] == 4096
        assert float(diffeqsolve_settings["dt0"]) == pytest.approx(0.5 / 256)
        return paths

    monkeypatch.setattr(simulator, "solve_ode_state_path", solve)
    actual = simulator.simulate(field, params, Intervention.none(), initial, grid)
    assert actual is paths


@pytest.mark.inference(concern="simulation")
def test_indexed_sde_keeps_its_brownian_path_and_uses_dynestyx_evolution(monkeypatch):
    simulator = import_module("grassmoth.models.ssm.dynamics.simulator")
    from grassmoth.models.ssm.dynamics.spec import DynamicsSpec, compile_dynamics

    field = compile_dynamics(
        DynamicsSpec(
            1,
            (
                potential_term(
                    0,
                    center=0,
                    stiffness=0.4,
                    quartic=0.2,
                ),
            ),
        )
    ).vector_field
    params = ({},)
    grid = jnp.array([2.0, 2.2, 2.5])
    initial = jnp.array([0.8])
    covariance = jnp.array([[0.3]])
    key = jax.random.key(5)
    paths = jnp.zeros((3, 1))

    def solve(terms, _solver, **settings):
        evolution = settings["args"]
        assert isinstance(evolution, dsx.StochasticContinuousTimeStateEvolution)
        brownian = terms.terms[1].control
        assert isinstance(brownian, simulator._IndexedBrownianPath)
        np.testing.assert_array_equal(jax.random.key_data(brownian.key), jax.random.key_data(key))
        np.testing.assert_allclose(
            evolution.diffusion.gram_matrix(x=initial, u=None, t=grid[0], state_dim=1),
            covariance,
            atol=1e-7,
        )
        base_drift = -0.4 * initial - 0.2 * initial**3
        for time in grid:
            np.testing.assert_allclose(
                terms.terms[0].vf(time, initial, settings["args"]), base_drift
            )
        assert settings["dt0"] is None
        controller = settings["stepsize_controller"]
        assert isinstance(controller, simulator._PreparedStepController)
        assert controller.steps is brownian.steps
        np.testing.assert_array_equal(controller.steps.times, grid)
        assert float(controller.steps.step_size) == pytest.approx(0.01)
        assert isinstance(settings["adjoint"], simulator.dfx.ForwardMode)
        return SimpleNamespace(ys=paths)

    monkeypatch.setattr(simulator.dfx, "diffeqsolve", solve)
    actual = simulator.simulate(
        field,
        params,
        Intervention.none(),
        initial,
        grid,
        config=simulator.SimulationConfig(brownian=simulator.IndexedBrownianSpec(step_size=0.01)),
        noise=simulator.ProcessNoise(key=key, diffusion=jnp.linalg.cholesky(covariance)),
    )
    assert actual is paths
