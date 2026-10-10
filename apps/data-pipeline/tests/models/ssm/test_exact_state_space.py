"""Declared deterministic state semantics survive compilation and posterior sampling."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import LiteralExpression, coefficient, state
from nof1_causal_lab.artifacts.identity import MechanismId
from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec, NormalLawSpec
from nof1_causal_lab.artifacts.mechanism import DriftMechanismSpec
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.runner import (
    _initialize_chain_state,
)
from nof1_causal_lab.models.ssm.inference.problem import build_particle_problem
from tests.helpers import make_model
from tests.inference_fixtures import bind_panel_fixture, compile_fit_fixture

pytestmark = pytest.mark.inference(concern="sampling")


def _model():
    model = make_model(
        ["baseline", "driver", "total", "outcome"],
        [("baseline", "driver"), ("driver", "total"), ("total", "outcome")],
    )
    constructs = tuple(
        c.revised(
            temporal_status="time_invariant" if c.name == "baseline" else "time_varying",
            dynamics=()
            if c.name == "baseline"
            else (
                DriftMechanismSpec(
                    id=MechanismId("mechanism:self_" + c.name),
                    expression=LiteralExpression(value=0.0),
                ),
            ),
            coefficients=(
                coefficient(0.0, role="initial_mean"),
                coefficient(0.0 if c.name == "total" else 1.0, role="initial_scale"),
                coefficient(
                    0.0 if c.name in {"baseline", "total"} else 0.4, role="diffusion_scale"
                ),
            ),
            indicators=tuple(
                i.revised(
                    observation=i.observation.revised(aggregation="last"),
                    likelihood=LikelihoodSpec(
                        law=NormalLawSpec(
                            loc=state(c.id), scale=coefficient(0.3, role="observation_scale")
                        ),
                        reasoning="Known Gaussian instrument",
                        standardized=True,
                    ),
                )
                for i in c.indicators
            ),
        )
        for c in model.constructs
    )
    edges = tuple(
        e.revised(
            mechanisms=(
                DriftMechanismSpec(
                    id=MechanismId("mechanism:" + e.cause.name), expression=state(e.cause.id)
                ),
            )
        )
        for e in replace_constructs(model.edges, constructs)
    )
    return model.with_entities(edges=edges)


def _problem(model, observations=None):
    fit = compile_fit_fixture(model)
    compiled = fit.compiled_dynamical_model
    times = jnp.arange(4.0)
    if observations is None:
        observations = jnp.full((4, len(compiled.observations)), jnp.nan)
    panel = bind_panel_fixture(compiled, observations, times)
    return compiled, build_particle_problem(
        fit.prior_runtime_bundle,
        panel,
        scheme="euler_maruyama",
        trace_key=jax.random.key(2),
        reparam=None,
    )


def _initial(target, path):
    return _initialize_chain_state(
        target.initial_position,
        observations=target.observations,
        times=target.times,
        target=target,
        initial_latent_delta=jnp.full(target.times.size, 0.1),
        param_step_size=0.01,
        param_min_scale=1e-6,
        param_max_scale=1.0,
        param_target_accept=0.35,
        initial_latent_trajectory=path,
        exact_constraints=target.exact_constraints,
    )


def test_static_and_accumulator_target_preserves_support_density_feedback_and_gradients():
    compiled, problem = _problem(_model())
    target = problem.runtime
    indices = {s.name: i for i, s in enumerate(compiled.states)}
    b, x, a, y = (indices[n] for n in ("baseline", "driver", "total", "outcome"))
    assert set(compiled.state_space.initial) == {b, x, y}
    assert set(compiled.state_space.transition) == {x, y}
    context = target.context(target.initial_position, target.times)
    coordinates = jnp.zeros((4, 4)).at[0, b].set(0.2).at[:, x].set(jnp.array([1.0, 2.0, 3.0, 4.0]))
    path = target.replay_path(context, coordinates)
    np.testing.assert_array_equal(path[:, b], jnp.full(4, coordinates[0, b]))
    np.testing.assert_array_equal(path[:, a], [0.0, 1.0, 3.0, 6.0])
    native = target._transition_distribution(context, path[0], jnp.array(1))
    np.testing.assert_array_equal(np.diag(native.covariance_matrix)[[b, a]], [0.0, 0.0])
    draws = native.sample(jax.random.key(3), (8,))
    np.testing.assert_array_equal(draws[:, a], np.ones(8))
    assert jnp.isneginf(native.log_prob(path[1].at[a].add(0.01)))
    expected = dist.Normal(0.0, 1.0).log_prob(path[0, jnp.array([b, x, y])]).sum()
    expected += dist.Normal(path[:-1, x] + 0.2, 0.4).log_prob(path[1:, x]).sum()
    expected += dist.Normal(path[:-1, y] + path[:-1, a], 0.4).log_prob(path[1:, y]).sum()
    np.testing.assert_allclose(
        target.path_log_prob(context, coordinates, target.observations), expected, rtol=2e-6
    )

    def evaluate(coords):
        return target.path_log_prob(context, coords, target.observations)

    gradient = jax.grad(evaluate)(coordinates)
    assert np.isfinite(gradient).all()
    # Driver x0 controls total1 and total2, hence the later outcome transitions.
    perturbed = coordinates.at[0, x].add(0.001)
    numerical = (evaluate(perturbed) - evaluate(coordinates.at[0, x].add(-0.001))) / 0.002
    np.testing.assert_allclose(gradient[0, x], numerical, rtol=0.01)
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_particles=4,
        num_parameter_particles=2,
        param_step_size=0.01,
        parameter_proposal="random_walk",
        dsmc_leaf_proposal="amala_exact",
        latent_block_coords=1,
        exact_constraints=problem.exact_constraints,
    )
    after, info = jax.jit(kernel.step_fn)(_initial(target, coordinates), jax.random.key(4))
    np.testing.assert_array_equal(
        after.latent_trajectory[:, b], np.repeat(after.latent_trajectory[0, b], 4)
    )
    np.testing.assert_array_equal(
        after.latent_trajectory[1:, a], jnp.cumsum(after.latent_trajectory[:-1, x])
    )
    assert np.isfinite(after.complete_log_posterior)
    assert np.any(info["latent_proposed"])


def test_unmeasured_input_is_pinned_at_every_grid_point_and_never_proposed():
    from tests.models.ssm.test_input_trajectories import _model as input_model

    compiled, problem = _problem(input_model())
    target = problem.runtime
    assert problem.exact_constraints is not None
    input_index = next(i for i, s in enumerate(compiled.states) if s.is_input)
    context = target.context(target.initial_position, target.times)
    path = target.initial_path(context, exact_constraints=problem.exact_constraints)
    np.testing.assert_array_equal(path[:, input_index], [1, 3, 3, 3, 2, 2])
    assert not target.independent_mask[:, input_index].any()
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_particles=4,
        num_parameter_particles=2,
        param_step_size=0.01,
        parameter_proposal="random_walk",
        dsmc_leaf_proposal="amala_exact",
        latent_block_coords=1,
        exact_constraints=problem.exact_constraints,
    )
    after, info = jax.jit(kernel.step_fn)(_initial(target, path), jax.random.key(6))
    np.testing.assert_array_equal(after.latent_trajectory[:, input_index], path[:, input_index])
    np.testing.assert_array_equal(
        info["latent_accepted"], jnp.any(after.latent_trajectory != path, axis=-1)
    )


@pytest.mark.inference(concern="recovery")
@pytest.mark.parametrize("parameter_mean", [False, True])
def test_replayed_static_sampler_recovers_one_shared_gaussian_unknown(parameter_mean):
    model = _model()
    baseline = next(c for c in model.constructs if c.name == "baseline")
    baseline = baseline.revised(
        indicators=tuple(
            i.revised(likelihood=i.likelihood.revised(standardized=False))
            for i in baseline.indicators
        )
    )
    readings = jnp.array([[1.0], [1.1], [0.7], [1.2]])
    model = model.with_entities(constructs=(baseline,), edges=())
    if parameter_mean:
        from nof1_causal_lab.artifacts.identity import DistributionId, ParameterId
        from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec

        identity = ParameterId("parameter:" + "3" * 64)
        law = DistributionId("distribution:" + "3" * 64)
        baseline = baseline.revised(
            coefficients=(
                coefficient(identity, role="initial_mean"),
                coefficient(0.0, role="initial_scale"),
            ),
            indicators=tuple(
                i.revised(likelihood=i.likelihood.revised(standardized=True))
                for i in baseline.indicators
            ),
        )
        model = model.with_entities(
            constructs=(baseline,),
            parameters=(
                ParameterSpec(
                    id=identity,
                    name="level",
                    description="Unknown constant level",
                    distribution=law,
                ),
            ),
            distributions={law: dist.Normal(0.0, 1.0)},
        )
    _, problem = _problem(model, readings)
    target = problem.runtime
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_particles=4,
        num_parameter_particles=2,
        param_step_size=0.01,
        parameter_proposal="pseudo_langevin",
        dsmc_leaf_proposal="amala_exact",
        latent_block_coords=1,
    )

    def chain(key):
        initial = _initial(target, jnp.zeros((4, 1)))

        def step(current, draw_key):
            after, _ = kernel.step_fn(current, draw_key)
            return after, after.latent_trajectory[0, 0]

        return jax.lax.scan(step, initial, jax.random.split(key, 700))[1]

    samples = np.asarray(jax.jit(jax.vmap(chain))(jax.random.split(jax.random.key(18), 8)))[:, 200:]
    variance = 1 / (1 + 4 / 0.3**2)
    mean = variance * float(target.observations.sum()) / 0.3**2
    assert np.mean(samples) == pytest.approx(mean, abs=0.025)
    assert np.var(samples) == pytest.approx(variance, abs=0.005)


def test_exact_static_readings_share_one_value_and_inconsistent_constraints_are_rejected():
    from nof1_causal_lab.artifacts.likelihood import DeltaLawSpec
    from nof1_causal_lab.models.ssm.inference.conditioning import compile_exact_state_constraints
    from nof1_causal_lab.models.ssm.preflight import (
        ObservationPreflightFailure,
        validate_observation_support_for_fit,
    )

    model = _model()
    revised = tuple(
        c.revised(
            indicators=tuple(
                i.revised(
                    likelihood=LikelihoodSpec(
                        law=DeltaLawSpec(v=state(c.id)),
                        reasoning="Exact coordinate",
                        standardized=False,
                    )
                )
                for i in c.indicators
            )
        )
        for c in model.constructs
        if c.name in {"baseline", "total"}
    )
    model = model.with_entities(edges=replace_constructs(model.edges, revised))
    compiled = compile_fit_fixture(model).compiled_dynamical_model
    columns = {compiled.states[o.state_index].name: i for i, o in enumerate(compiled.observations)}
    baseline = columns["baseline"]
    total = columns["total"]
    observations = jnp.full((4, 4), jnp.nan).at[0, baseline].set(0.5).at[0, total].set(0.0)
    panel = bind_panel_fixture(compiled, observations, jnp.arange(4.0))
    assert validate_observation_support_for_fit(panel) is None
    constraints = compile_exact_state_constraints(compiled, panel.observations)
    assert constraints is not None
    baseline_state = compiled.observations[baseline].state_index
    np.testing.assert_array_equal(constraints.values[:, baseline_state], jnp.full(4, 0.5))
    for invalid in (
        observations.at[1, baseline].set(0.6),
        observations.at[1, total].set(1.0),
        observations.at[0, total].set(1.0),
    ):
        invalid_panel = bind_panel_fixture(compiled, invalid, jnp.arange(4.0))
        assert isinstance(
            validate_observation_support_for_fit(invalid_panel), ObservationPreflightFailure
        )
