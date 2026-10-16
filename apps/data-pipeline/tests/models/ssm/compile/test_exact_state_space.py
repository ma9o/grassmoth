"""Declared deterministic state semantics survive compilation and posterior sampling."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.artifacts.identity import ConstructId, EdgeId, MechanismId
from nof1_causal_lab.artifacts.model.construct import (
    CausalEdgeSpec,
    ConstructSpec,
    replace_constructs,
)
from nof1_causal_lab.artifacts.model.expressions import LiteralExpression, coefficient, state
from nof1_causal_lab.artifacts.model.likelihood import DeltaLawSpec, LikelihoodSpec, NormalLawSpec
from nof1_causal_lab.artifacts.model.mechanism import DriftMechanismSpec, PotentialMechanismSpec
from nof1_causal_lab.artifacts.sampling import DsmcSpec, SamplerSpec
from nof1_causal_lab.models.ssm.compile.observation_partition import compile_observation_partition
from nof1_causal_lab.models.ssm.compile.state_space import (
    ParallelRecurrences,
    SequentialRecurrences,
)
from nof1_causal_lab.models.ssm.execution.supported_normal import SupportedNormal
from nof1_causal_lab.models.ssm.inference.problem import build_particle_problem
from nof1_causal_lab.models.ssm.inference.samplers.mpgibbs.fit import (
    fit_marginal_particle_gibbs,
)
from nof1_causal_lab.models.ssm.inference.samplers.mpgibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from nof1_causal_lab.models.ssm.inference.samplers.mpgibbs.runner import (
    _initialize_chain_state,
)
from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure
from tests.support.compute import COMPUTE_BUDGET
from tests.support.helpers import make_model
from tests.support.inference import bind_panel_fixture, compile_fit_fixture

pytestmark = pytest.mark.inference(concern="sampling")


def test_supported_density_and_gradients_preserve_correlated_nonleading_coordinates():
    target = _problem(_model())[1].runtime
    active = jnp.array([1, 3])
    mean = jnp.array([0.0, 0.2, 0.0, -0.4])

    def factor(parameters):
        return jnp.array([[jnp.exp(parameters[0]), 0.0], [parameters[1], jnp.exp(parameters[2])]])

    def score(parameters, values):
        chol = factor(parameters)
        covariance = jnp.zeros((4, 4)).at[jnp.ix_(active, active)].set(chol @ chol.T)
        law = SupportedNormal(mean, covariance, (1, 3))
        return target._modeled_log_prob(law, values, (1, 3))

    def analytic(parameters, values):
        residual = jnp.linalg.solve(factor(parameters), values[active] - mean[active])
        return -jnp.log(2 * jnp.pi) - parameters[0] - parameters[2] - 0.5 * residual @ residual

    parameters = jnp.array([-0.7, 0.3, 0.2])
    values = jnp.array([0.0, 0.8, 0.0, -0.1])
    actual = jax.jit(jax.value_and_grad(score, argnums=(0, 1)))(parameters, values)
    expected = jax.value_and_grad(analytic, argnums=(0, 1))(parameters, values)
    for measured, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(measured, reference, rtol=2e-6, atol=2e-6)


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


def _initial(target, path, *, kernel, position=None):
    return _initialize_chain_state(
        target.initial_position if position is None else position,
        observations=target.observations,
        times=target.times,
        target=target,
        parameter_gradient_fn=kernel.parameter_gradient_fn,
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
    position = target.initial_position.at[-1].set(0.2)
    context = target.context(position, target.times)
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
        exact_constraints=problem.exact_constraints,
        dsmc=DsmcSpec(drift="amala_plus", latent_block_coords=1),
    )
    after, info = jax.jit(kernel.step_fn)(
        _initial(target, coordinates, kernel=kernel), jax.random.key(4)
    )
    np.testing.assert_array_equal(
        after.latent_trajectory[:, b], np.repeat(after.latent_trajectory[0, b], 4)
    )
    np.testing.assert_array_equal(
        after.latent_trajectory[1:, a], jnp.cumsum(after.latent_trajectory[:-1, x])
    )
    assert np.isfinite(after.complete_log_posterior)
    assert np.any(info["latent_proposed"])
    expected_complete, expected_path = target.log_posterior_from_context(
        after.position, after.latent_context, after.latent_trajectory, target.observations
    )
    np.testing.assert_allclose(after.trajectory_log_prob, expected_path, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(
        after.complete_log_posterior, expected_complete, rtol=2e-6, atol=2e-5
    )


def test_unmeasured_input_is_pinned_at_every_grid_point_and_never_proposed():
    from tests.models.ssm.preparation.test_input_trajectories import _model as input_model

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
        exact_constraints=problem.exact_constraints,
        dsmc=DsmcSpec(drift="amala_plus", latent_block_coords=1),
    )
    after, info = jax.jit(kernel.step_fn)(_initial(target, path, kernel=kernel), jax.random.key(6))
    np.testing.assert_array_equal(after.latent_trajectory[:, input_index], path[:, input_index])
    np.testing.assert_array_equal(
        info["latent_accepted"], jnp.any(after.latent_trajectory != path, axis=-1)
    )


@pytest.mark.inference(concern="recovery")
@pytest.mark.parametrize("parameter_mean", [False, True])
def test_parallel_static_sampler_recovers_one_shared_gaussian_unknown(parameter_mean):
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
        from nof1_causal_lab.artifacts.model.parameter_spec import ParameterSpec

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
        dsmc=DsmcSpec(drift="amala_plus", latent_block_coords=1),
    )

    def chain(key):
        initial = _initial(target, jnp.zeros((4, 1)), kernel=kernel)

        def step(current, draw_key):
            after, _ = kernel.step_fn(current, draw_key)
            return after, after.latent_trajectory[0, 0]

        return jax.lax.scan(step, initial, jax.random.split(key, 700))[1]

    samples = np.asarray(jax.jit(jax.vmap(chain))(jax.random.split(jax.random.key(18), 8)))[:, 200:]
    variance = 1 / (1 + 4 / 0.3**2)
    mean = variance * float(target.observations.sum()) / 0.3**2
    assert np.mean(samples) == pytest.approx(mean, abs=0.025)
    assert np.var(samples) == pytest.approx(variance, abs=0.005)


def test_static_promotion_keeps_correlated_initial_density_and_shared_coordinate():
    model = _model()
    baseline = next(c for c in model.constructs if c.name == "baseline")
    driver = next(c for c in model.constructs if c.name == "driver")
    driver = driver.revised(
        coefficients=(
            *driver.coefficients,
            coefficient(0.4, "initial_correlation", construct_ids=(baseline.id,)),
        )
    )
    common = ConstructSpec(
        id=ConstructId("construct:initial_common_cause"),
        name="initial_common_cause",
        description="Marginalized root of the correlated initial law",
        role="endogenous",
        temporal_status="time_invariant",
    )
    edges = replace_constructs(model.edges, (driver,)) + tuple(
        CausalEdgeSpec(
            id=EdgeId("edge:initial_common_cause_" + child.name),
            cause=common,
            effect=child,
            description="Shared initial cause",
        )
        for child in (baseline, driver)
    )
    compiled, problem = _problem(model.with_entities(edges=edges))
    target = problem.runtime
    indices = {item.name: i for i, item in enumerate(compiled.states)}
    b, x, y = (indices[name] for name in ("baseline", "driver", "outcome"))
    assert target.initial_position.size == 1
    assert not np.any(target.independent_mask[:, b])
    np.testing.assert_array_equal(target.state_density_mask[:, b], [True, False, False, False])
    position = jnp.array([0.7])
    context = target.context(position, target.times)
    coordinates = jnp.zeros((4, 4)).at[:, x].set(-0.2).at[:, y].set(0.3)
    path = target.replay_path(context, coordinates)
    np.testing.assert_array_equal(path[:, b], jnp.full(4, position[0]))
    expected = dist.MultivariateNormal(
        jnp.zeros(3),
        covariance_matrix=jnp.array([[1.0, 0.4, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ).log_prob(jnp.array([0.7, -0.2, 0.3]))
    np.testing.assert_allclose(
        target.initial_log_prob(context, coordinates[0]), expected, atol=2e-6
    )


def test_parallel_recurrence_proof_rejects_nonlinearity_only_in_deterministic_subsystem():
    model = _model()
    assert isinstance(_problem(model)[0].state_space.recurrence, ParallelRecurrences)
    driver = next(c for c in model.constructs if c.name == "driver")
    driver = driver.revised(
        dynamics=(
            DriftMechanismSpec(
                id=MechanismId("mechanism:nonlinear_driver"), expression=-(state(driver.id) ** 3)
            ),
        )
    )
    model = model.with_entities(edges=replace_constructs(model.edges, (driver,)))
    assert isinstance(_problem(model)[0].state_space.recurrence, ParallelRecurrences)
    total = next(c for c in model.constructs if c.name == "total")
    potential = total.revised(
        dynamics=(
            PotentialMechanismSpec(
                id=MechanismId("mechanism:quadratic_total"), expression=state(total.id) ** 2 * 0.2
            ),
        )
    )
    compiled, problem = _problem(
        model.with_entities(edges=replace_constructs(model.edges, (potential,)))
    )
    assert isinstance(compiled.state_space.recurrence, ParallelRecurrences)
    indices = {item.name: i for i, item in enumerate(compiled.states)}
    coordinates = (
        jnp.zeros((4, len(compiled.states)))
        .at[:, indices["driver"]]
        .set(jnp.array([0.5, -0.2, 0.8, 1.0]))
    )
    target = problem.runtime
    path = target.replay_path(target.context(target.initial_position, target.times), coordinates)
    np.testing.assert_allclose(path[:, indices["total"]], [0.0, 0.5, 0.1, 0.86], atol=1e-6)
    total = total.revised(
        dynamics=(
            DriftMechanismSpec(
                id=MechanismId("mechanism:nonlinear_total"), expression=state(total.id) ** 2
            ),
        )
    )
    prepared = compile_fit_fixture(
        model.with_entities(edges=replace_constructs(model.edges, (total,)))
    )
    compiled = prepared.compiled_dynamical_model
    assert isinstance(compiled.state_space.recurrence, SequentialRecurrences)
    panel = bind_panel_fixture(
        compiled, jnp.full((4, len(compiled.observations)), jnp.nan), jnp.arange(4.0)
    )
    failure = fit_marginal_particle_gibbs(
        prepared.prior_runtime_bundle,
        panel,
        compute_budget=COMPUTE_BUDGET,
        sampler=SamplerSpec(),
        clock=lambda: 0.0,
    )
    assert isinstance(failure, InitializationFailure)
    assert "O(log T)" in failure.message
    assert "total" in failure.message


@pytest.mark.inference(concern="recovery")
@pytest.mark.parametrize("method", ["amala", "amala_plus"])
def test_parallel_accumulator_kernel_preserves_gaussian_joint_distribution(method):
    compiled, problem = _problem(_model())
    target = problem.runtime
    indices = {item.name: i for i, item in enumerate(compiled.states)}
    b, x, a, y = (indices[name] for name in ("baseline", "driver", "total", "outcome"))
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_particles=9,
        num_parameter_particles=2,
        param_step_size=0.1,
        parameter_proposal="random_walk",
        dsmc=DsmcSpec(drift=method),
    )

    def draw_and_step(key):
        draw_key, step_key = jax.random.split(key)
        noises = jax.random.normal(draw_key, (4, 3))
        path = jnp.zeros((4, 4)).at[:, b].set(noises[0, 0])
        path = path.at[0, x].set(noises[0, 1]).at[0, y].set(noises[0, 2])
        for t in range(1, 4):
            path = path.at[t, x].set(path[t - 1, x] + path[0, b] + 0.4 * noises[t, 1])
            path = path.at[t, a].set(path[t - 1, a] + path[t - 1, x])
            path = path.at[t, y].set(path[t - 1, y] + path[t - 1, a] + 0.4 * noises[t, 2])
        initial = _initial(target, path, kernel=kernel, position=path[0, b : b + 1])._replace(
            latent_delta=jnp.full(4, 0.5)
        )
        after, _ = kernel.step_fn(initial, step_key)
        chosen = jnp.array([b, x, a, y])
        return path[-1, chosen], after.latent_trajectory[-1, chosen]

    previous, updated = jax.jit(jax.vmap(draw_and_step))(
        jax.random.split(jax.random.key(814), 1024)
    )
    before, after = np.asarray(previous), np.asarray(updated)
    differences = np.concatenate((after - before, after**2 - before**2), axis=1)
    errors = np.abs(differences.mean(axis=0)) / (
        differences.std(axis=0, ddof=1) / np.sqrt(len(before))
    )
    assert np.max(errors) < 4.5, errors


def test_parallel_accumulator_preserves_an_exact_random_initial_value():
    model = _model()
    total = next(c for c in model.constructs if c.name == "total")
    total = total.revised(
        coefficients=tuple(
            coefficient(1.0, "initial_scale") if c.role == "initial_scale" else c
            for c in total.coefficients
        ),
        indicators=tuple(
            indicator.revised(
                likelihood=LikelihoodSpec(
                    law=DeltaLawSpec(v=state(total.id)),
                    reasoning="An exact initial accumulator reading",
                    standardized=False,
                )
            )
            for indicator in total.indicators
        ),
    )
    model = model.with_entities(edges=replace_constructs(model.edges, (total,)))
    compiled = compile_fit_fixture(model).compiled_dynamical_model
    indices = {item.name: i for i, item in enumerate(compiled.states)}
    column = next(
        i for i, law in enumerate(compiled.observations) if law.state_index == indices["total"]
    )
    compiled, problem = _problem(
        model, jnp.full((4, len(compiled.observations)), jnp.nan).at[0, column].set(0.7)
    )
    target = problem.runtime
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_particles=4,
        num_parameter_particles=2,
        param_step_size=0.01,
        exact_constraints=target.exact_constraints,
    )
    initial = _initial(target, jnp.zeros((4, len(compiled.states))), kernel=kernel)
    after, _ = jax.jit(kernel.step_fn)(initial, jax.random.key(582))
    path = after.latent_trajectory
    assert np.isfinite(after.complete_log_posterior)
    np.testing.assert_allclose(
        path[:, indices["total"]],
        jnp.concatenate((jnp.array([0.7]), 0.7 + jnp.cumsum(path[:-1, indices["driver"]]))),
        atol=1e-6,
    )


@pytest.mark.parametrize("accumulator_observation", ["gaussian", "delta", "delta_fixed_initial"])
def test_tree_factors_and_replay_correction_preserve_joint_density_and_gradients(
    accumulator_observation,
):
    model = _model()
    if accumulator_observation == "delta_fixed_initial":
        driver = next(c for c in model.constructs if c.name == "driver")
        driver = driver.revised(
            coefficients=tuple(
                coefficient(0.0, "initial_scale") if c.role == "initial_scale" else c
                for c in driver.coefficients
            )
        )
        model = model.with_entities(edges=replace_constructs(model.edges, (driver,)))
    total = next(c for c in model.constructs if c.name == "total")
    if accumulator_observation.startswith("delta"):
        total = total.revised(
            indicators=tuple(
                indicator.revised(
                    likelihood=LikelihoodSpec(
                        law=DeltaLawSpec(v=state(total.id)),
                        reasoning="Exact initial accumulator reading",
                        standardized=False,
                    )
                )
                for indicator in total.indicators
            )
        )
    model = model.with_entities(edges=replace_constructs(model.edges, (total,)))
    compiled = compile_fit_fixture(model).compiled_dynamical_model
    observations = jnp.full((4, len(compiled.observations)), 0.15)
    if accumulator_observation.startswith("delta"):
        column = next(
            i
            for i, observation in enumerate(compiled.observations)
            if compiled.states[observation.state_index].name == "total"
        )
        observations = observations.at[:, column].set(jnp.nan).at[0, column].set(0.0)
    _, problem = _problem(model, observations)
    target = problem.runtime
    local = target.tree_target()
    expected_local = {"gaussian": 0, "delta": 3, "delta_fixed_initial": 0}[accumulator_observation]
    assert len(target.observation_partition.local) == expected_local
    position = target.initial_position.at[-1].set(0.2)
    path = jnp.arange(16.0).reshape(4, 4) / 25

    def split_score(z, x):
        context = target.context(z, target.times)
        return local.factor_log_prob(context, x, local.observations) + target.replay_correction(
            local, context, x
        )

    actual = jax.jit(jax.value_and_grad(split_score, argnums=(0, 1)))(position, path)
    expected = jax.value_and_grad(
        lambda z, x: target.path_log_prob(target.context(z, target.times), x, target.observations),
        argnums=(0, 1),
    )(position, path)
    for measured, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert np.isfinite(measured).all()
        np.testing.assert_allclose(measured, reference, rtol=2e-5, atol=2e-5)

    context = target.context(position, target.times)

    def prefixes(coordinates):
        replayed = target.replay_path(context, coordinates)
        incoming = jax.vmap(
            lambda previous, current, t: target.transition_log_prob(context, previous, current, t)
        )(replayed[:-1], replayed[1:], jnp.arange(1, len(path)))
        return jnp.cumsum(
            jnp.concatenate((target.initial_log_prob(context, replayed[0])[None], incoming))
            + target.observation_log_probs(context, replayed, target.observations)
        )

    increments, filtering_gradient = target.filtering_log_prob_and_grad(
        context, path, target.observations
    )
    jacobian = jax.jacrev(prefixes)(path)
    np.testing.assert_allclose(jnp.cumsum(increments), prefixes(path), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        filtering_gradient,
        jacobian[jnp.arange(len(path)), jnp.arange(len(path))],
        rtol=2e-5,
        atol=2e-5,
    )


def test_observation_partition_keeps_free_loadings_and_joint_gaussian_blocks():
    compiled, _ = _problem(_model())
    state_space = compiled.state_space
    recurrence = state_space.recurrence.indices[0]
    # A free loading with a zero template still depends on the recurrence.
    template = jnp.zeros_like(compiled.loading_block.template)
    support = np.zeros_like(compiled.loading_block.free_support)
    support[0, recurrence] = True
    loading = replace(
        compiled.loading_block,
        template=template,
        free_support=support,
        free_positions=((0, recurrence),),
    )
    partition = compile_observation_partition(
        loading, tuple(observation.law for observation in compiled.observations), state_space
    )
    assert partition.local == ()
    assert partition.replayed == tuple(range(len(compiled.observations)))


def test_exact_static_readings_share_one_value_and_inconsistent_constraints_are_rejected():
    from nof1_causal_lab.models.ssm.inference.conditioning import compile_exact_state_constraints
    from nof1_causal_lab.models.ssm.preparation.preflight import (
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
