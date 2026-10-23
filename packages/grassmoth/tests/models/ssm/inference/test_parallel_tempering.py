import logging
from datetime import UTC, datetime
from time import perf_counter
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.expressions import state
from grassmoth.artifacts.model.likelihood import DeltaLawSpec
from grassmoth.artifacts.sampling import DsmcSpec, GaussianReferenceTemperingSpec
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import (
    run_marginal_particle_gibbs,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.tempering import (
    ReplicaExchangeKernel,
    adapt_temperature_ladder,
    exchange_permutation,
)
from grassmoth.models.ssm.inference.targets.gaussian_reference import GaussianReferenceBridge
from grassmoth.models.ssm.inference.warmup.gaussian_reference import build_gaussian_reference
from grassmoth.models.ssm.inference.warmup.initialization import ParticleInitialization
from grassmoth_test_support.compute import COMPUTE_BUDGET, compile_program, prepare_program
from grassmoth_test_support.inference import compile_fit_fixture
from grassmoth_test_support.interval_particle import (
    gaussian_interval_case,
    initialize_interval_chain,
)
from packages.grassmoth.tests.models.ssm.compile.test_exact_state_space import (
    _initial,
    _model,
    _problem,
)

"""Replica exchange preserves exact cold laws, compiled support, and chain ownership."""


pytestmark = pytest.mark.inference(concern="sampling")


def _reference(target):
    context = target.context(target.initial_position, target.times)
    path = target.initial_path(context, exact_constraints=target.exact_constraints)
    return build_gaussian_reference(
        target,
        positions=target.initial_position[None],
        paths=path[None],
        parameter_chol=None,
        scale=1.0,
    )


def _local_kernel(target):
    return build_marginal_particle_gibbs_kernel(
        target,
        path_update=DsmcSpec(num_particles=4),
        num_parameter_particles=2,
        param_step_size=0.4,
        parameter_proposal="pseudo_langevin",
        exact_constraints=target.exact_constraints,
    )


@pytest.mark.parametrize("beta", [0.0, 0.37, 1.0])
@pytest.mark.inference(concern="warmup")
def test_bridge_scores_and_gradients_include_interval_likelihood_once(beta):
    target, _, _ = gaussian_interval_case()
    reference = _reference(target)
    tempered = target.at_temperature(reference, jnp.asarray(beta))
    joint = jnp.linspace(-0.8, 0.9, target.times.size + 1)

    def reference_density(q):
        return (
            dist.Normal(0, 1).log_prob(q[:2]).sum()
            + dist.Normal(0.7 * q[1:-1], jnp.sqrt(0.3)).log_prob(q[2:]).sum()
        )

    def expected(q):
        exact = target.log_posterior(q[:1], q[1:, None], target.observations, target.times)
        return beta * exact + (1 - beta) * reference_density(q)

    def actual(q):
        return tempered.log_posterior(q[:1], q[1:, None], target.observations, target.times)

    actual_value, actual_gradient = jax.jit(jax.value_and_grad(actual))(joint)
    expected_value, expected_gradient = jax.value_and_grad(expected)(joint)
    np.testing.assert_allclose(actual_value, expected_value, rtol=2e-6)
    np.testing.assert_allclose(actual_gradient, expected_gradient, atol=2e-5)
    with jax.enable_x64():
        exact_value = jnp.asarray(2.123456789, dtype=jnp.float64)
        reference_value = jnp.asarray(-0.2, dtype=jnp.float32)
        bridge = GaussianReferenceBridge(reference, jnp.asarray(beta))
        mixed = jax.jit(lambda: bridge.combine(lambda: exact_value, reference_value))()
        assert mixed.dtype == jnp.float64
        np.testing.assert_allclose(
            mixed,
            beta * exact_value + (1 - beta) * reference_value.astype(jnp.float64),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.inference(concern="warmup")
def test_parallel_reference_sampling_has_the_declared_temporal_covariance():
    target, _, _ = gaussian_interval_case()
    reference = _reference(target)
    parameters, paths = jax.jit(jax.vmap(reference.sample))(
        jax.random.split(jax.random.key(109), 8192)
    )
    # Independent AR(1) covariance, including the nonstationary initial variance.
    steps = target.times.size
    propagation = np.tril(0.7 ** (np.arange(steps)[:, None] - np.arange(steps)[None, :]))
    expected = propagation @ np.diag([1, *([0.3] * (steps - 1))]) @ propagation.T
    joint = np.column_stack((parameters[:, 0], paths[:, :, 0]))
    np.testing.assert_allclose(joint.mean(axis=0), 0, atol=0.04)
    np.testing.assert_allclose(np.cov(joint[:, 1:], rowvar=False), expected, atol=0.035)
    np.testing.assert_allclose(np.cov(joint, rowvar=False)[0], [1, *([0] * steps)], atol=0.04)


@pytest.mark.inference(concern="warmup")
def test_reference_and_swaps_preserve_static_accumulator_and_exact_coordinates():
    model = _model()
    constructs = tuple(
        construct.revised(
            indicators=tuple(
                indicator.revised(
                    likelihood=indicator.likelihood.revised(
                        law=DeltaLawSpec(v=state(construct.id)), standardized=False
                    )
                )
                for indicator in construct.indicators
            )
        )
        if construct.name == "driver"
        else construct
        for construct in model.constructs
    )
    model = model.with_entities(edges=replace_constructs(model.edges, constructs))
    compiled = compile_fit_fixture(model).compiled_dynamical_model
    column = next(
        i
        for i, observation in enumerate(compiled.observations)
        if isinstance(observation.law, DeltaLawSpec)
    )
    readings = jnp.full((4, len(compiled.observations)), jnp.nan).at[2, column].set(0.8)
    compiled, problem = _problem(model, readings)
    indices = {s.name: i for i, s in enumerate(compiled.states)}
    baseline, driver, total = (indices[n] for n in ("baseline", "driver", "total"))
    target = problem.runtime
    reference = _reference(target)
    assert target.initial_position.size == 1
    assert not reference.free_mask[:, baseline].any()
    assert not reference.free_mask[:, total].any()
    assert not reference.free_mask[2, driver]
    parameters, coordinates = jax.vmap(reference.sample)(jax.random.split(jax.random.key(12), 3))
    kernel = _local_kernel(target)
    states = jax.vmap(
        lambda position, path: _initial(target, path, kernel=kernel, position=position)
    )(parameters, coordinates)
    betas = jnp.array([1.0, 0.4, 0.0])
    exchange = ReplicaExchangeKernel(
        target, reference, _local_kernel, GaussianReferenceTemperingSpec(num_temperatures=3)
    )
    states = exchange.rebind(states, betas)
    after, _, _ = jax.jit(exchange.step)(
        states, jax.random.split(jax.random.key(13), 3), betas, jnp.asarray(0)
    )
    paths = after.latent_trajectory
    np.testing.assert_array_equal(
        paths[:, :, baseline], jnp.repeat(paths[:, :1, baseline], 4, axis=1)
    )
    np.testing.assert_allclose(
        paths[:, 1:, total], jnp.cumsum(paths[:, :-1, driver], axis=1), atol=1e-6
    )
    assert target.exact_constraints is not None
    np.testing.assert_array_equal(
        paths[:, 2, driver], jnp.repeat(target.exact_constraints.values[2, driver], 3)
    )
    assert parameters.shape == (3, 1)
    np.testing.assert_array_equal(paths[:, 0, baseline], after.position[:, 0])
    np.testing.assert_allclose(
        after.trajectory_log_prob, exchange.rebind(after, betas).trajectory_log_prob, atol=2e-5
    )
    np.testing.assert_allclose(
        after.complete_log_posterior,
        exchange.rebind(after, betas).complete_log_posterior,
        atol=2e-5,
    )
    np.testing.assert_array_equal(after.param_step_size, states.param_step_size)
    np.testing.assert_array_equal(after.latent_delta, states.latent_delta)


def test_exchange_ratio_uses_reference_and_adaptation_keeps_strict_endpoints():
    betas = jnp.array([1.0, 0.8, 0.2, 0.0])
    points = jnp.array([-1.0, 0.3, 1.7, -0.5])
    exact = dist.Normal(1.0, 0.7).log_prob(points)
    reference = dist.Normal(0.0, 2.0).log_prob(points)
    energies = exact - reference
    keys = jax.random.split(jax.random.key(781), 10000)
    _, records = jax.vmap(lambda key: exchange_permutation(betas, energies, jnp.asarray(0), key))(
        keys
    )
    for edge in (0, 2):

        def density(beta, i):
            return beta * exact[i] + (1 - beta) * reference[i]

        ratio = (
            density(betas[edge], edge + 1)
            + density(betas[edge + 1], edge)
            - density(betas[edge], edge)
            - density(betas[edge + 1], edge + 1)
        )
        np.testing.assert_allclose(
            records.accepted[:, edge].mean(), jnp.minimum(1, jnp.exp(ratio)), atol=0.015
        )
    assert not records.attempted[:, 1].any()
    adapted = adapt_temperature_ladder(betas, jnp.array([[0.0, 40.0, 5.0]]), jnp.full((1, 3), 50))
    assert adapted[0] == 1
    assert adapted[-1] == 0
    assert np.all(np.diff(adapted) < 0)
    assert abs(adapted[1] - adapted[2]) < abs(betas[1] - betas[2])
    with jax.enable_x64():
        mixed = adapt_temperature_ladder(
            betas.astype(jnp.float32), jnp.ones((1, 3), dtype=jnp.float64), jnp.full((1, 3), 50)
        )
        assert mixed.dtype == jnp.float32


@pytest.mark.inference(concern="recovery")
@pytest.mark.inference(concern="warmup")
def test_local_moves_refresh_and_swaps_preserve_all_analytic_bridge_moments():
    target, posterior_mean, posterior_covariance = gaussian_interval_case()
    reference = _reference(target)
    betas = jnp.array([1.0, 0.45, 0.0])
    exchange = ReplicaExchangeKernel(
        target, reference, _local_kernel, GaussianReferenceTemperingSpec(num_temperatures=3)
    )
    dimension = posterior_mean.size

    def reference_density(q):
        return reference.parameter_log_prob(q[:1]) + reference.path_log_prob(q[1:, None])

    reference_precision = -np.asarray(jax.hessian(reference_density)(jnp.zeros(dimension)))
    posterior_precision = np.linalg.inv(posterior_covariance)
    covariances = np.linalg.inv(
        np.asarray(betas)[:, None, None] * posterior_precision
        + (1 - np.asarray(betas))[:, None, None] * reference_precision
    )
    means = np.einsum(
        "bij,bj->bi",
        covariances,
        np.asarray(betas)[:, None] * (posterior_precision @ posterior_mean)[None],
    )
    repeats = 2048
    rng = np.random.default_rng(925)
    joint = np.stack(
        [
            rng.multivariate_normal(mean, covariance, repeats)
            for mean, covariance in zip(means, covariances, strict=True)
        ],
        axis=1,
    )
    kernel = _local_kernel(target)
    states = jax.vmap(lambda q: initialize_interval_chain(target, q, kernel=kernel))(
        jnp.asarray(joint.reshape(-1, dimension))
    )
    states = exchange.rebind(states, betas)
    # Tuning is a property of the temperature slot, and must not follow swaps.
    states = states._replace(param_step_size=jnp.tile(jnp.array([0.2, 0.5, 0.9]), repeats))
    step = jax.jit(exchange.step)
    for parity in (0, 1):
        after, _, swaps = step(
            states,
            jax.random.split(jax.random.key(81 + parity), repeats * 3),
            betas,
            jnp.asarray(parity),
        )
        np.testing.assert_array_equal(after.param_step_size, states.param_step_size)
        assert swaps.accepted.any()
        states = after
    samples = np.concatenate(
        (np.asarray(states.position), np.asarray(states.latent_trajectory[:, :, 0])), axis=1
    ).reshape(repeats, 3, dimension)
    standard_error = np.sqrt(np.diagonal(covariances, axis1=-2, axis2=-1) / repeats)
    assert np.max(np.abs(samples.mean(axis=0) - means) / standard_error) < 4.5
    for level in range(3):
        np.testing.assert_allclose(
            np.cov(samples[:, level], rowvar=False), covariances[level], atol=0.085
        )
    rebound = exchange.rebind(states, betas)
    np.testing.assert_allclose(
        states.complete_log_posterior, rebound.complete_log_posterior, atol=2e-4, rtol=2e-5
    )
    # A fixed zero pilot gives an independent analytic parameter-gradient
    # oracle. Check it after exchanges and after changing the interior beta;
    # gradients must follow the arriving theta and the receiving temperature.
    adapted_betas = jnp.array([1.0, 0.7, 0.0])
    adapted = exchange.rebind(states, adapted_betas)
    for cached, ladder in ((states, betas), (adapted, adapted_betas)):
        position = np.asarray(cached.position[:, 0])
        exact_gradient = (posterior_precision @ posterior_mean)[0] - posterior_precision[
            0, 0
        ] * position
        beta = np.tile(ladder, repeats)
        expected_gradient = beta * exact_gradient - (1 - beta) * position
        np.testing.assert_allclose(
            cached.parameter_gradient[:, 0], expected_gradient, rtol=2e-5, atol=2e-4
        )


@pytest.mark.inference(concern="warmup")
@pytest.mark.parametrize(
    ("tempered", "adaptation_scheme"), [(False, "dual_averaging"), (True, "simple")]
)
def test_runner_retains_only_cold_chains_and_freezes_ladder_before_retention(
    tempered, adaptation_scheme
):
    target, _, _ = gaussian_interval_case()
    reference = _reference(target)
    spec = GaussianReferenceTemperingSpec(num_temperatures=3, adaptation_interval=2)
    exchange = ReplicaExchangeKernel(target, reference, _local_kernel, spec)
    result = run_marginal_particle_gibbs(
        target,
        kernel=_local_kernel(target),
        compute_budget=COMPUTE_BUDGET,
        num_warmup=40,
        num_samples=27,
        num_chains=2,
        key=jax.random.PRNGKey(711),
        adaptation_rate=0.05,
        adaptation_scheme=adaptation_scheme,
        initialization=Mock(
            spec=ParticleInitialization,
            positions=jnp.broadcast_to(target.initial_position, (2, 1)),
            paths=jnp.broadcast_to(reference.path_mean, (2, *reference.path_mean.shape)),
            parameter_chol=jnp.eye(1),
        ),
        tempering=exchange if tempered else None,
        clock=perf_counter,
        compiler=compile_program,
        prepare_program=prepare_program,
        logger=logging.getLogger(__name__),
        wall_clock=lambda: datetime.now(UTC),
    )
    assert result.grouped_positions.shape == (2, 27, 1)
    assert result.latent_paths is not None
    assert result.latent_paths.shape == (2, 27, 9, 1)
    assert result.initial_param_step_size.shape == result.final_param_step_size.shape == (2,)
    assert result.final_latent_delta.shape == (2, 9)
    if tempered:
        assert result.tempering is not None
        assert result.tempering.ladder_frozen_at_warmup_step == 20
        assert result.tempering.inverse_temperatures[0] == 1
        assert result.tempering.inverse_temperatures[-1] == 0
        assert tuple(
            tuple(edge.attempted for edge in chain) for chain in result.tempering.swaps
        ) == (
            (14, 13),
            (14, 13),
        )
        assert sum(edge.accepted for chain in result.tempering.swaps for edge in chain) > 0
    else:
        assert result.tempering is None
    score = jax.vmap(
        jax.vmap(
            lambda position, path: target.log_posterior(
                position, path, target.observations, target.times
            )
        )
    )
    np.testing.assert_allclose(
        result.complete_log_posterior_history,
        score(result.grouped_positions, result.latent_paths),
        atol=2e-5,
    )
    observation_scores = jax.vmap(
        jax.vmap(
            lambda position, path: target.observation_log_probs(
                target.context(position, target.times), path, target.observations
            )
        )
    )
    np.testing.assert_allclose(
        result.observation_log_probs,
        observation_scores(result.grouped_positions, result.latent_paths),
        atol=2e-5,
    )
