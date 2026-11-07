"""Interval densities, sequential window memory, and posterior recovery."""

from __future__ import annotations

from dataclasses import replace

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.artifacts.sampling import (
    AuxiliaryKalmanMethodSpec,
    BlockedPathUpdateSpec,
    ConditionalParameterProposalSpec,
    GuidedMethodSpec,
    ParameterMeanProposalSpec,
    PathUpdateSpec,
    WholePathUpdateSpec,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from grassmoth.models.ssm.inference.targets.path_factors import prepare_path_factors
from grassmoth.models.ssm.inference.targets.sequential import sequential_observations
from grassmoth_test_support.interval_particle import (
    gaussian_interval_case,
    initialize_interval_chain,
    interval_target,
)

pytestmark = pytest.mark.inference(concern="sampling")


def test_mixed_overlapping_intervals_match_independent_laws_and_gradients():
    times = np.array([0.0, 0.5, 1.5, 2.0, 3.0, 4.5, 5.0])
    summaries = ("last", "sum", "mean", "std")
    observations = np.full((7, 4), np.nan)
    starts = observations.copy()
    starts[:, 0] = times
    observations[1, 0] = 0.2
    for row, column, start, value in (
        (3, 1, 0.25, 3.0),
        (5, 1, 1.75, 5.0),
        (6, 1, 3.25, 2.0),
        (5, 2, 0.75, 0.3),
        (6, 3, 1.75, 0.2),
    ):
        starts[row, column], observations[row, column] = start, value
    target = interval_target(
        times,
        observations,
        starts,
        summaries,
        families=(
            DistributionFamily.GAUSSIAN,
            DistributionFamily.NEGATIVE_BINOMIAL,
            DistributionFamily.GAUSSIAN,
            DistributionFamily.GAUSSIAN,
        ),
    )
    path = jnp.array([[0.0], [0.4], [-0.2], [0.5], [0.1], [-0.3], [0.3]])
    position = jnp.array([0.15])
    context = target.context(position, target.times)

    def integral(values, start, row):
        begin = int(np.searchsorted(times, start, side="right"))
        grid = jnp.asarray(np.r_[start, times[begin : row + 1]])
        boundary = jnp.interp(start, jnp.asarray(times), values)
        samples = jnp.concatenate((boundary[None], values[begin : row + 1]))
        return jnp.trapezoid(samples, grid)

    def independent(z, x):
        response = x[:, 0] + z[0]
        factors = jnp.zeros(7).at[1].set(dist.Normal(response[1], jnp.sqrt(0.2)).log_prob(0.2))
        for row, column in zip(*np.where(np.isfinite(observations[:, 1:])), strict=True):
            column += 1
            start = starts[row, column]
            if column == 1:
                mean = integral(jnp.exp(response), start, row)
                density = dist.NegativeBinomial2(mean, 5.0).log_prob(observations[row, column])
            else:
                width = times[row] - start
                mean = integral(response, start, row) / width
                if column == 3:
                    second = integral(response**2, start, row) / width
                    mean = jnp.sqrt(second - mean**2)
                density = dist.Normal(mean, jnp.sqrt(0.2)).log_prob(observations[row, column])
            factors = factors.at[row].add(density)
        return factors

    expected = jax.jit(independent)(position, path)
    actual = jax.jit(lambda x: target.observation_log_probs(context, x, target.observations))(path)
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    expected_gradient = jax.jit(jax.grad(lambda z, x: independent(z, x).sum(), argnums=(0, 1)))(
        position, path
    )
    actual_gradient = jax.jit(
        jax.grad(
            lambda z, x: target.observation_log_probs(
                target.context(z, target.times), x, target.observations
            ).sum(),
            argnums=(0, 1),
        )
    )(position, path)
    for actual_grad, expected_grad in zip(actual_gradient, expected_gradient, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=2e-5)

    factors = prepare_path_factors(target, context, target.observations)
    cache = jax.jit(factors.initialize)(path)
    for start in (-1, 2, 6):
        changed = (jnp.arange(len(times)) >= start) & (jnp.arange(len(times)) < start + 2)
        candidate = cache.path + jnp.where(changed[:, None], 0.13, 0.0)
        updated = jax.jit(lambda old, new, first: factors.update(old, new, first, 2))(
            cache, candidate, jnp.asarray(start)
        )
        complete = jax.jit(factors.initialize)(candidate)
        for value, expected_value in zip(updated, complete, strict=True):
            np.testing.assert_allclose(value, expected_value, atol=1e-5)
        expected_difference = target.factor_log_prob(
            context, candidate, target.observations
        ) - target.factor_log_prob(context, cache.path, target.observations)
        np.testing.assert_allclose(updated.difference(cache), expected_difference, atol=2e-5)
        cache = updated

    intervals = target.intervals
    assert intervals is not None
    reference_memory = [jnp.zeros(intervals.memory.shape)]
    for index in range(1, len(times)):
        reference_memory.append(
            intervals.advance(context, reference_memory[-1], path[index - 1], index)
        )
    np.testing.assert_allclose(
        jax.jit(intervals.memory_path)(context, path), jnp.stack(reference_memory), atol=2e-6
    )
    current_gradient = jax.jit(
        jax.grad(
            lambda x: intervals.trajectory_log_probs(
                context, x, target.observations, gradient="current"
            ).sum()
        )
    )(path)
    observation_jacobian = jax.jit(jax.jacrev(lambda x: independent(position, x)))(path)
    np.testing.assert_allclose(
        current_gradient, observation_jacobian[jnp.arange(7), jnp.arange(7)], atol=2e-5
    )
    # Two parameter labels exercise label-dependent responses, slot reuse and
    # overlapping windows in the sequential window memory.
    contexts = jax.jit(jax.vmap(lambda z: target.context(z, target.times)))(
        jnp.array([[0.15], [-0.4]])
    )
    sequential = sequential_observations(
        target, contexts, target.context(jnp.zeros(1), target.times)
    )
    memory = jax.jit(sequential.path)(path)
    emitted = jax.jit(jax.vmap(sequential.log_probs))(memory, jnp.arange(len(times)))
    expected_labels = jax.jit(
        jax.vmap(lambda bound: target.observation_log_probs(bound, path, target.observations))
    )(contexts)
    np.testing.assert_allclose(emitted.T, expected_labels, atol=2e-5)
    current_scores = jax.jit(
        jax.vmap(
            lambda state, carried, index: jax.grad(
                lambda value: sequential.log_probs(
                    sequential.replace_current(carried, value, index), index
                )[0]
            )(state)
        )
    )(path, memory, jnp.arange(len(times)))
    np.testing.assert_allclose(current_scores, current_gradient, atol=2e-5)


@pytest.mark.parametrize(
    "method", ["twisted_particle_agrad", "kalman_full", "kalman_2", "kalman_4"]
)
def test_updates_preserve_analytic_interval_posterior(method):

    target, mean, covariance = gaussian_interval_case()
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_parameter_particles=2,
        param_step_size=0.3,
        parameter_proposal="random_walk",
        parameter_reference_path=jnp.asarray(mean[1:, None]),
        path_update=(
            PathUpdateSpec(
                method=AuxiliaryKalmanMethodSpec(),
                update=WholePathUpdateSpec()
                if method == "kalman_full"
                else BlockedPathUpdateSpec(block_length=int(method[-1])),
                proposal=ParameterMeanProposalSpec()
                if method == "kalman_full"
                else ConditionalParameterProposalSpec(),
            )
            if method.startswith("kalman")
            else PathUpdateSpec(
                method=GuidedMethodSpec(kind=method, num_particles=8, kappa=1),
                proposal=ParameterMeanProposalSpec(),
            )
        ),
    )
    chains = 4096
    joint = jnp.asarray(
        np.random.default_rng(29).multivariate_normal(mean, covariance, size=chains)
    )
    initial = jax.jit(
        jax.vmap(lambda point: initialize_interval_chain(target, point, kernel=kernel))
    )(joint)

    @jax.jit
    def sample(states):
        def sweep(current, key):
            updated, _ = jax.vmap(kernel.step_fn)(current, jax.random.split(key, chains))
            return updated, None

        return jax.lax.scan(sweep, states, jax.random.split(jax.random.key(35), 3))[0]

    final = sample(initial)
    draws = np.asarray(jnp.concatenate((final.position, final.latent_trajectory[..., 0]), axis=1))
    np.testing.assert_allclose(draws.mean(0), mean, atol=0.045)
    np.testing.assert_allclose(np.cov(draws.T), covariance, atol=0.055)
    assert np.mean(draws[:, 1:] != np.asarray(joint[:, 1:])) > 0.25


@pytest.mark.inference(concern="recovery")
@pytest.mark.parametrize("kappa", [1, 0])
def test_interval_particle_kernel_recovers_analytic_joint_posterior(kappa):
    target, mean, covariance = gaussian_interval_case()
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_parameter_particles=3,
        param_step_size=0.3,
        parameter_proposal="random_walk",
        parameter_reference_path=jnp.asarray(mean[1:, None]),
        path_update=PathUpdateSpec(
            method=GuidedMethodSpec(kind="twisted_particle_agrad", num_particles=16, kappa=kappa)
        ),
    )
    # Replicate chains to resolve the weakly identified offset/path direction
    # without loosening the analytic moment tolerances.
    chains = 16
    # Draw starts from the declared prior, including its temporal dependence.
    # Independent unit-normal path coordinates can create ten-SD innovations;
    # recovery from those starts measures initialization rather than this kernel.
    joint = np.random.default_rng(24).normal(size=(chains, len(mean)))
    for index in range(2, len(mean)):
        joint[:, index] = 0.7 * joint[:, index - 1] + np.sqrt(0.3) * joint[:, index]
    joint = jnp.asarray(joint)
    initial = jax.vmap(lambda point: initialize_interval_chain(target, point, kernel=kernel))(joint)

    @jax.jit
    def sample(states, keys):
        def sweep(current, key):
            next_states, info = jax.vmap(kernel.step_fn)(current, jax.random.split(key, chains))
            joint = jnp.concatenate(
                [next_states.position, next_states.latent_trajectory[..., 0]], axis=1
            )
            return next_states, (joint, info["latent_frozen_frac"])

        return jax.lax.scan(sweep, states, keys)

    _, (draws, frozen) = sample(initial, jax.random.split(jax.random.key(21), 2200))
    retained_chains = np.asarray(draws[700:])
    assert np.all(np.ptp(retained_chains, axis=0) > 0)
    retained = retained_chains.reshape(-1, len(mean))
    np.testing.assert_allclose(retained.mean(axis=0), mean, atol=0.12)
    np.testing.assert_allclose(np.cov(retained.T), covariance, atol=0.12)
    assert float(jnp.mean(frozen[700:])) < 0.8
    # The observations identify a parameter/path combination; checking only
    # parameter marginals would miss a broken dependence structure.
    contrast = retained[:, 0] + retained[:, 1:].mean(axis=1)
    direction = np.r_[1.0, np.ones(len(mean) - 1) / (len(mean) - 1)]
    assert abs(contrast.mean() - direction @ mean) < 0.07
    assert abs(contrast.var() - direction @ covariance @ direction) < 0.04


def test_calendar_totals_and_grid_refinement_preserve_response_scale():
    errors = []
    for count in (3, 5, 9):
        times = np.linspace(0, 366, count)
        observations = np.full((count, 1), np.nan)
        starts = observations.copy()
        observations[-1, 0], starts[-1, 0] = 500.0, 0.0
        target = interval_target(
            times, observations, starts, ("sum",), families=(DistributionFamily.NEGATIVE_BINOMIAL,)
        )
        assert target.intervals is not None
        context = target.context(jnp.zeros(1), target.times)
        # Integral exp(t/366) dt = 366(e - 1), not exp(mean(t/366))*366.
        path = jnp.asarray(times[:, None] / 366)
        response = jax.vmap(target.intervals.response, in_axes=(None, 0, 0))(
            context, path, jnp.arange(path.shape[0])
        )
        means, _ = target.intervals.operator.project_response_trajectory(response)
        errors.append(abs(float(means[-1, 0]) - 366 * np.expm1(1)))
        constant, _ = target.intervals.operator.project_response_trajectory(
            jnp.ones_like(response) * 2
        )
        np.testing.assert_allclose(constant[-1, 0], 732.0, rtol=1e-6)
    assert errors[1] < errors[0] / 3.9
    assert errors[2] < errors[1] / 3.9


def test_joint_interval_noise_with_different_windows_keeps_cross_channel_covariance():
    times = np.arange(4.0)
    observations = np.full((4, 2), np.nan)
    starts = observations.copy()
    observations[3] = [0.8, -0.2]
    starts[3] = [0.0, 1.0]
    covariance = np.array([[0.4, 0.12], [0.12, 0.2]])
    target = interval_target(times, observations, starts, ("mean", "mean"), covariance=covariance)
    path = jnp.array([[0.0], [0.2], [0.6], [-0.2]])
    context = target.context(jnp.zeros(1), target.times)
    contexts = jax.vmap(lambda z: target.context(z, target.times))(jnp.zeros((1, 1)))
    means = jnp.array([jnp.trapezoid(path[:, 0]) / 3, jnp.trapezoid(path[1:, 0]) / 2])
    expected = dist.MultivariateNormal(means, covariance_matrix=jnp.asarray(covariance)).log_prob(
        observations[3]
    )
    exact = jax.jit(lambda x: target.observation_log_probs(context, x, target.observations))(path)
    np.testing.assert_allclose(exact[3], expected, atol=2e-5)
    sequential = sequential_observations(target, contexts, context)
    memory = jax.jit(sequential.path)(path)
    emitted = jax.jit(jax.vmap(sequential.log_probs))(memory, jnp.arange(len(times)))
    np.testing.assert_allclose(emitted[3, 0], expected, atol=2e-5)


@pytest.mark.inference(concern="recovery")
def test_interval_posterior_converges_under_euler_maruyama_grid_refinement():
    # This Gaussian special case has a continuous-time analytic answer. The
    # evaluated density remains the production particle target on each grid.
    horizon, decay, diffusion = 4.0, 0.3, 0.3
    stationary_variance = diffusion / (2 * decay)
    integrated_decay = -np.expm1(-decay * horizon) / decay
    average_variance = (
        2 * stationary_variance * (horizon - integrated_decay) / decay
        + (1 - stationary_variance) * integrated_decay**2
    ) / horizon**2
    exact_mean = 1.2 / (1 + average_variance + 0.2)
    exact_variance = (average_variance + 0.2) / (1 + average_variance + 0.2)
    errors = []
    for steps in (9, 17, 33):
        times = np.linspace(0, horizon, steps)
        observations = np.full((steps, 1), np.nan)
        starts = observations.copy()
        observations[-1, 0], starts[-1, 0] = 1.2, 0.0
        target = interval_target(times, observations, starts, ("mean",))
        assert target.intervals is not None

        def declared(context, original=target.dynamical_model):
            return eqx.tree_at(
                lambda model: model.state_evolution,
                original(context),
                dsx.LinearGaussianStateEvolution(
                    lambda a, b: jnp.eye(1) * (1 - decay * (b - a)),
                    lambda a, b: jnp.eye(1) * diffusion * (b - a),
                ),
            )

        refined = replace(
            target,
            dynamical_model=declared,
            intervals=replace(target.intervals, dynamical_model=declared),
        )

        def density(joint, refined=refined):
            return refined.log_posterior(
                joint[:1], joint[1:, None], refined.observations, refined.times
            )

        origin = jnp.zeros(steps + 1)
        precision = -np.asarray(jax.jit(jax.hessian(density))(origin))
        covariance = np.linalg.inv(precision)
        mean = covariance @ np.asarray(jax.grad(density)(origin))
        errors.append(abs(mean[0] - exact_mean) + abs(covariance[0, 0] - exact_variance))
    assert errors[1] < 0.6 * errors[0]
    assert errors[2] < 0.6 * errors[1]
    assert errors[2] < 0.01
