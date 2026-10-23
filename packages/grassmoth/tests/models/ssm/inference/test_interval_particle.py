"""Interval densities, particle-tree factorization, and posterior recovery."""

from __future__ import annotations

from dataclasses import replace

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy.stats import norm

from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.artifacts.sampling import DsmcSpec
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from grassmoth.models.ssm.inference.samplers.particle.contract import PaidMixture
from grassmoth.models.ssm.inference.targets.intervals import gather_interval_statistics
from grassmoth_test_support.interval_particle import (
    gaussian_interval_case,
    initialize_interval_chain,
    interval_target,
)

pytestmark = pytest.mark.inference(concern="sampling")


def _interval_tree_score(intervals, contexts, bank):
    stats = intervals.leaf_statistics(contexts, bank)
    total = jnp.zeros((bank.shape[1], jax.tree.leaves(contexts)[0].shape[0]))
    for level in intervals.tree.levels[1:]:
        if level.emission_slots.shape[1]:
            # Pairwise selection must preserve every crossed path and parameter
            # label, including zero-padded child slots and joint channel noise.
            def compare_pairs(left, right, node, level=level):
                full_left = gather_interval_statistics(left, jnp.asarray(level.left_slots)[node])
                full_right = gather_interval_statistics(right, jnp.asarray(level.right_slots)[node])
                expected = jax.vmap(
                    lambda row: intervals.emission_log_probs(
                        contexts, row[None] + full_right, level, node
                    )
                )(full_left)
                actual = intervals.pairwise_emission_log_probs(contexts, left, right, level, node)
                return actual, expected

            actual, expected = jax.vmap(compare_pairs)(
                stats[::2], stats[1::2], jnp.arange(len(level.left_slots))
            )
            np.testing.assert_allclose(actual, expected, atol=2e-5)
        stats = jax.vmap(
            lambda left, right, node, level=level: (
                gather_interval_statistics(left, jnp.asarray(level.left_slots)[node])
                + gather_interval_statistics(right, jnp.asarray(level.right_slots)[node])
            )
        )(stats[::2], stats[1::2], jnp.arange(len(level.left_slots)))
        if level.emission_slots.shape[1]:
            total = total + jax.vmap(
                lambda s, node, level=level: intervals.emission_log_probs(contexts, s, level, node)
            )(stats, jnp.arange(len(level.left_slots))).sum(axis=0)
    return total


def test_mixed_overlapping_intervals_match_independent_laws_gradients_and_tree():
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

    expected = independent(position, path)
    actual = target.observation_log_probs(context, path, target.observations)
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    expected_gradient = jax.grad(lambda z, x: independent(z, x).sum(), argnums=(0, 1))(
        position, path
    )
    actual_gradient = jax.grad(
        lambda z, x: target.observation_log_probs(
            target.context(z, target.times), x, target.observations
        ).sum(),
        argnums=(0, 1),
    )(position, path)
    for actual_grad, expected_grad in zip(actual_gradient, expected_gradient, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=2e-5)

    intervals = target.intervals
    assert intervals is not None
    current_gradient = jax.grad(
        lambda x: intervals.trajectory_log_probs(
            context, x, target.observations, gradient="current"
        ).sum()
    )(path)
    observation_jacobian = jax.jacrev(lambda x: independent(position, x))(path)
    np.testing.assert_allclose(
        current_gradient, observation_jacobian[jnp.arange(7), jnp.arange(7)], atol=2e-5
    )
    # Two parameter labels, two distinct paths, and seven real leaves exercise
    # label-dependent responses, slot reuse, overlapping windows, and padding.
    contexts = jax.vmap(lambda z: target.context(z, target.times))(jnp.array([[0.15], [-0.4]]))
    bank = jnp.stack([path, path + 0.6], axis=1)
    total = _interval_tree_score(intervals, contexts, bank)
    reference = jax.vmap(
        lambda x: jax.vmap(
            lambda bound: (
                target.observation_log_probs(bound, x, target.observations).sum()
                - target.observation_increment(bound, x[1], 1, target.observations)
            )
        )(contexts)
    )(jnp.swapaxes(bank, 0, 1))
    np.testing.assert_allclose(total, reference, atol=2e-5)
    for channels in ((0, 1), (2, 3)):
        selected = intervals.select_channels(channels)
        assert selected is not None
        selected_observations = jnp.where(
            jnp.asarray(np.isin(np.arange(4), channels)), target.observations, jnp.nan
        )
        selected_reference = jax.vmap(
            lambda x, selected_observations=selected_observations: jax.vmap(
                lambda bound: (
                    target.observation_log_probs(bound, x, selected_observations).sum()
                    - target.observation_increment(bound, x[1], 1, selected_observations)
                )
            )(contexts)
        )(jnp.swapaxes(bank, 0, 1))
        np.testing.assert_allclose(
            _interval_tree_score(selected, contexts, bank), selected_reference, atol=2e-5
        )
    assert intervals.select_channels((0,)) is None


@pytest.mark.inference(concern="recovery")
@pytest.mark.parametrize(
    ("method", "kappa"),
    [("amala", 1.0), ("amala_plus", 1.0), ("amala_plus", 0.0)],
)
def test_interval_particle_kernel_recovers_analytic_joint_posterior(method, kappa):
    target, mean, covariance = gaussian_interval_case()
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_parameter_particles=3,
        param_step_size=0.3,
        parameter_proposal="random_walk",
        parameter_reference_path=jnp.asarray(mean[1:, None]),
        paid_mix=PaidMixture(
            jnp.zeros((len(mean) - 1, 1)), jnp.ones((len(mean) - 1, 1)), jnp.asarray([True])
        ),
        path_update=DsmcSpec(
            num_particles=12,
            method=(
                "particle_rwm"
                if (method if kappa else "none") == "none"
                else "particle_amala"
                if (method if kappa else "none") == "amala"
                else "particle_amala_plus"
            ),
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


@pytest.mark.parametrize(
    ("method", "kappa"),
    [("amala", 1.0), ("amala_plus", 1.0), ("amala_plus", 0.0)],
)
def test_reusable_tree_levels_preserve_joint_posterior_with_padding(method, kappa):
    """An unbalanced 33-time tree exercises multiple shared upper levels and ancestry."""
    target, mean, covariance = gaussian_interval_case(steps=33)
    chol = np.linalg.cholesky(covariance)
    size = 2048
    before = mean + np.random.default_rng(410).normal(size=(size, mean.size)) @ chol.T
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_parameter_particles=2,
        param_step_size=0.3,
        parameter_proposal="random_walk",
        parameter_reference_path=jnp.asarray(mean[1:, None]),
        path_update=DsmcSpec(
            num_particles=4,
            method=(
                "particle_rwm"
                if (method if kappa else "none") == "none"
                else "particle_amala"
                if (method if kappa else "none") == "amala"
                else "particle_amala_plus"
            ),
            pilot_mixture=False,
        ),
    )

    def advance(point, key):
        initial = initialize_interval_chain(target, point, kernel=kernel)
        updated, _ = kernel.step_fn(initial, key)
        return jnp.concatenate((updated.position, updated.latent_trajectory[:, 0]))

    after = np.asarray(
        jax.jit(jax.vmap(advance))(jnp.asarray(before), jax.random.split(jax.random.key(411), size))
    )
    initial = np.linalg.solve(chol, (before - mean).T).T
    observed = np.linalg.solve(chol, (after - mean).T).T
    left, right = np.triu_indices(mean.size)
    differences = np.column_stack(
        (
            observed - initial,
            observed[:, left] * observed[:, right] - initial[:, left] * initial[:, right],
        )
    )
    critical = norm.isf(0.001 / (2 * differences.shape[1]))
    uncertainty = differences.std(axis=0, ddof=1) / np.sqrt(size)
    assert np.all(np.abs(differences.mean(axis=0)) <= critical * uncertainty)
    assert np.all(np.mean((after - before) ** 2, axis=0) > 1e-4 * np.diag(covariance))


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
        response = jax.vmap(target.intervals.response, in_axes=(None, 0))(context, path)
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
    contexts = jax.vmap(lambda z: target.context(z, target.times))(jnp.zeros((1, 1)))
    intervals = target.intervals
    assert intervals is not None
    density = _interval_tree_score(intervals, contexts, path[:, None, :])
    means = jnp.array([jnp.trapezoid(path[:, 0]) / 3, jnp.trapezoid(path[1:, 0]) / 2])
    expected = dist.MultivariateNormal(means, covariance_matrix=jnp.asarray(covariance)).log_prob(
        observations[3]
    )
    np.testing.assert_allclose(density[0, 0], expected, atol=2e-5)


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
