"""Independent Gaussian oracles for guided windows, their auxiliary law and weights."""

from dataclasses import replace

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, nbinom, norm

from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.artifacts.sampling import ParticleAGradSpec, TwistedParticleAGradSpec
from grassmoth.models.ssm.inference.samplers.particle.adaptation import guided_delta_limit
from grassmoth.models.ssm.inference.samplers.particle.context import build_smoother_context
from grassmoth.models.ssm.inference.samplers.particle.guided import (
    _memory_guide,
    _window_backward,
    guided_step,
    trace_guided_step,
)
from grassmoth.models.ssm.inference.samplers.particle.setup import build_smoother_setup
from grassmoth.models.ssm.inference.targets.sequential import sequential_observations
from grassmoth_test_support.interval_particle import interval_target

pytestmark = pytest.mark.inference(concern="sampling")


def _prior_covariance(steps):
    mapping = np.eye(steps)
    for index in range(1, steps):
        mapping[index, :index] = 0.7 * mapping[index - 1, :index]
    return (mapping * np.r_[1.0, np.full(steps - 1, 0.3)]) @ mapping.T


@pytest.mark.parametrize("variance", [0.15, 3.2])
def test_memory_guide_matches_dense_auxiliary_conditioning(variance):
    times = np.arange(3.0)
    observed = np.array([[np.nan], [np.nan], [0.7]])
    starts = np.array([[np.nan], [np.nan], [0.0]])
    target = interval_target(times, observed, starts, ("mean",))
    context = target.context(jnp.zeros(1), target.times)
    contexts = jax.tree.map(lambda value: value[None], context)
    memory = sequential_observations(target, contexts, target.context(jnp.zeros(1), target.times))
    auxiliary = jnp.array([[0.1], [-0.2], [0.4]])
    mask = target.independent_mask
    variances = jnp.full(auxiliary.shape, variance)

    def observe(value, index):
        def reading(current):
            return memory.log_probs(memory.replace_current(value, current, index), index)[0]

        return memory.log_probs(value, index), jax.grad(reading)(memory.state(value))

    _, geometry = jax.jit(
        lambda: _memory_guide(
            target,
            context,
            memory,
            observe,
            jnp.zeros(1),
            auxiliary,
            variances,
            mask,
            TwistedParticleAGradSpec(kappa=1),
        )
    )()
    design = np.array([0.25, 0.5, 0.25])
    auxiliary_map = np.eye(3)
    auxiliary_map[2] -= variance * design[2] * design / 0.2
    auxiliary_offset = np.array([0.0, 0.0, variance * design[2] * 0.7 / 0.2])
    covariance = np.linalg.inv(
        np.linalg.inv(_prior_covariance(3))
        + np.outer(design, design) / 0.2
        + auxiliary_map.T @ auxiliary_map / variance
    )
    mean = covariance @ (
        design * 0.7 / 0.2
        + auxiliary_map.T @ (np.asarray(auxiliary[:, 0]) - auxiliary_offset) / variance
    )
    values = jnp.array([[0.25], [-0.1], [0.3]])
    carried = memory.path(values)
    for index in range(3):
        local = jax.tree.map(lambda value, index=index: value[index], geometry)
        if index:
            prior_mean = target.transition_moments(context, values[index - 1], jnp.asarray(index))[
                0
            ]
            center = local.center(prior_mean, mask[index]) + local.parent_map @ carried[index - 1]
            gain = np.linalg.solve(covariance[:index, :index], covariance[:index, index])
            expected_mean = mean[index] + gain @ (np.asarray(values[:index, 0]) - mean[:index])
            expected_variance = covariance[index, index] - gain @ covariance[:index, index]
        else:
            center = local.center(target.initial_moments(context)[0], mask[0])
            expected_mean, expected_variance = mean[0], covariance[0, 0]
        np.testing.assert_allclose(center, expected_mean, atol=3e-5)
        np.testing.assert_allclose(local.chol[0, 0] ** 2, expected_variance, atol=3e-5)
        np.testing.assert_allclose(
            local.log_density(values[index] - center, mask[index]),
            norm.logpdf(float(values[index, 0]), expected_mean, np.sqrt(expected_variance)),
            atol=3e-5,
        )
    limits = guided_delta_limit(target, context, values, mask)
    np.testing.assert_allclose(limits[:, 0], [10.0, 10.0, 6.4], atol=2e-5)


@pytest.mark.parametrize(
    ("spec_type", "kappa"),
    [(TwistedParticleAGradSpec, 1), (TwistedParticleAGradSpec, 0), (ParticleAGradSpec, 1)],
)
def test_guided_windows_preserve_joint_path_and_label_posterior(spec_type, kappa):
    times = np.arange(4.0)
    observed = np.array([[0.2, np.nan], [-0.15, -0.1], [0.4, 0.25], [-0.1, 0.35]])
    starts = np.column_stack((times, [np.nan, 0.0, 0.5, 1.5]))
    target = interval_target(times, observed, starts, ("last", "mean"))
    # Explicit trapezoids, independently integrating the two partial boundary segments.
    design = np.vstack(
        (np.eye(4), [[0.5, 0.5, 0, 0], [1 / 12, 7 / 12, 1 / 3, 0], [0, 1 / 12, 7 / 12, 1 / 3]])
    )
    readings = np.r_[observed[:, 0], observed[1:, 1]]
    prior = _prior_covariance(4)
    predictive = design @ prior @ design.T + np.eye(7) * 0.2
    gain = np.linalg.solve(predictive, design @ prior).T
    covariance = prior - gain @ design @ prior
    labels = np.array([-0.3, 0.5])
    means = np.stack([gain @ (readings - label) for label in labels])
    evidence = np.array(
        [
            norm.logpdf(label) + multivariate_normal.logpdf(readings - label, cov=predictive)
            for label in labels
        ]
    )
    probabilities = np.exp(evidence - logsumexp(evidence))
    expected_mean = probabilities @ means
    expected_second = np.einsum(
        "k,kij->ij", probabilities, covariance[None] + np.einsum("ki,kj->kij", means, means)
    )
    rng = np.random.default_rng(194)
    selected = rng.choice(2, 4096, p=probabilities)
    starts = jnp.asarray(
        (means[selected] + rng.normal(size=(4096, 4)) @ np.linalg.cholesky(covariance).T)[..., None]
    )
    spec = spec_type(num_particles=8, kappa=kappa)
    setup = build_smoother_setup(
        target,
        spec,
        num_parameter_particles=2,
        free_mask=target.independent_mask,
        paid_mix=None,
        diagnostic_metrics=("particle_weights",),
    )
    ctx = build_smoother_context(
        setup,
        jnp.asarray(labels[:, None]),
        jnp.zeros(2),
        reference=starts[0],
        delta=jnp.full(4, 0.3),
        trajectory_dtype=starts.dtype,
    )
    guide = target.context(jnp.asarray([labels.mean()]), target.times)

    def advance(path, key):
        return guided_step(target, ctx, guide, spec, key, path).latent_path

    @jax.jit
    def run(paths):
        def sweep(current, key):
            return jax.vmap(advance)(current, jax.random.split(key, len(current))), None

        return jax.lax.scan(sweep, paths, jax.random.split(jax.random.key(321), 3))[0]

    paths = run(starts)
    draws = np.asarray(paths[..., 0])
    np.testing.assert_allclose(draws.mean(0), expected_mean, atol=0.035)
    np.testing.assert_allclose(draws.T @ draws / len(draws), expected_second, atol=0.035)
    actual_labels = jax.jit(jax.vmap(lambda path: jnp.exp(ctx.score_labels(path).log_probs)))(paths)
    np.testing.assert_allclose(np.asarray(actual_labels).mean(0), probabilities, atol=0.025)
    assert np.min(np.mean(draws != np.asarray(starts[..., 0]), axis=0)) > 0.8
    traced = jax.jit(
        lambda path: trace_guided_step(target, ctx, guide, spec, jax.random.key(88), path)
    )(starts[0])
    for actual, expected in zip(
        traced.result.label_scores, ctx.score_labels(traced.result.latent_path), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=3e-5)
    np.testing.assert_allclose(
        traced.result.latent_path, jax.jit(advance)(starts[0], jax.random.key(88)), atol=1e-7
    )


def test_window_backward_matches_full_nonlinear_augmented_density_ratios():
    times = np.arange(4.0)
    readings = np.array([[0.2, np.nan], [-0.1, np.nan], [0.3, 3.0], [0.0, 1.0]])
    starts = np.column_stack((times, [np.nan, np.nan, 0.5, 2.0]))
    base = interval_target(
        times,
        readings,
        starts,
        ("last", "sum"),
        families=(DistributionFamily.GAUSSIAN, DistributionFamily.NEGATIVE_BINOMIAL),
    )

    def declared(context):
        evolution = dsx.GaussianStateEvolution(
            lambda x, _u, _t0, _t1: 0.65 * x - 0.3 * jnp.tanh(x) ** 3,
            lambda _x, _u, _t0, _t1: jnp.eye(1) * 0.3,
        )
        return eqx.tree_at(
            lambda model: model.state_evolution, base.dynamical_model(context), evolution
        )

    target = replace(base, dynamical_model=declared)
    labels = np.array([-0.3, 0.5])
    log_labels = norm.logpdf(labels)
    log_labels -= logsumexp(log_labels)
    contexts = jax.vmap(lambda label: target.context(label, target.times))(
        jnp.asarray(labels[:, None])
    )
    memory = sequential_observations(target, contexts, target.context(jnp.zeros(1), target.times))
    paths = np.array([[0.2, -0.3, 0.5, -0.2], [-0.2, 0.4, -0.1, 0.3], [0.6, 0.1, -0.4, 0.2]])
    design = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0.125, 0.875, 0.5, 0], [0, 0, 0.5, 0.5]])
    auxiliary = np.array([0.1, 0.3, -0.1, 0.2])
    variance = 0.3

    def factors(path):
        observed, derivatives, densities = [], [], []
        for label in labels:
            point = norm.logpdf(readings[:, 0], path + label, np.sqrt(0.2))
            derivative = (readings[:, 0] - path - label) / 0.2
            response = np.exp(path + label)
            means = design @ response
            point[2:] += nbinom.logpmf(readings[2:, 1], 5, 5 / (5 + means[2:]))
            derivative[2:] += (
                (readings[2:, 1] / means[2:] - (5 + readings[2:, 1]) / (5 + means[2:]))
                * np.diag(design)[2:]
                * response[2:]
            )
            prior = np.r_[
                norm.logpdf(path[0]),
                norm.logpdf(
                    path[1:], 0.65 * path[:-1] - 0.3 * np.tanh(path[:-1]) ** 3, np.sqrt(0.3)
                ),
            ]
            observed.append(point)
            derivatives.append(derivative)
            densities.append(prior + point)
        observed, derivatives = np.asarray(observed).T, np.asarray(derivatives).T
        weights = np.exp(
            observed + log_labels - logsumexp(observed + log_labels, axis=1, keepdims=True)
        )
        score = np.sum(weights * derivatives, axis=1)
        auxiliary_density = norm.logpdf(auxiliary, path + variance * score, np.sqrt(variance))
        return np.asarray(densities).T, auxiliary_density

    prefix = np.stack([np.cumsum(factors(path)[0], axis=0) + log_labels for path in paths], axis=1)
    scores = jnp.asarray(prefix - logsumexp(prefix, axis=-1, keepdims=True))
    bank = jnp.swapaxes(jax.vmap(memory.path)(jnp.asarray(paths[..., None])), 0, 1)
    weights = jnp.asarray([[0.2, -0.3, 0.1], [-0.1, 0.4, -0.2], [0.3, -0.1, 0.2], [0, 0, 0]])

    def observe(value, index):
        def mixture(current):
            probabilities = memory.log_probs(memory.replace_current(value, current, index), index)
            return jax.scipy.special.logsumexp(jnp.asarray(log_labels) + probabilities)

        return memory.log_probs(value, index), jax.grad(mixture)(memory.state(value))

    def transition(previous, current, index):
        return jax.vmap(
            lambda context: target.transition_log_prob(
                context, memory.state(previous), memory.state(current), index
            )
        )(contexts)

    keys = jax.random.split(jax.random.key(195), 32)
    actual = jax.jit(
        jax.vmap(
            lambda key: _window_backward(
                memory,
                bank,
                weights,
                scores,
                jnp.asarray(1),
                key,
                jnp.asarray(auxiliary[:, None]),
                jnp.full((4, 1), variance),
                jnp.ones((4, 1), bool),
                observe,
                transition,
            )
        )
    )(keys)
    expected = []
    for key in keys:
        selected = paths[1].copy()
        picks = np.full(4, 1)
        noise = np.asarray(jax.vmap(lambda k: jax.random.gumbel(k, (3,)))(jax.random.split(key, 3)))
        for offset, index in enumerate(range(2, -1, -1)):
            logits = []
            for particle in range(3):
                candidate = np.r_[paths[particle, : index + 1], selected[index + 1 :]]
                density, auxiliary_density = factors(candidate)
                logits.append(
                    float(weights[index, particle])
                    + logsumexp(log_labels + density.sum(0))
                    - logsumexp(prefix[index, particle])
                    + auxiliary_density[index + 1 :].sum()
                )
            pick = int(np.argmax(np.asarray(logits) + noise[offset]))
            picks[index] = pick
            selected[index] = paths[pick, index]
        expected.append(picks)
    np.testing.assert_array_equal(actual, expected)
