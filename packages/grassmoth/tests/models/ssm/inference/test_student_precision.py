"""Independent density and posterior checks for the collapsed StudentT augmentation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import gamma, multivariate_normal, norm, t

from grassmoth.artifacts.model.likelihood import StudentTLawSpec
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.artifacts.sampling import TwistedParticleAGradSpec
from grassmoth.models.ssm.execution.observation_model import precision_normal_log_prob
from grassmoth.models.ssm.inference.samplers.particle.context import build_smoother_context
from grassmoth.models.ssm.inference.samplers.particle.guided import trace_guided_step
from grassmoth.models.ssm.inference.samplers.particle.setup import build_smoother_setup
from grassmoth.models.ssm.inference.targets.sequential import sequential_observations
from grassmoth.models.ssm.inference.targets.student_precision import (
    draw_student_precisions,
)
from grassmoth_test_support.interval_particle import interval_target

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize("df", [0.15, 1.5, 5.0])
def test_gamma_normal_integrates_to_native_student(df):
    scale, loc, y = 0.7, -0.3, 2.1
    law = StudentTLawSpec(df=jnp.asarray(df), loc=jnp.asarray(loc), scale=jnp.asarray(scale))
    evaluate = jax.jit(
        lambda log_precision: precision_normal_log_prob(
            law, jnp.asarray(y), jnp.asarray(True), log_precision
        )
    )
    shape = df / 2

    def integrand(log_precision):
        log_gamma = (
            shape * np.log(shape)
            - gammaln(shape)
            + (shape - 1) * log_precision
            - shape * np.exp(log_precision)
        )
        return np.exp(float(evaluate(jnp.asarray(log_precision))) + log_gamma + log_precision)

    integral = quad(integrand, -1000, 12, points=[-100, -20, 0], epsabs=1e-10)[0]
    np.testing.assert_allclose(integral, t.pdf(y, df, loc, scale), rtol=2e-6)


@pytest.mark.parametrize("summary", ["last", "mean", "std"])
def test_emitted_precision_moments_and_exact_conditional_factors(summary):
    times = np.arange(4.0)
    readings = np.array([[0.1], [np.nan], [0.4], [-0.2]])
    starts = np.array([[0.0], [np.nan], [0.0], [1.0]])
    target = interval_target(
        times, readings, starts, (summary,), families=(DistributionFamily.STUDENT_T,)
    )
    path = jnp.array([[0.2], [-0.4], [0.7], [1.1]])
    context = target.context(jnp.array([0.3]), target.times)
    responses = np.asarray(path[:, 0]) + 0.3
    means = responses.copy()
    if summary != "last":
        for end, start in [(2, 0), (3, 1)]:
            weights = np.array([0.25, 0.5, 0.25])
            values = responses[start : end + 1]
            mean = weights @ values
            means[end] = mean if summary == "mean" else np.sqrt(weights @ values**2 - mean**2)
    present = np.isfinite(readings[:, 0])
    shapes = 3.0
    rates = (5 + (readings[present, 0] - means[present]) ** 2 / 0.2) / 2

    draws = jax.jit(jax.vmap(lambda key: draw_student_precisions(target, context, path, key)))(
        jax.random.split(jax.random.key(74), 32768)
    )
    values = np.exp(np.asarray(draws.log_values[:, present, 0]))
    np.testing.assert_allclose(values.mean(axis=0), shapes / rates, rtol=0.02)
    np.testing.assert_allclose(values.var(axis=0), shapes / rates**2, rtol=0.04)
    assert np.all(np.asarray(draws.log_values[:, ~present, 0]) == 0)
    precision = jax.tree.map(lambda value: value[0], draws)
    expected_prior = gamma.logpdf(
        np.exp(np.asarray(precision.log_values[present, 0])), 2.5, scale=1 / 2.5
    ).sum()
    np.testing.assert_allclose(precision.log_prior(target, context), expected_prior, atol=5e-6)
    contexts = jax.tree.map(lambda value: value[None], context)
    memory = sequential_observations(target, contexts, context, precision.log_values)
    observed = jax.vmap(memory.log_probs)(memory.path(path), jnp.arange(4))[:, 0]
    expected = norm.logpdf(
        readings[present, 0],
        means[present],
        np.sqrt(0.2) * np.exp(-0.5 * np.asarray(precision.log_values[present, 0])),
    )
    np.testing.assert_allclose(np.asarray(observed)[present], expected, atol=5e-6)
    assert np.all(np.asarray(observed)[~present] == 0)


def test_extreme_residual_uses_log_precision_without_clipping():
    # The squared residual overflows float32 and the precision, near 1e-60,
    # underflows it, so both must stay in log space.
    target = interval_target(
        np.arange(1.0),
        np.array([[1e30]]),
        np.zeros((1, 1)),
        ("last",),
        families=(DistributionFamily.STUDENT_T,),
    )
    context = target.context(jnp.zeros(1), target.times)
    precision = draw_student_precisions(target, context, jnp.zeros((1, 1)), jax.random.key(8))
    assert precision is not None
    assert float(precision.log_values[0, 0]) < -130
    contexts = jax.tree.map(lambda value: value[None], context)
    memory = sequential_observations(target, contexts, context, precision.log_values)
    assert np.isfinite(memory.log_probs(memory.initial(jnp.zeros(1)), jnp.asarray(0))).all()
    assert np.isfinite(precision.log_prior(target, context))


def test_conditional_curvature_is_label_symmetric_and_ignores_missing_rows():
    target = interval_target(
        np.arange(3.0),
        np.array([[0.2], [np.nan], [-0.5]]),
        np.arange(3.0)[:, None],
        ("last",),
        families=(DistributionFamily.STUDENT_T,),
    )
    contexts = jax.vmap(lambda z: target.context(z, target.times))(jnp.array([[0.1], [-0.4]]))
    context = target.context(jnp.array([0.1]), target.times)
    precision = draw_student_precisions(target, context, jnp.zeros((3, 1)), jax.random.key(17))
    assert precision is not None
    limit = precision.step_limit(target, contexts, target.independent_mask, 10.0)
    conditional_information = np.where(
        np.asarray(precision.observed[:, 0]),
        np.exp(np.asarray(precision.log_values[:, 0])) / 0.2,
        0.0,
    )
    np.testing.assert_allclose(limit[:, 0], 2 / np.maximum(conditional_information, 0.2), rtol=1e-6)
    assert float(limit[1, 0]) == 10.0
    permuted = jax.tree.map(lambda value: value[::-1], contexts)
    np.testing.assert_array_equal(
        limit, precision.step_limit(target, permuted, target.independent_mask, 10.0)
    )


def test_augmented_student_channels_preserve_other_correlated_gaussian_blocks():
    readings = np.array([[0.4, -0.2, 0.9], [np.nan, 0.6, np.nan]])
    covariance = np.array([[0.2, 0.1, 0.1], [0.1, 0.4, 0.15], [0.1, 0.15, 0.3]])
    target = interval_target(
        np.arange(2.0),
        readings,
        np.broadcast_to(np.arange(2.0)[:, None], readings.shape),
        ("last",) * 3,
        families=(
            DistributionFamily.STUDENT_T,
            DistributionFamily.GAUSSIAN,
            DistributionFamily.GAUSSIAN,
        ),
        covariance=covariance,
    )
    context = target.context(jnp.array([0.2]), target.times)
    path = jnp.array([[0.1], [-0.3]])
    precision = draw_student_precisions(target, context, path, jax.random.key(39))
    assert precision is not None
    np.testing.assert_array_equal(precision.observed, [[True, False, False], [False, False, False]])
    contexts = jax.tree.map(lambda value: value[None], context)
    memory = sequential_observations(target, contexts, context, precision.log_values)
    observed = jax.vmap(memory.log_probs)(memory.path(path), jnp.arange(2))[:, 0]
    expected = [
        norm.logpdf(0.4, 0.3, np.sqrt(0.2) * np.exp(-0.5 * float(precision.log_values[0, 0])))
        + multivariate_normal.logpdf(readings[0, 1:], np.full(2, 0.3), covariance[1:, 1:]),
        norm.logpdf(0.6, -0.1, np.sqrt(0.4)),
    ]
    np.testing.assert_allclose(observed, expected, atol=1e-6)


@pytest.mark.parametrize("summary", ["last", "mean", "std"])
def test_conditional_path_scores_and_guides_preserve_label_permutation(summary):
    times = np.arange(4.0)
    readings = np.array([[0.1], [np.nan], [0.4], [-0.2]])
    target = interval_target(
        times,
        readings,
        np.array([[0.0], [np.nan], [0.0], [1.0]]),
        (summary,),
        families=(DistributionFamily.STUDENT_T,),
    )
    reference = jnp.array([[0.2], [-0.4], [0.7], [1.1]])
    labels = jnp.array([[-0.4], [0.2], [0.6]])
    context = target.context(labels[0], target.times)
    precision = draw_student_precisions(target, context, reference, jax.random.key(89))
    assert precision is not None
    spec = TwistedParticleAGradSpec(num_particles=8, kappa=1)
    setup = build_smoother_setup(
        target,
        spec,
        num_parameter_particles=3,
        free_mask=target.independent_mask,
        paid_mix=None,
        diagnostic_metrics=(),
    )._replace(
        trajectory_log_prob_fn=lambda context, path, _: precision.path_log_prob(
            target, context, path
        )
    )

    def run(candidates):
        bound = build_smoother_context(
            setup,
            candidates,
            jnp.zeros(3),
            reference=reference,
            delta=jnp.full((4,), 0.4),
            trajectory_dtype=reference.dtype,
            label_factor=lambda context: precision.log_prior(target, context),
        )
        guide = target.context(bound.guide_position, target.times)
        traced = trace_guided_step(
            target,
            bound,
            guide,
            spec,
            jax.random.key(9),
            reference,
            log_precisions=precision.log_values,
        )
        return traced.result, bound.score_labels(traced.result.latent_path)

    result, exact = jax.jit(run)(labels)
    permuted, _ = jax.jit(run)(labels[::-1])
    np.testing.assert_allclose(result.latent_path, permuted.latent_path, atol=1e-6)
    for actual, expected in zip(result.label_scores, exact, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-5)
    np.testing.assert_allclose(
        result.label_scores.log_probs[::-1], permuted.label_scores.log_probs, atol=1e-5
    )
    path = np.asarray(result.latent_path[:, 0])
    prior = norm.logpdf(path[0]) + norm.logpdf(path[1:], 0.7 * path[:-1], np.sqrt(0.3)).sum()
    expected = []
    present = np.isfinite(readings[:, 0])
    for label in np.asarray(labels[:, 0]):
        response = path + label
        means = response.copy()
        if summary != "last":
            for end, start in [(2, 0), (3, 1)]:
                values = response[start : end + 1]
                mean = np.array([0.25, 0.5, 0.25]) @ values
                means[end] = (
                    mean
                    if summary == "mean"
                    else np.sqrt(np.array([0.25, 0.5, 0.25]) @ values**2 - mean**2)
                )
        expected.append(
            prior
            + norm.logpdf(
                readings[present, 0],
                means[present],
                np.sqrt(0.2) * np.exp(-0.5 * np.asarray(precision.log_values[present, 0])),
            ).sum()
        )
    np.testing.assert_allclose(exact.trajectory_log_probs, expected, atol=1e-5)
