"""Exact Jacobian, analytic invariance, center learning and continuation of scale charts."""

from dataclasses import replace

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.continuation import SamplerCheckpoint
from grassmoth.artifacts.sampling import (
    GaussianInitializationSpec,
    GuidedPathSpec,
    MarginalParticleGibbsSpec,
    SamplerSpec,
)
from grassmoth.models.ssm.inference.continuation import frozen_particle_tuning
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import run_marginal_particle_gibbs
from grassmoth.models.ssm.inference.targets.coordinates import ScaleChart, TranslationChart
from grassmoth.models.ssm.inference.warmup.initialization import (
    ParticleInitialization,
    prepare_particle_initialization,
)
from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure
from grassmoth.models.ssm.inference.warmup.scale_adaptation import (
    CenterMoments,
    adapt_center,
    initialize_center_moments,
)
from grassmoth_test_support.compute import EXECUTION_CONTEXT
from grassmoth_test_support.interval_particle import initialize_interval_chain, interval_target
from packages.grassmoth.tests.models.ssm.inference.test_translation_chart import _gaussian_joint


def _chart() -> ScaleChart:
    return ScaleChart(
        TranslationChart(jnp.array([[[-0.85]], [[-0.7]], [[-0.95]]]), jnp.array([0.3])),
        jnp.array([[0.4]]),
        jnp.array([[0.2], [-0.1], [0.3]]),
    )


@pytest.mark.inference(concern="sampling")
def test_scale_chart_pays_its_jacobian_and_expresses_moments_in_chart_units():
    times = np.arange(3.0)
    base = interval_target(times, np.array([[0.2], [-0.3], [0.5]]), times[:, None], ("last",))

    def model(context):
        return eqx.tree_at(
            lambda value: value.state_evolution,
            base.dynamical_model(context),
            dsx.GaussianStateEvolution(
                lambda x, _u, _start, _stop: 0.7 * x - 0.2 * jnp.tanh(x) ** 3,
                lambda _x, _u, _start, _stop: jnp.array([[0.3]]),
            ),
        )

    target = replace(base, dynamical_model=model)
    readings, times = target.observations, target.times
    chart = _chart()
    scaled = target.in_coordinates(chart)
    position = jnp.array([0.6])
    path = jnp.array([[0.1], [-0.5], [0.8]])
    encoded = chart.encode(position, path)
    physical_context = target.context(position, target.times)
    context = scaled.context(position, scaled.times)
    log_scale = chart.log_scale(position)
    # One scaled coordinate in the initial slice and in both transition slices.
    log_jacobian = 3 * log_scale[0]
    decode_jacobian = jax.jacfwd(lambda value: chart.decode(position, value).ravel())(encoded)
    np.testing.assert_allclose(chart.decode(position, encoded), path, atol=1e-6)
    np.testing.assert_allclose(
        jnp.linalg.slogdet(decode_jacobian.reshape(3, 3))[1], log_jacobian, atol=1e-6
    )
    np.testing.assert_allclose(
        scaled.log_posterior(position, encoded, readings, times),
        target.log_posterior(position, path, readings, times) + log_jacobian,
        rtol=2e-6,
    )
    actual_gradient = jax.grad(scaled.log_posterior, argnums=(0, 1))(
        position, encoded, readings, times
    )
    expected_gradient = jax.grad(
        lambda parameter, coordinates: (
            target.log_posterior(parameter, chart.decode(parameter, coordinates), readings, times)
            + 3 * chart.log_scale(parameter)[0]
        ),
        argnums=(0, 1),
    )(position, encoded)
    for actual, expected in zip(actual_gradient, expected_gradient, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-6)
    scale = jnp.exp(log_scale)
    noise = scaled.prepare_noise_factors(context, readings)
    for index in (0, 1, 2):
        if index == 0:
            chart_mean, chart_covariance = scaled.initial_moments(context)
            mean, covariance = target.initial_moments(physical_context)
        else:
            chart_mean, chart_covariance = scaled.transition_moments(
                context, encoded[index - 1], jnp.asarray(index)
            )
            mean, covariance = target.transition_moments(
                physical_context, path[index - 1], jnp.asarray(index)
            )
            np.testing.assert_allclose(
                scaled.transition_log_prob_with_noise(
                    context, noise, encoded[index - 1], encoded[index], index
                ),
                target.transition_log_prob(physical_context, path[index - 1], path[index], index)
                + log_scale[0],
                atol=2e-6,
            )
        np.testing.assert_allclose(chart.frame(position).decode(chart_mean, index), mean, atol=1e-6)
        np.testing.assert_allclose(
            chart_covariance * scale[:, None] * scale[None, :], covariance, rtol=1e-6
        )
    previous = jnp.array([[0.1], [-0.2]])
    current = jnp.array([[0.3], [0.0], [-0.4]])
    elementwise = jax.vmap(
        lambda ancestor: jax.vmap(
            lambda descendant: scaled.transition_log_prob(context, ancestor, descendant, 1)
        )(current)
    )(previous)
    np.testing.assert_allclose(
        scaled.pairwise_transition_log_prob(context, previous, current, 1), elementwise, atol=3e-6
    )
    value, gradient = scaled._transition_value_and_grad(
        context, previous[0], current[0], 1, current_gradient=True
    )
    np.testing.assert_allclose(value, elementwise[0, 0], atol=2e-6)
    np.testing.assert_allclose(
        gradient,
        jax.grad(lambda state: scaled.transition_log_prob(context, previous[0], state, 1))(
            current[0]
        ),
        atol=3e-6,
    )


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("method", ["particle_agrad", "twisted_particle_agrad", "particle_mgrad"])
def test_scale_charted_joint_kernel_preserves_analytic_physical_posterior(method):
    target, mean, covariance = _gaussian_joint()
    chart = _chart()
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        path_update=GuidedPathSpec(method=method, num_particles=8, kappa=1),
        num_parameter_particles=2,
        param_step_size=0.3,
        initial_chart=chart,
    )
    joint = jnp.asarray(np.random.default_rng(819).multivariate_normal(mean, covariance, 4096))
    initial = jax.vmap(lambda draw: initialize_interval_chain(target, draw, kernel=kernel))(joint)
    initial = initial._replace(
        chart=jax.tree.map(lambda value: jnp.broadcast_to(value, (len(joint), *value.shape)), chart)
    )

    @jax.jit
    def run(states):
        def advance(current, key):
            following, _ = jax.vmap(kernel.step_fn)(current, jax.random.split(key, len(joint)))
            return following, None

        return jax.lax.scan(advance, states, jax.random.split(jax.random.key(918), 4))[0]

    final = run(initial)
    draws = np.concatenate(
        (np.asarray(final.position), np.asarray(final.latent_trajectory)[..., 0]), axis=1
    )
    np.testing.assert_allclose(draws.mean(axis=0), mean, atol=0.04)
    np.testing.assert_allclose(np.cov(draws.T), covariance, atol=0.04)
    assert np.mean(draws[:, 0] != np.asarray(joint)[:, 0]) > 0.2
    exact = jax.vmap(
        lambda position, path: target.log_posterior(
            position, path, target.observations, target.times
        )
    )(final.position, final.latent_trajectory)
    np.testing.assert_allclose(final.complete_log_posterior, exact, atol=3e-5)


@pytest.mark.inference(concern="warmup")
def test_center_adaptation_freezes_each_window_mean_translated_path():
    chart = ScaleChart(
        TranslationChart(jnp.broadcast_to(jnp.array([[-1.0], [0.5]]), (3, 2, 1)), jnp.array([0.2])),
        jnp.ones((2, 1)),
        jnp.zeros((3, 2)),
    )
    keys = jax.random.split(jax.random.key(7), 7)
    positions = jax.vmap(lambda key: jax.random.normal(key, (4, 1)))(keys)
    paths = jax.vmap(lambda key: jax.random.normal(jax.random.fold_in(key, 1), (4, 3, 2)))(keys)
    windows = jnp.array([[1, 2], [3, 5]])

    def advance(state: tuple[CenterMoments, ScaleChart], index: jax.Array):
        moments, current = adapt_center(*state, positions[index], paths[index], index, windows)
        return (moments, current), current.center

    (moments, learned), centers = jax.lax.scan(
        advance, (initialize_center_moments(positions[0], paths[0]), chart), jnp.arange(7)
    )
    translated = jax.vmap(jax.vmap(chart.translation.encode))(positions, paths)
    window_means = translated.mean(axis=1)
    np.testing.assert_array_equal(centers[:2], 0.0)
    np.testing.assert_allclose(centers[2], window_means[1:3].mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(centers[3:5], np.broadcast_to(centers[2], (2, 3, 2)))
    np.testing.assert_allclose(centers[5], window_means[3:6].mean(axis=0), atol=1e-6)
    np.testing.assert_array_equal(centers[6], centers[5])
    assert int(moments.sample_count) == 0
    np.testing.assert_array_equal(learned.exponents, chart.exponents)


@pytest.mark.inference(concern="warmup")
@pytest.mark.inference(concern="sampling")
def test_scale_chart_checkpoint_continuation_restores_its_learned_center():
    target, _, _ = _gaussian_joint()
    initialization = prepare_particle_initialization(
        target,
        spec=GaussianInitializationSpec(),
        key=jax.random.key(22),
        num_chains=1,
        execution_context=EXECUTION_CONTEXT,
    )
    assert isinstance(initialization, ParticleInitialization)
    chart = _chart()
    spec = GuidedPathSpec(method="particle_mgrad", num_particles=8, kappa=1)
    kernel = build_marginal_particle_gibbs_kernel(
        target, path_update=spec, num_parameter_particles=2, initial_chart=chart
    )
    options = MarginalParticleGibbsSpec(path_update=spec, scale_chart=True)

    def run(warmup, draws, continuation=None):
        # A resumed run must take the saved chart, never a rebuilt template.
        resumed_kernel = (
            kernel
            if continuation is None
            else kernel._replace(
                initial_chart=ScaleChart(
                    chart.translation, chart.exponents + 2.0, chart.center + 5.0
                )
            )
        )
        result = run_marginal_particle_gibbs(
            target,
            kernel=resumed_kernel,
            sampler=SamplerSpec(
                algorithm=options, num_warmup=warmup, num_samples_per_chain=draws, num_chains=1
            ),
            key=jax.random.key(31),
            initialization=initialization,
            execution_context=EXECUTION_CONTEXT,
            continuation=continuation,
        )
        assert not isinstance(result, InitializationFailure)
        return result

    full = run(120, 4)
    prefix = run(120, 2)
    assert isinstance(full.final_chart, ScaleChart)
    assert not np.allclose(full.final_chart.center, chart.center)
    checkpoint = SamplerCheckpoint(
        settings=SamplerSpec(algorithm=options),
        parameters=NumericalArray.from_numpy(np.asarray(prefix.last_parameters)),
        paths=NumericalArray.from_numpy(np.asarray(prefix.last_paths)),
        random_key=NumericalArray.from_numpy(np.asarray(jax.random.key_data(prefix.random_key))),
        native_state=prefix.final_state,
        tuning=frozen_particle_tuning(initialization, None),
    )
    restored = SamplerCheckpoint.model_validate(checkpoint.model_dump())
    continued = run(0, 2, restored)
    np.testing.assert_array_equal(full.latent_paths[:, 2:], continued.latent_paths)
    np.testing.assert_array_equal(full.grouped_positions[:, 2:], continued.grouped_positions)
    for actual, expected in zip(continued.final_state, full.final_state, strict=True):
        np.testing.assert_array_equal(actual.values, expected.values)
