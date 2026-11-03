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
    AuxiliaryKalmanSpec,
    GaussianInitializationSpec,
    GuidedPathSpec,
    SamplerSpec,
)
from grassmoth.models.ssm.inference.continuation import frozen_particle_tuning
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import run_marginal_particle_gibbs
from grassmoth.models.ssm.inference.targets.coordinates import PathChart
from grassmoth.models.ssm.inference.warmup.computed_chart import computed_chart
from grassmoth.models.ssm.inference.warmup.initialization import (
    ParticleInitialization,
    prepare_particle_initialization,
)
from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure
from grassmoth_test_support.compute import EXECUTION_CONTEXT
from grassmoth_test_support.interval_particle import (
    computed_translation_slopes,
    initialize_interval_chain,
    interval_target,
)
from packages.grassmoth.tests.models.ssm.inference.test_joint_coordinates import _direct_constraints
from packages.grassmoth.tests.models.ssm.inference.test_translation_chart import _gaussian_joint


def _chart() -> PathChart:
    return PathChart(
        jnp.array([[[-0.85]], [[-0.7]], [[-0.95]]]),
        jnp.array([0.3]),
        jnp.array([[[0.4]], [[-0.2]], [[0.65]]]),
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
    log_jacobian = jnp.sum(log_scale)
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
            + jnp.sum(chart.log_scale(parameter))
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
                + log_scale[index, 0],
                atol=2e-6,
            )
        np.testing.assert_allclose(chart.frame(position).decode(chart_mean, index), mean, atol=1e-6)
        np.testing.assert_allclose(
            chart_covariance * scale[index, :, None] * scale[index, None, :], covariance, rtol=1e-6
        )


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(
    "method", ["twisted_particle_agrad", "particle_mgrad", "kalman_full", "kalman_blocked"]
)
@pytest.mark.parametrize("pinned", [False, True])
def test_scale_charted_joint_kernel_preserves_analytic_physical_posterior(method, pinned):
    target, mean, covariance = _gaussian_joint()
    if pinned:
        gain = covariance[:, 2] / covariance[2, 2]
        mean = mean + gain * (0.25 - mean[2])
        covariance = covariance - gain[:, None] * covariance[2]
        target = replace(
            target,
            exact_constraints=_direct_constraints(0.25, steps=3, state_names=("x",)),
        )
    chart = replace(
        _chart(),
        slopes=computed_translation_slopes(target, jnp.array([0.3]), jnp.zeros((3, 1))),
    )
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        path_update=(
            AuxiliaryKalmanSpec(block_length=2 if method == "kalman_blocked" else None)
            if method.startswith("kalman")
            else GuidedPathSpec(method=method, num_particles=8, kappa=1)
        ),
        num_parameter_particles=2,
        param_step_size=0.3,
        initial_chart=chart,
        exact_constraints=target.exact_constraints,
    )
    joint = jnp.asarray(np.random.default_rng(819).multivariate_normal(mean, covariance, 4096))
    if pinned:
        joint = joint.at[:, 2].set(0.25)
    initial = jax.vmap(lambda draw: initialize_interval_chain(target, draw, kernel=kernel))(joint)
    initial = initial._replace(
        chart=jax.tree.map(
            lambda value: jnp.broadcast_to(value, (len(joint), *value.shape)), kernel.initial_chart
        )
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
    options = SamplerSpec(path_update=spec, compiler_reparameterization=True)
    chart_builder = EXECUTION_CONTEXT.compiler(
        lambda positions, paths: computed_chart(target, positions, paths),
        initialization.positions,
        initialization.paths,
    ).call

    def run(warmup, draws, continuation=None):
        # A resumed run must take the saved chart, never a rebuilt template.
        resumed_kernel = (
            kernel
            if continuation is None
            else kernel._replace(
                initial_chart=replace(
                    chart, exponents=chart.exponents + 2.0, center=chart.center + 5.0
                )
            )
        )
        result = run_marginal_particle_gibbs(
            target,
            kernel=resumed_kernel,
            sampler=options.revised(num_warmup=warmup, num_samples_per_chain=draws, num_chains=1),
            key=jax.random.key(31),
            initialization=initialization,
            execution_context=EXECUTION_CONTEXT,
            continuation=continuation,
            chart_builder=chart_builder if warmup else None,
        )
        assert not isinstance(result, InitializationFailure)
        return result

    full = run(120, 4)
    prefix = run(120, 2)
    assert isinstance(full.final_chart, PathChart)
    assert not np.allclose(full.final_chart.center, chart.center)
    checkpoint = SamplerCheckpoint(
        settings=options,
        parameters=NumericalArray.from_numpy(np.asarray(prefix.last_parameters)),
        paths=NumericalArray.from_numpy(np.asarray(prefix.last_paths)),
        random_key=NumericalArray.from_numpy(np.asarray(jax.random.key_data(prefix.random_key))),
        native_state=prefix.final_state,
        tuning=frozen_particle_tuning(initialization),
    )
    restored = SamplerCheckpoint.model_validate(checkpoint.model_dump())
    continued = run(0, 2, restored)
    np.testing.assert_array_equal(full.latent_paths[:, 2:], continued.latent_paths)
    np.testing.assert_array_equal(full.grouped_positions[:, 2:], continued.grouped_positions)
    for actual, expected in zip(continued.final_state, full.final_state, strict=True):
        np.testing.assert_array_equal(actual.values, expected.values)


@pytest.mark.inference(concern="sampling")
def test_float32_distant_scale_labels_have_zero_mass_without_saturating_the_map():
    from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import _initialize_chain_state

    target, _, _ = _gaussian_joint()
    chart = replace(_chart(), exponents=jnp.full((3, 1, 1), 200.0))
    assert bool(chart.representable(chart.reference))
    assert not bool(chart.representable(chart.reference + 1.0))
    assert not bool(chart.representable(chart.reference - 1.0))
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_parameter_particles=2,
        path_update=GuidedPathSpec(method="particle_mgrad", num_particles=4, kappa=1),
        initial_chart=chart,
    )
    state = _initialize_chain_state(
        chart.reference,
        observations=target.observations,
        times=target.times,
        target=target,
        parameter_gradient_fn=kernel.parameter_gradient_fn,
        parameter_chol=kernel.initial_parameter_chol,
        initial_latent_delta=jnp.full((3,), 0.3),
        param_step_size=1e3,
        param_min_scale=1e-6,
        param_max_scale=1e3,
        initial_latent_trajectory=jnp.zeros((3, 1)),
        chart=chart,
    )
    updated, info = jax.jit(kernel.step_fn)(state, jax.random.key(382))
    np.testing.assert_array_equal(updated.position, state.position)
    assert np.isfinite(updated.complete_log_posterior)
    assert np.isfinite(updated.latent_trajectory).all()
    np.testing.assert_array_equal(info["final_label_log_probs"], [0.0, -np.inf])


@pytest.mark.inference(concern="warmup")
@pytest.mark.parametrize("old_shape", ["translation", "scale"])
def test_old_native_chart_shapes_return_expected_failure(old_shape):
    from grassmoth.models.ssm.inference.continuation import restore_state, retain_state

    chart = _chart()
    old = (
        retain_state((chart.slopes, chart.reference))
        if old_shape == "translation"
        else retain_state((chart.slopes, chart.reference, chart.exponents[0], chart.center))
    )
    outcome = restore_state(chart, old)
    assert isinstance(outcome, InitializationFailure)
    assert "incompatible" in outcome.message
