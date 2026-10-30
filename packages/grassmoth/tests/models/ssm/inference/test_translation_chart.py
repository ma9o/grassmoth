"""Physical density, analytic invariance and continuation of particle path charts."""

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
    DsmcSpec,
    GaussianInitializationSpec,
    GuidedPathSpec,
    MarginalParticleGibbsSpec,
    SamplerSpec,
)
from grassmoth.models.ssm.inference.continuation import frozen_particle_tuning
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import (
    _initialize_chain_state,
    run_marginal_particle_gibbs,
)
from grassmoth.models.ssm.inference.samplers.particle.contract import PaidMixture
from grassmoth.models.ssm.inference.targets.coordinates import TranslationChart
from grassmoth.models.ssm.inference.targets.sequential import sequential_observations
from grassmoth.models.ssm.inference.warmup.computed_chart import (
    computed_chart,
    computed_translation_slopes,
)
from grassmoth.models.ssm.inference.warmup.initialization import (
    ParticleInitialization,
    prepare_particle_initialization,
)
from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure
from grassmoth_test_support.compute import EXECUTION_CONTEXT
from grassmoth_test_support.interval_particle import initialize_interval_chain, interval_target
from packages.grassmoth.tests.models.ssm.inference.test_joint_coordinates import _direct_constraints


def _gaussian_joint():
    """Condition a dense joint Gaussian independently of the sampler's factors."""
    times = np.arange(3.0)
    readings = np.array([0.15, -0.25, 0.45])
    target = interval_target(times, readings[:, None], times[:, None], ("last",))
    recurrence = np.eye(3)
    for index in range(1, 3):
        recurrence[index, :index] = 0.7 * recurrence[index - 1, :index]
    prior = np.zeros((4, 4))
    prior[0, 0] = 1
    prior[1:, 1:] = recurrence @ np.diag([1.0, 0.3, 0.3]) @ recurrence.T
    design = np.column_stack((np.ones(3), np.eye(3)))
    observed = design @ prior @ design.T + 0.2 * np.eye(3)
    gain = np.linalg.solve(observed, design @ prior).T
    return target, gain @ readings, prior - gain @ design @ prior


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("interval", [False, True])
def test_translation_preserves_nonlinear_factors_gradients_and_proposal_moments(interval):
    times = np.arange(3.0)
    readings = np.array([[0.2], [-0.3], [0.5]])
    starts = np.array([[np.nan], [0.0], [0.5]]) if interval else times[:, None]
    if interval:
        readings[0] = np.nan
    base = interval_target(times, readings, starts, ("mean" if interval else "last",))

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
    chart = TranslationChart(
        computed_translation_slopes(target, jnp.array([0.3]), jnp.zeros((3, 1))), jnp.array([0.3])
    )
    translated = target.in_coordinates(chart)
    position = jnp.array([0.6])
    path = jnp.array([[0.1], [-0.5], [0.8]])
    encoded = chart.encode(position, path)
    physical_context = target.context(position, target.times)
    context = translated.context(position, translated.times)
    np.testing.assert_allclose(chart.decode(position, encoded), path, atol=1e-7)
    np.testing.assert_allclose(
        translated.log_posterior(position, encoded, readings, times),
        target.log_posterior(position, path, readings, times),
        rtol=2e-6,
    )
    actual_gradient = jax.grad(translated.log_posterior, argnums=(0, 1))(
        position, encoded, readings, times
    )
    expected_gradient = jax.grad(
        lambda parameter, coordinates: target.log_posterior(
            parameter, chart.decode(parameter, coordinates), readings, times
        ),
        argnums=(0, 1),
    )(position, encoded)
    for actual, expected in zip(actual_gradient, expected_gradient, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-6)
    for index in (0, 1, 2):
        if index == 0:
            shifted_mean, shifted_cov = translated.initial_moments(context)
            mean, covariance = target.initial_moments(physical_context)
        else:
            shifted_mean, shifted_cov = translated.transition_moments(
                context, encoded[index - 1], jnp.asarray(index)
            )
            mean, covariance = target.transition_moments(
                physical_context, path[index - 1], jnp.asarray(index)
            )
            noise = translated.prepare_noise_factors(context, readings)
            expected = target.transition_log_prob(
                physical_context, path[index - 1], path[index], index
            )
            actual = translated.transition_log_prob_with_noise(
                context, noise, encoded[index - 1], encoded[index], index
            )
            np.testing.assert_allclose(actual, expected, atol=2e-6)
        np.testing.assert_allclose(
            chart.frame(position).decode(shifted_mean, index), mean, atol=1e-7
        )
        np.testing.assert_array_equal(shifted_cov, covariance)
    np.testing.assert_allclose(
        chart.decode(position, translated.replay_path(context, encoded)),
        target.replay_path(physical_context, path),
        atol=1e-7,
    )
    labels = jnp.array([[-0.4], [0.6]])
    contexts = jax.vmap(lambda value: translated.context(value, translated.times))(labels)
    memory = sequential_observations(translated, contexts, context)
    carried = jax.jit(memory.path)(encoded)
    sequential = jax.jit(jax.vmap(memory.log_probs))(carried, jnp.arange(3)).T
    expected = jax.vmap(
        lambda label: target.observation_log_probs(
            target.context(label, target.times), chart.decode(label, encoded), readings
        )
    )(labels)
    np.testing.assert_allclose(sequential, expected, atol=3e-6)


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(
    ("method", "paid_pilot"),
    [
        ("particle_rwm", False),
        ("particle_amala", False),
        ("particle_amala_plus", False),
        ("particle_amala_plus", True),
        ("particle_agrad", False),
        ("twisted_particle_agrad", False),
        ("particle_mgrad", False),
    ],
)
def test_charted_joint_kernel_preserves_analytic_physical_posterior(method, paid_pilot):
    target, mean, covariance = _gaussian_joint()
    chart = TranslationChart(
        computed_translation_slopes(target, jnp.array([0.3]), jnp.zeros((3, 1))), jnp.array([0.3])
    )
    path_update = (
        DsmcSpec(method=method, num_particles=8)
        if method in ("particle_rwm", "particle_amala", "particle_amala_plus")
        else GuidedPathSpec(method=method, num_particles=8, kappa=1)
    )
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        path_update=path_update,
        paid_mix=PaidMixture(jnp.full((3, 1), 0.2), jnp.full((3, 1), 0.7), jnp.array([True]))
        if paid_pilot
        else None,
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
@pytest.mark.inference(concern="sampling")
def test_chart_checkpoint_continuation_preserves_physical_draws_and_frozen_native_state(
    monkeypatch,
):
    from grassmoth.models.ssm.inference.samplers.mpgibbs import runner
    from grassmoth.models.ssm.inference.warmup.parameter_adaptation import covariance_windows

    compile_sweeps = runner.compile_projected_sweeps

    def capture_charts(program, initial, **kwargs):
        def step_factory(batch):
            step = program.step_factory(batch)

            def sweep(current, posterior):
                updated, record = step(current, posterior)
                return updated, {**record, "chart_reference": updated.chains.chart.reference}

            return sweep

        return compile_sweeps(replace(program, step_factory=step_factory), initial, **kwargs)

    monkeypatch.setattr(runner, "compile_projected_sweeps", capture_charts)
    target, _, _ = _gaussian_joint()
    initialization = prepare_particle_initialization(
        target,
        spec=GaussianInitializationSpec(),
        key=jax.random.key(22),
        num_chains=1,
        execution_context=EXECUTION_CONTEXT,
    )
    assert isinstance(initialization, ParticleInitialization)
    chart = computed_chart(target, initialization.positions, initialization.paths)
    spec = GuidedPathSpec(method="twisted_particle_agrad", num_particles=8, kappa=1)
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        path_update=spec,
        num_parameter_particles=2,
        initial_chart=chart,
        recompute_translation=True,
    )
    options = MarginalParticleGibbsSpec(
        path_update=spec,
        compiler_reparameterization=True,
    )

    def run(warmup, draws, continuation=None):
        resumed_kernel = (
            kernel
            if continuation is None
            else kernel._replace(
                initial_chart=TranslationChart(chart.slopes + 3.0, chart.reference + 7.0)
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
    references = np.asarray(full.all_chain_extra_fields["chart_reference"])[0]
    changed = np.any(
        references != np.concatenate((np.asarray(chart.reference)[None], references[:-1])), axis=1
    )
    boundaries = np.asarray(covariance_windows(120))[:, 1]
    assert changed.any()
    assert set(np.flatnonzero(changed)) <= set(boundaries)
    prefix = run(120, 2)
    assert not np.array_equal(prefix.final_chart.reference, chart.reference)
    np.testing.assert_array_equal(prefix.final_chart.slopes, chart.slopes)
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
    np.testing.assert_array_equal(prefix.last_paths, prefix.latent_paths[:, -1])

    def failed_chart(_target, _positions, _paths):
        return TranslationChart(jnp.full_like(chart.slopes, jnp.nan), chart.reference)

    monkeypatch.setattr(runner, "computed_chart", failed_chart)
    failed = run_marginal_particle_gibbs(
        target,
        kernel=kernel,
        sampler=SamplerSpec(
            algorithm=options, num_warmup=120, num_samples_per_chain=2, num_chains=1
        ),
        key=jax.random.key(31),
        initialization=initialization,
        execution_context=EXECUTION_CONTEXT,
    )
    assert isinstance(failed, InitializationFailure)
    assert "chart rebuild" in failed.message


@pytest.mark.inference(concern="sampling")
def test_partial_exact_observation_excludes_its_coordinate_from_translation():
    times = np.arange(4.0)
    constraints = _direct_constraints(0.25)
    target = replace(
        interval_target(times, np.zeros((4, 1)), times[:, None], ("last",), state_dim=2),
        exact_constraints=constraints,
    )
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        path_update=GuidedPathSpec(method="twisted_particle_agrad", num_particles=8, kappa=1),
        num_parameter_particles=2,
        exact_constraints=constraints,
        initial_chart=TranslationChart(
            jnp.broadcast_to(jnp.array([[-1.0], [0.5]]), (4, 2, 1)), jnp.zeros(1)
        ),
        recompute_translation=True,
    )
    assert isinstance(kernel.initial_chart, TranslationChart)
    np.testing.assert_array_equal(kernel.initial_chart.slopes[1], [[0.0], [0.5]])
    np.testing.assert_array_equal(kernel.initial_chart.slopes[0], [[-1.0], [0.5]])
    path = jnp.array([[0.0, 0.0], [0.25, 0.5], [0.4, 0.2], [0.1, 0.3]])
    state = _initialize_chain_state(
        jnp.array([0.2]),
        observations=target.observations,
        times=target.times,
        target=target,
        parameter_gradient_fn=kernel.parameter_gradient_fn,
        parameter_chol=kernel.initial_parameter_chol,
        initial_latent_delta=jnp.full((4,), 0.5),
        param_step_size=0.3,
        param_min_scale=1e-6,
        param_max_scale=1.0,
        initial_latent_trajectory=path,
        chart=kernel.initial_chart,
    )
    final, _ = jax.jit(kernel.step_fn)(state, jax.random.key(23))
    assert final.latent_trajectory[1, 0] == 0.25
