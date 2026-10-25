"""Curvature-scaled warmup and freezing of guided particle step sizes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.artifacts.sampling import ParticleAGradSpec, TwistedParticleAGradSpec
from grassmoth.models.ssm.inference.samplers.particle.adaptation import (
    _curvature_limits,
    adapt_scales,
    initialize_adaptation,
    latent_adaptation,
)

pytestmark = pytest.mark.inference(concern="warmup")


@pytest.mark.parametrize("variance", [1e-8, 0.01, 0.04, 0.16])
def test_gaussian_noise_scaling_multichannel_and_pinned_coordinates(variance):
    # Two readings of a free coordinate add information; the sharper pinned
    # coordinate must not determine the free proposal's step.
    path = jnp.zeros((3, 2))
    mask = jnp.tile(jnp.array([True, False]), (3, 1))
    present = jnp.array([2.0, 1.0, 0.0])

    def observe(value, index):
        return -0.5 * (present[index] * value[0] ** 2 + 100 * value[1] ** 2) / variance

    bound = jax.jit(_curvature_limits, static_argnums=0)(observe, path, mask)
    np.testing.assert_allclose(bound[:, 0], [variance, 2 * variance, 10], rtol=2e-6)
    np.testing.assert_array_equal(bound[:, 1], np.full(3, 10))


@pytest.mark.parametrize("kind", ["poisson", "nonconcave"])
def test_non_gaussian_limits_follow_current_warmup_state(kind):
    path = jnp.array([[0.0], [1.0]])

    def observe(value, index):
        del index
        return (
            3 * value[0] - jnp.exp(value[0])
            if kind == "poisson"
            else -0.5 * (1 - value[0] ** 2) ** 2
        )

    actual = _curvature_limits(observe, path, jnp.ones_like(path, bool))
    expected = [2.0, 2 / np.e] if kind == "poisson" else [1.0, 0.5]
    np.testing.assert_allclose(actual[:, 0], expected, rtol=2e-6)


@pytest.mark.parametrize("spec", [ParticleAGradSpec(kappa=1), TwistedParticleAGradSpec(kappa=1)])
def test_controller_obeys_changing_parameter_scale_and_freezes(spec):
    settings = latent_adaptation(spec)

    def step(state, index):
        # A parameter-dependent observation variance changes during warmup.
        variance = jnp.where(index >= 500, 1e-8, jnp.where(index < 250, 0.04, 0.01))
        limit = _curvature_limits(
            lambda x, _t: -0.5 * jnp.sum(x * x) / variance,
            jnp.zeros((2, 1)),
            jnp.ones((2, 1), bool),
        )
        result = adapt_scales(state, settings, jnp.ones(2), jnp.ones(2), 500, limit[:, 0])
        return result, result.delta

    final, deltas = jax.jit(
        lambda: jax.lax.scan(step, initialize_adaptation(jnp.full(2, 0.01)), jnp.arange(510))
    )()
    np.testing.assert_allclose(final.delta, 0.02, rtol=1e-6)
    assert (np.asarray(deltas[:250]) <= 0.080001).all()
    assert (np.asarray(deltas[250:]) <= 0.020001).all()
    np.testing.assert_array_equal(deltas[500:], jnp.broadcast_to(deltas[499], (10, 2)))
    assert final.iteration == 500


@pytest.mark.parametrize("spec", [ParticleAGradSpec(kappa=0), TwistedParticleAGradSpec(kappa=0)])
def test_zero_drift_retains_existing_unbounded_controller(spec):
    settings = latent_adaptation(spec)

    def step(state, index):
        del index
        state = adapt_scales(state, settings, jnp.ones(2), jnp.ones(2), 500, jnp.full(2, 10.0))
        return state, None

    final = jax.jit(
        lambda: jax.lax.scan(step, initialize_adaptation(jnp.full(2, 0.01)), jnp.arange(500))[0]
    )()
    np.testing.assert_allclose(final.delta, 10.0, rtol=1e-6)


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("fixed", [False, True])
def test_runner_initializes_inside_curvature_window_and_keeps_fixed_assignment(fixed):
    import logging
    import time
    from datetime import UTC, datetime

    from grassmoth.artifacts.sampling import FixedLatentStepSpec, GaussianInitializationSpec
    from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
        build_marginal_particle_gibbs_kernel,
    )
    from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import run_marginal_particle_gibbs
    from grassmoth.models.ssm.inference.warmup.initialization import (
        ParticleInitialization,
        prepare_particle_initialization,
    )
    from grassmoth_test_support.compute import COMPUTE_BUDGET, compile_program, prepare_program
    from grassmoth_test_support.interval_particle import interval_target

    target = interval_target(
        np.arange(2.0),
        np.array([[0.1], [-0.1]]),
        np.arange(2.0)[:, None],
        ("last",),
        covariance=np.array([[0.0001]]),
    )
    initialization = prepare_particle_initialization(
        target,
        spec=GaussianInitializationSpec(),
        key=jax.random.key(77),
        num_chains=1,
        compute_budget=COMPUTE_BUDGET,
        compiler=compile_program,
    )
    assert isinstance(initialization, ParticleInitialization)
    spec = (
        TwistedParticleAGradSpec(
            kappa=1, num_particles=4, latent_step=FixedLatentStepSpec(delta=0.003)
        )
        if fixed
        else TwistedParticleAGradSpec(kappa=1, num_particles=4)
    )
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        path_update=spec,
        num_parameter_particles=2,
        exact_constraints=target.exact_constraints,
    )
    result = run_marginal_particle_gibbs(
        target,
        kernel=kernel,
        compiler=compile_program,
        prepare_program=prepare_program,
        num_warmup=500,
        num_samples=2,
        num_chains=1,
        key=jax.random.key(9),
        initialization=initialization,
        compute_budget=COMPUTE_BUDGET,
        clock=time.monotonic,
        wall_clock=lambda: datetime.now(UTC),
        logger=logging.getLogger(__name__),
    )
    if fixed:
        np.testing.assert_allclose(result.initial_latent_delta, 0.003, rtol=1e-6)
        np.testing.assert_array_equal(result.initial_latent_delta, result.final_latent_delta)
    else:
        assert result.initial_latent_delta.shape == result.final_latent_delta.shape == (1, 2)
        np.testing.assert_allclose(result.initial_latent_delta, 0.25)
        assert (np.asarray(result.final_latent_delta) <= 1).all()
        assert (
            np.asarray(result.final_latent_delta) > np.asarray(result.initial_latent_delta)
        ).all()


def test_target_limit_tracks_parameter_noise_and_missing_readings():
    from dataclasses import replace

    import equinox as eqx

    from grassmoth.models.ssm.inference.samplers.particle.adaptation import guided_delta_limit
    from grassmoth_test_support.interval_particle import interval_target

    base = interval_target(
        np.arange(3.0), np.array([[0.1], [np.nan], [-0.1]]), np.arange(3.0)[:, None], ("last",)
    )

    def context(position, times):
        arrays, grid, _static = base.context(position, times)
        return arrays, grid, position

    def declared(bound):
        model = base.dynamical_model(bound)
        return eqx.tree_at(
            lambda m: m.observation_model.measurement.manifest_cov,
            model,
            jnp.eye(1) * 0.04 * jnp.exp(2 * bound[2][0]),
        )

    target = replace(base, context=context, dynamical_model=declared)
    evaluate = jax.jit(
        lambda theta: guided_delta_limit(
            target,
            target.context(jnp.array([theta]), target.times),
            jnp.zeros((3, 1)),
            jnp.ones((3, 1), bool),
        )
    )
    for theta in [-0.5, 0.5]:
        np.testing.assert_allclose(
            evaluate(theta)[:, 0],
            [0.08 * np.exp(2 * theta), 10.0, 0.08 * np.exp(2 * theta)],
            rtol=3e-6,
        )


def test_diagonal_curvature_separates_tight_weak_and_missing_directions():
    path = jnp.zeros((2, 3))
    mask = jnp.array([[True, True, False], [True, True, False]])
    precision = jnp.array([[625.0, 2.0, 0.0], [2.0, 6.25, 0.0], [0.0, 0.0, 1e8]])

    def observe(value, index):
        return jnp.where(index == 0, -0.5 * value @ precision @ value, 0.0)

    steps = jax.jit(_curvature_limits, static_argnums=0)(observe, path, mask)
    np.testing.assert_allclose(steps[0], [2 / 627, 2 / 8.25, 10], rtol=2e-6)
    np.testing.assert_array_equal(steps[1], np.full(3, 10))
    scaled = np.sqrt(steps[0, :2] / 2)
    assert np.linalg.eigvalsh(scaled[:, None] * precision[:2, :2] * scaled[None, :]).max() <= 1
    assert steps[0, 1] > 70 * steps[0, 0]


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("family", ["poisson", "student_t"])
def test_state_dependent_metric_freezes_and_continues_without_recalibration(family):
    import logging
    import time
    from datetime import UTC, datetime

    from grassmoth.artifacts.arrays import NumericalArray
    from grassmoth.artifacts.continuation import SamplerCheckpoint
    from grassmoth.artifacts.probability.families import DistributionFamily
    from grassmoth.artifacts.sampling import (
        GaussianInitializationSpec,
        MarginalParticleGibbsSpec,
        SamplerSpec,
    )
    from grassmoth.models.ssm.inference.continuation import frozen_particle_tuning
    from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
        build_marginal_particle_gibbs_kernel,
    )
    from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import run_marginal_particle_gibbs
    from grassmoth.models.ssm.inference.warmup.initialization import (
        ParticleInitialization,
        prepare_particle_initialization,
    )
    from grassmoth_test_support.compute import COMPUTE_BUDGET, compile_program, prepare_program
    from grassmoth_test_support.interval_particle import interval_target

    times = np.arange(3.0)
    target = interval_target(
        times,
        np.array([[1.0], [4.0], [2.0]]),
        times[:, None],
        ("last",),
        families=(DistributionFamily(family),),
    )
    initialization = prepare_particle_initialization(
        target,
        spec=GaussianInitializationSpec(),
        key=jax.random.key(22),
        num_chains=1,
        compute_budget=COMPUTE_BUDGET,
        compiler=compile_program,
    )
    assert isinstance(initialization, ParticleInitialization)
    spec = TwistedParticleAGradSpec(num_particles=8, kappa=1)
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        path_update=spec,
        num_parameter_particles=2,
        exact_constraints=target.exact_constraints,
    )

    def run(warmup, draws, continuation=None):
        return run_marginal_particle_gibbs(
            target,
            kernel=kernel,
            compiler=compile_program,
            prepare_program=prepare_program,
            num_warmup=warmup,
            num_samples=draws,
            num_chains=1,
            key=jax.random.key(31),
            initialization=initialization,
            compute_budget=COMPUTE_BUDGET,
            clock=time.monotonic,
            wall_clock=lambda: datetime.now(UTC),
            logger=logging.getLogger(__name__),
            continuation=continuation,
        )

    full = run(120, 4)
    prefix = run(120, 2)
    checkpoint = SamplerCheckpoint(
        settings=SamplerSpec(algorithm=MarginalParticleGibbsSpec(path_update=spec)),
        parameters=NumericalArray.from_numpy(np.asarray(prefix.last_parameters)),
        paths=NumericalArray.from_numpy(np.asarray(prefix.last_paths)),
        random_key=NumericalArray.from_numpy(np.asarray(jax.random.key_data(prefix.random_key))),
        native_state=prefix.final_state,
        tuning=frozen_particle_tuning(initialization, None),
    )
    continued = run(0, 2, checkpoint)
    np.testing.assert_array_equal(full.latent_paths[:, 2:], continued.latent_paths)
    np.testing.assert_array_equal(full.grouped_positions[:, 2:], continued.grouped_positions)
    np.testing.assert_array_equal(prefix.final_latent_delta, continued.final_latent_delta)
    # The final native leaf owns the warmup metric; neither a new current path
    # nor the resumed initialization may replace it during retained sampling.
    np.testing.assert_array_equal(prefix.final_state[-1].values, continued.final_state[-1].values)
    assert continued.final_latent_delta.shape == (1, 3)
