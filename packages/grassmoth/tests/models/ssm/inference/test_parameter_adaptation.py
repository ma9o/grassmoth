"""Self-preconditioning learns geometry without changing the exact posterior."""

import logging
import time
from dataclasses import replace
from datetime import UTC, datetime

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from grassmoth.artifacts.sampling import GaussianInitializationSpec
from grassmoth.models.ssm.inference.parameter_transform import ParameterTransform
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import run_marginal_particle_gibbs
from grassmoth.models.ssm.inference.warmup.initialization import (
    ParticleInitialization,
    prepare_particle_initialization,
)
from grassmoth.models.ssm.inference.warmup.parameter_adaptation import (
    adapt_parameter_covariance,
    covariance_windows,
    initialize_parameter_covariance,
)
from grassmoth_test_support.compute import COMPUTE_BUDGET, compile_program, prepare_program
from grassmoth_test_support.interval_particle import interval_target

pytestmark = pytest.mark.inference(concern="warmup")


def test_covariance_counts_repeated_states_excludes_between_chain_means_and_freezes():
    # Distant chain means must not become apparent posterior variance. Repeated
    # selected states must count just as often as moves to new positions.
    values = jnp.array([[-1.0, -2.0], [-1.0, -2.0], [1.0, 2.0], [1.0, 2.0]])
    offsets = jnp.array([[100.0, -200.0], [-300.0, 400.0]])
    positions = values[:, None, :] + offsets[None, :, :]
    initial = jnp.diag(jnp.array([0.5, 2.0]))
    state = initialize_parameter_covariance(positions[0], initial)
    windows = jnp.array([[0, 3]])

    def update(carry, inputs):
        moments, chol = carry
        index, points = inputs
        moments, chol, _ = adapt_parameter_covariance(moments, points, chol, index, windows)
        return (moments, chol), chol

    (state, learned), _ = jax.jit(
        lambda: jax.lax.scan(update, (state, initial), (jnp.arange(4), positions))
    )()
    expected = 4 / 9 * np.cov(np.asarray(values).T) + 5 / 9 * np.asarray(initial @ initial.T)
    np.testing.assert_allclose(learned @ learned.T, expected, rtol=2e-5, atol=2e-5)
    assert int(state.sample_count) == 0
    frozen = adapt_parameter_covariance(state, positions[0] * 100, learned, jnp.array(4), windows)
    np.testing.assert_array_equal(frozen[1], learned)
    assert not bool(frozen[2])
    assert covariance_windows(0).shape == (0, 2)
    assert covariance_windows(10).shape == (0, 2)


@pytest.mark.inference(concern="sampling")
@pytest.mark.inference(concern="recovery")
@pytest.mark.parametrize("proposal", ["random_walk", "pseudo_langevin"])
def test_adapted_particle_kernel_recovers_correlated_unequal_scale_gaussian(proposal):
    mean = jnp.array([0.4, -0.7])
    covariance = jnp.array([[0.04, 0.18], [0.18, 1.0]])
    law = dist.MultivariateNormal(mean, covariance_matrix=covariance)
    target = replace(
        interval_target(np.array([0.0]), np.array([[np.nan]]), np.array([[0.0]]), ("last",)),
        parameters=ParameterTransform(
            jnp.zeros(2), lambda z: {"theta": z}, lambda z: {"theta": z}, law.log_prob
        ),
    )
    initialization = prepare_particle_initialization(
        target,
        spec=GaussianInitializationSpec(),
        key=jax.random.key(15),
        num_chains=4,
        compute_budget=COMPUTE_BUDGET,
    )
    assert isinstance(initialization, ParticleInitialization)
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_particles=4,
        num_parameter_particles=2,
        param_step_size=0.02,
        parameter_preconditioner_chol=initialization.parameter_chol,
        parameter_proposal=proposal,
    )
    result = run_marginal_particle_gibbs(
        target,
        kernel=kernel,
        compiler=compile_program,
        prepare_program=prepare_program,
        num_warmup=1000,
        num_samples=4000,
        num_chains=4,
        key=jax.random.key(293),
        adaptation_rate=0.05,
        adaptation_scheme="dual_averaging",
        retain_latent_paths=False,
        initialization=initialization,
        compute_budget=COMPUTE_BUDGET,
        clock=time.monotonic,
        wall_clock=lambda: datetime.now(UTC),
        logger=logging.getLogger(__name__),
    )
    draws = np.asarray(result.grouped_positions).reshape((-1, 2))
    np.testing.assert_allclose(draws.mean(axis=0), mean, atol=0.06)
    np.testing.assert_allclose(np.cov(draws.T), covariance, rtol=0.2, atol=0.012)
    learned = np.asarray(result.final_parameter_chol @ result.final_parameter_chol.T)
    np.testing.assert_allclose(learned, covariance, rtol=0.35, atol=0.02)
    assert not np.array_equal(result.final_parameter_chol, initialization.parameter_chol)
