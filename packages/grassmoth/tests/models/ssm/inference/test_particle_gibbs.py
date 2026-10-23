import logging
import time
from datetime import UTC, datetime

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.expressions import coefficient
from grassmoth.artifacts.reports.posterior_diagnostics import (
    ConditionalMetropolisDiagnostics,
    ConditionalNutsDiagnostics,
    ParticleGibbsDiagnostics,
)
from grassmoth.artifacts.sampling import (
    ConditionalNutsSpec,
    DsmcSpec,
    FixedLatentStepSpec,
    MetropolisHastingsSpec,
    ParticleGibbsSpec,
    SamplerSpec,
)
from grassmoth.models.ssm.inference.fit import fit
from grassmoth.models.ssm.inference.samplers.particle.contract import PaidMixture
from grassmoth.models.ssm.inference.samplers.pgibbs.kernel import build_particle_gibbs_kernel
from grassmoth.models.ssm.inference.samplers.pgibbs.parameters import (
    fixed_parameter_kernel,
    metropolis_kernel,
    nuts_kernel,
)
from grassmoth.models.ssm.inference.types import MCMCPosterior
from grassmoth_test_support.compute import COMPUTE_BUDGET, compile_program, prepare_program
from grassmoth_test_support.inference import (
    nonlinear_fit_fixture,
)
from grassmoth_test_support.interval_particle import gaussian_interval_case
from packages.grassmoth.tests.models.ssm.compile.test_exact_state_space import _model, _problem

"""Blocked Particle Gibbs stationarity and its production posterior contract."""


pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize("fixed_parameters", [False, True])
def test_pg_replays_static_and_accumulated_states_with_or_without_parameters(fixed_parameters):
    model = _model()
    if fixed_parameters:
        baseline = next(c for c in model.constructs if c.name == "baseline").revised(
            coefficients=tuple(
                coefficient(0.0, role=role)
                for role in (
                    "initial_mean",
                    "initial_scale",
                    "diffusion_scale",
                )
            ),
        )
        model = model.with_entities(edges=replace_constructs(model.edges, (baseline,)))
    compiled, problem = _problem(model)
    target = problem.runtime
    assert (target.initial_position.size == 0) == fixed_parameters
    parameters = (
        fixed_parameter_kernel()
        if fixed_parameters
        else nuts_kernel(
            target,
            ConditionalNutsSpec(initial_step_size=0.02, max_tree_depth=3),
            jnp.eye(target.initial_position.size),
            2,
        )
    )
    kernel = build_particle_gibbs_kernel(
        target,
        DsmcSpec(latent_step=FixedLatentStepSpec(delta=0.1), pilot_mixture=False),
        parameters,
        num_particles=8,
        num_warmup=2,
        paid_mix=None,
    )
    context = target.context(target.initial_position, target.times)
    initial = kernel.initialize(
        target.initial_position, target.initial_path(context), jax.random.key(1)
    )
    step = jax.jit(kernel.step)
    current = initial
    for key in jax.random.split(jax.random.key(2), 4):
        current, info = step(current, key)
        position = parameters.position(current.parameters)
        context = target.context(position, target.times)
        np.testing.assert_array_equal(current.path, target.replay_path(context, current.path))
        assert np.isfinite(current.log_posterior)
        assert ("parameter_accept_prob" in info) != fixed_parameters
    indices = {s.name: i for i, s in enumerate(compiled.states)}
    np.testing.assert_allclose(
        current.path[1:, indices["total"]],
        jnp.cumsum(current.path[:-1, indices["driver"]]),
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        current.path[:, indices["baseline"]],
        jnp.repeat(current.path[0, indices["baseline"]], len(current.path)),
    )


@pytest.mark.parametrize("method", ["mh", "nuts"])
def test_blocked_pg_preserves_exact_joint_interval_target(method):
    target, mean, covariance = gaussian_interval_case()
    options = DsmcSpec(latent_step=FixedLatentStepSpec(delta=0.3))
    chol = jnp.linalg.cholesky(jnp.asarray(covariance))
    parameter_chol = jnp.sqrt(jnp.asarray(covariance[:1, :1]))
    parameters = (
        metropolis_kernel(target, MetropolisHastingsSpec(initial_step_size=1.0), parameter_chol, 0)
        if method == "mh"
        else nuts_kernel(
            target, ConditionalNutsSpec(initial_step_size=0.3, max_tree_depth=4), parameter_chol, 0
        )
    )
    kernel = build_particle_gibbs_kernel(
        target,
        options,
        parameters,
        num_particles=8,
        num_warmup=0,
        paid_mix=PaidMixture(
            jnp.asarray(mean[1:, None]),
            jnp.asarray(np.diag(covariance)[1:, None]),
            jnp.asarray([True]),
        ),
    )
    reference_key, init_key, step_key = jax.random.split(jax.random.key(518), 3)
    draws = jnp.asarray(mean) + jax.random.normal(reference_key, (2048, len(mean))) @ chol.T
    states = jax.jit(jax.vmap(kernel.initialize))(
        draws[:, :1], draws[:, 1:, None], jax.random.split(init_key, len(draws))
    )
    updated, info = jax.jit(jax.vmap(kernel.step))(states, jax.random.split(step_key, len(draws)))
    result = jnp.concatenate(
        (jax.vmap(parameters.position)(updated.parameters), updated.path[:, :, 0]), axis=-1
    )
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), (np.asarray(result) - mean).T).T
    np.testing.assert_allclose(whitened.mean(axis=0), 0.0, atol=0.10)
    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(len(mean)), atol=0.13)
    assert float(jnp.mean(info["latent_accept_prob"])) > 0.1
    assert float(jnp.mean(info["parameter_accepted"])) > 0.1


@pytest.mark.parametrize("method", ["mh", "nuts"])
def test_pg_fit_preserves_nonlinear_model_and_frozen_adaptation(method):
    prepared, panel = nonlinear_fit_fixture()
    result = fit(
        prepared.prior_runtime_bundle,
        panel,
        sampler=SamplerSpec(
            algorithm=ParticleGibbsSpec(
                diagnostic_metrics_all=method == "mh",
                parameter_kernel=MetropolisHastingsSpec()
                if method == "mh"
                else ConditionalNutsSpec(max_tree_depth=3, initial_step_size=0.01),
                dsmc=DsmcSpec(),
            ),
            num_warmup=5,
            num_samples_per_chain=4,
            num_chains=2,
            num_particles=4,
        ),
        compute_budget=COMPUTE_BUDGET,
        reparam=None,
        clock=time.monotonic,
        compiler=compile_program,
        prepare_program=prepare_program,
        logger=logging.getLogger(__name__),
        wall_clock=lambda: datetime.now(UTC),
    )
    assert isinstance(result, MCMCPosterior)
    assert result.evidence.method == "pgibbs"
    assert result.draws.describe().n_draws == 8
    assert result.draws.latent_paths is not None
    assert np.isfinite(result.diagnostics.observation_log_probs).all()
    diagnostics = result.diagnostics.sampler_diagnostics
    assert isinstance(diagnostics, ParticleGibbsDiagnostics)
    assert isinstance(
        diagnostics.parameters,
        ConditionalMetropolisDiagnostics if method == "mh" else ConditionalNutsDiagnostics,
    )
    phases = result.diagnostics.phase_extra_fields
    assert phases is not None
    retained = phases["post_warmup"]
    if method == "mh":
        assert diagnostics.dsmc_weights is not None
        assert diagnostics.dsmc_weights.phase == "post_warmup"
        assert diagnostics.dsmc_weights.terminal.candidates == 4
        assert retained["dsmc_weight_ess"].shape == (2, 4, len(diagnostics.dsmc_weights.tree) + 1)
    else:
        assert diagnostics.dsmc_weights is None
        assert not any(key.startswith("dsmc_weight_") for key in retained)
    assert retained["latent_accept_prob"].shape == (2, 4)
    assert ("diverging" in retained) == (method == "nuts")
    assert "energy" not in retained
    np.testing.assert_array_equal(
        retained["step_size"],
        np.broadcast_to(np.asarray(diagnostics.parameters.final_step_size)[:, None], (2, 4)),
    )
    np.testing.assert_array_equal(
        retained["latent_delta"],
        np.broadcast_to(
            np.asarray(result.final_latent_delta)[:, None], retained["latent_delta"].shape
        ),
    )
