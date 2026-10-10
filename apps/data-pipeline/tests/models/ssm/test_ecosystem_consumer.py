"""Numerical acceptance of Dynestyx model interpretation with local inference."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import dynestyx as dsx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.models.ssm.dynamics.vector_field import StructuralDrift
from nof1_causal_lab.models.ssm.execution.dynamical_model import HeterogeneousObservation
from nof1_causal_lab.models.ssm.inference import fit
from nof1_causal_lab.models.ssm.inference.problem import build_particle_problem
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure
from nof1_causal_lab.sampler_config import MarginalParticleGibbsSpec, SamplerSpec
from tests.compute_fixtures import COMPUTE_BUDGET
from tests.inference_fixtures import bind_panel_fixture, compile_fit_fixture
from tests.model_fixtures import load_model_fixture


def _nonlinear_mixed_missing_irregular_particle_fit_and_exact_diagnostics_nonlinear_model() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "ecosystem_consumer/nonlinear_mixed_missing_irregular_particle_fit_and_exact_diagnostics_nonlinear_model.json"
    )


if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.dynamical_model_spec import DynamicalModelSpec


pytestmark = pytest.mark.inference(concern="sampling")


def test_nonlinear_mixed_missing_irregular_particle_fit_and_exact_diagnostics(monkeypatch):
    from nof1_causal_lab.models.ssm.inference.warmup import latent_init

    def _unexpected_ieks(*_args, **_kwargs):
        raise AssertionError("supplied trajectories must skip IEKS initialization")

    monkeypatch.setattr(latent_init, "compute_ieks_latent_paths", _unexpected_ieks)
    model = compile_fit_fixture(
        _nonlinear_mixed_missing_irregular_particle_fit_and_exact_diagnostics_nonlinear_model()
    )
    times = jnp.array([0.0, 0.05, 0.17, 0.4, 0.9])
    observations = jnp.array(
        [[0.2, 1.0], [jnp.nan, 2.0], [0.3, jnp.nan], [jnp.nan, jnp.nan], [-0.2, 0.0]]
    )
    gaussian = observations[:, 0]
    observations = observations.at[:, 0].set(
        (gaussian - jnp.nanmean(gaussian)) / jnp.nanstd(gaussian)
    )
    problem = build_particle_problem(
        model.prior_runtime_bundle,
        bind_panel_fixture(model.compiled_dynamical_model, observations, times),
        scheme="euler_maruyama",
        trace_key=jax.random.key(7),
        reparam=None,
    )
    runtime = problem.runtime
    context = runtime.context(runtime.initial_position, times)
    path = jnp.array([[0.1], [0.15], [-0.2], [0.05], [0.3]])
    # Independent emission expressions, including all- and partially-missing rows.
    observation_model = context[0].observation_model
    assert isinstance(observation_model, HeterogeneousObservation)
    evolution = context[0].state_evolution
    assert isinstance(evolution, dsx.StochasticContinuousTimeStateEvolution)
    assert isinstance(evolution.drift, StructuralDrift)
    assert evolution.diffusion is not None
    variance = observation_model.measurement.manifest_cov[0, 0]
    expected = jnp.array(
        [
            dist.Normal(path[0, 0], jnp.sqrt(variance)).log_prob(observations[0, 0])
            + dist.Poisson(jnp.exp(path[0, 0])).log_prob(1.0),
            dist.Poisson(jnp.exp(path[1, 0])).log_prob(2.0),
            dist.Normal(path[2, 0], jnp.sqrt(variance)).log_prob(observations[2, 0]),
            0.0,
            dist.Normal(path[4, 0], jnp.sqrt(variance)).log_prob(observations[4, 0])
            + dist.Poisson(jnp.exp(path[4, 0])).log_prob(0.0),
        ]
    )
    np.testing.assert_allclose(
        runtime.observation_log_probs(context, path, observations), expected, atol=2e-5
    )
    transition = runtime.transition_log_prob(context, path[0], path[1], jnp.asarray(1))
    # Genuine nonlinear drift, rather than agreement between two linear surrogates.
    # Center is the sole free drift coefficient; stiffness and quartic are fixed.
    (center,) = evolution.drift.args.params[0].values()
    drift = -0.4 * (path[0] - center) - 0.2 * (path[0] - center) ** 3
    law = dist.MultivariateNormal(
        path[0] + 0.05 * drift,
        covariance_matrix=0.05 * evolution.diffusion.gram_matrix(x=None, u=None, t=0, state_dim=1)
        + jnp.eye(1) * 1e-8,
    )
    np.testing.assert_allclose(transition, law.log_prob(path[1]), atol=2e-5)
    result = fit(
        model.prior_runtime_bundle,
        bind_panel_fixture(model.compiled_dynamical_model, observations, times),
        reparam=None,
        sampler=SamplerSpec(
            num_warmup=2,
            num_samples_per_chain=4,
            num_chains=1,
            seed=9,
            num_particles=4,
            retain_latent_paths=True,
            marginal_particle_gibbs=MarginalParticleGibbsSpec(
                n_parameter_particles=2,
                parameter_proposal="pseudo_langevin",
                dsmc_leaf_proposal="paid_mix",
                adaptation_scheme="dual_averaging",
                param_step_size=0.0005,
                latent_delta=0.2,
                init_method="random",
                auto_preconditioner_method="none",
                init_scale=0.0,
            ),
        ),
        clock=time.monotonic,
        compute_budget=COMPUTE_BUDGET,
    )
    assert not isinstance(result, ObservationPreflightFailure)
    assert not hasattr(result.diagnostics, "likelihood_backend")
    diagnostics = result.diagnostics.marginal_particle_gibbs
    assert diagnostics is not None
    assert diagnostics.parameter_kernel == "m_pgibbs_pseudo_langevin"
    assert diagnostics.dsmc_leaf_proposal == "paid_mix"
    assert diagnostics.settings.marginal_particle_gibbs.adaptation_scheme == "dual_averaging"
    latent_paths = result.draws.latent_paths
    assert latent_paths is not None
    assert latent_paths.shape == (4, 5, 1)
    factors = result.diagnostics.observation_log_probs
    assert factors.shape == (1, 4, 5)
    assert jnp.all(jnp.isfinite(factors))
    assert jnp.all(factors[..., 3] == 0.0)
    for values in result.get_samples().values():
        assert values.shape[0] == 4
        assert jnp.all(jnp.isfinite(values))
