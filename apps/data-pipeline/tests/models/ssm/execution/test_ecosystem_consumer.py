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

from nof1_causal_lab.artifacts.model.construct import replace_constructs
from nof1_causal_lab.artifacts.reports.posterior_diagnostics import ParticleSamplerDiagnostics
from nof1_causal_lab.artifacts.sampling import (
    DsmcSpec,
    LapsInitializationSpec,
    MarginalParticleGibbsSpec,
    SamplerSpec,
)
from nof1_causal_lab.models.ssm.dynamics.vector_field import StructuralDrift
from nof1_causal_lab.models.ssm.execution.dynamical_model import HeterogeneousObservation
from nof1_causal_lab.models.ssm.inference.fit import fit
from nof1_causal_lab.models.ssm.inference.problem import build_particle_problem
from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure
from nof1_causal_lab.models.ssm.preparation.preflight import ObservationPreflightFailure
from tests.support.compute import COMPUTE_BUDGET
from tests.support.inference import bind_panel_fixture, compile_fit_fixture
from tests.support.model import construct_named, load_model_fixture


def _nonlinear_mixed_missing_irregular_particle_fit_and_exact_diagnostics_nonlinear_model() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "ecosystem_consumer/nonlinear_mixed_missing_irregular_particle_fit_and_exact_diagnostics_nonlinear_model.json"
    )


if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec


pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize("intervals", [False, True])
@pytest.mark.inference(concern="warmup")
def test_nonlinear_mixed_missing_irregular_particle_fit_and_exact_diagnostics(intervals):
    spec = _nonlinear_mixed_missing_irregular_particle_fit_and_exact_diagnostics_nonlinear_model()
    if intervals:
        construct = construct_named(spec, "latent_0")
        revised = construct.revised(
            indicators=tuple(
                indicator.revised(
                    observation=indicator.observation.revised(
                        aggregation="mean" if index == 0 else "sum",
                        observation_window="72m",
                    )
                )
                for index, indicator in enumerate(construct.indicators)
            )
        )
        spec = spec.with_entities(edges=replace_constructs(spec.edges, (revised,)))
    model = compile_fit_fixture(spec)
    times = jnp.array([0.0, 0.05, 0.17, 0.4, 0.9])
    observations = jnp.array(
        [[0.2, 1.0], [jnp.nan, 2.0], [0.3, jnp.nan], [jnp.nan, jnp.nan], [-0.2, 0.0]]
    )
    if intervals:
        observations = observations.at[0].set(jnp.nan)
    gaussian = observations[:, 0]
    observations = observations.at[:, 0].set(
        (gaussian - jnp.nanmean(gaussian)) / jnp.nanstd(gaussian)
    )
    panel = bind_panel_fixture(model.compiled_dynamical_model, observations, times)
    problem = build_particle_problem(
        model.prior_runtime_bundle,
        panel,
        scheme="euler_maruyama",
        trace_key=jax.random.key(7),
        reparam=None,
    )
    runtime = problem.runtime
    times, observations = panel.times, panel.observations
    context = runtime.context(runtime.initial_position, times)
    path = jnp.linspace(0.1, 0.3, len(times))[:, None]
    # Independent emission expressions, including all- and partially-missing rows.
    observation_model = context[0].observation_model
    assert isinstance(observation_model, HeterogeneousObservation)
    evolution = context[0].state_evolution
    assert isinstance(evolution, dsx.StochasticContinuousTimeStateEvolution)
    assert isinstance(evolution.drift, StructuralDrift)
    assert evolution.diffusion is not None
    variance = observation_model.measurement.manifest_cov[0, 0]
    expected = jnp.zeros(len(times))
    for row, column in zip(*np.where(np.isfinite(observations)), strict=True):
        response = path[:, 0] if column == 0 else jnp.exp(path[:, 0])
        mean = response[row]
        if intervals:
            start = panel.observation_support.support_start_times[row, column]
            first = int(np.argmin(np.abs(np.asarray(times) - start)))
            width = times[row] - times[first]
            mean = jnp.trapezoid(response[first : row + 1], times[first : row + 1])
            if column == 0:
                mean = mean / width
        law = dist.Normal(mean, jnp.sqrt(variance)) if column == 0 else dist.Poisson(mean)
        expected = expected.at[row].add(law.log_prob(observations[row, column]))
    np.testing.assert_allclose(
        runtime.observation_log_probs(context, path, observations), expected, atol=2e-5
    )
    transition = runtime.transition_log_prob(context, path[0], path[1], jnp.asarray(1))
    step = times[1] - times[0]
    # Genuine nonlinear drift, rather than agreement between two linear surrogates.
    # Center is the sole free drift coefficient; stiffness and quartic are fixed.
    (center,) = evolution.drift.args.params[0].values()
    drift = -0.4 * (path[0] - center) - 0.2 * (path[0] - center) ** 3
    law = dist.MultivariateNormal(
        path[0] + step * drift,
        covariance_matrix=step * evolution.diffusion.gram_matrix(x=None, u=None, t=0, state_dim=1),
    )
    np.testing.assert_allclose(transition, law.log_prob(path[1]), atol=2e-5)
    result = fit(
        model.prior_runtime_bundle,
        panel,
        reparam=None,
        sampler=SamplerSpec(
            num_warmup=2,
            num_samples_per_chain=4,
            num_chains=1,
            seed=9,
            num_particles=4,
            retain_latent_paths=True,
            algorithm=MarginalParticleGibbsSpec(
                n_parameter_particles=2,
                parameter_proposal="pseudo_langevin",
                adaptation_scheme="dual_averaging",
                param_step_size=0.0005,
                laps=LapsInitializationSpec(
                    ensemble_size=128, num_unadjusted_steps=1000, num_adjusted_steps=20
                ),
                dsmc=DsmcSpec(drift="amala_plus"),
            ),
        ),
        clock=time.monotonic,
        compute_budget=COMPUTE_BUDGET,
    )
    assert not isinstance(result, (ObservationPreflightFailure, InitializationFailure))
    assert not hasattr(result.diagnostics, "likelihood_backend")
    diagnostics = result.diagnostics.sampler_diagnostics
    assert isinstance(diagnostics, ParticleSamplerDiagnostics)
    assert isinstance(diagnostics.settings.algorithm, MarginalParticleGibbsSpec)
    assert diagnostics.parameter_kernel == "m_pgibbs_pseudo_langevin"
    assert diagnostics.drift == "none"
    assert diagnostics.settings.algorithm.adaptation_scheme == "dual_averaging"
    latent_paths = result.draws.latent_paths
    assert latent_paths is not None
    assert latent_paths.shape == (4, len(times), 1)
    factors = result.diagnostics.observation_log_probs
    assert factors.shape == (1, 4, len(times))
    assert jnp.all(jnp.isfinite(factors))
    assert jnp.all(factors[..., jnp.all(jnp.isnan(observations), axis=1)] == 0.0)
    for values in result.get_samples().values():
        assert values.shape[0] == 4
        assert jnp.all(jnp.isfinite(values))
