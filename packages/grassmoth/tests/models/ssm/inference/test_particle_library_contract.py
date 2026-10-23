"""Cheap checks of the app/library boundary; no fitting or sampler execution."""

import ast
from importlib import import_module
from operator import setitem
from pathlib import Path
from typing import Never
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.models.ssm.compile.observation_partition import CompiledObservationPartition
from grassmoth.models.ssm.compile.state_space import StateSpace
from grassmoth.models.ssm.inference.extraction import (
    extract_grouped_public_samples,
)
from grassmoth.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
from grassmoth.models.ssm.inference.parameter_tracing import extract_constrained_samples
from grassmoth.models.ssm.inference.parameter_transform import ParameterTransform
from grassmoth.models.ssm.inference.problem import ParticleProblem
from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import (
    _initialize_chain_state,
)
from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
from grassmoth.models.ssm.inference.types import (
    JointPosteriorDraws,
    MCMCPosterior,
    ProductionDiagnostics,
)
from grassmoth_test_support.inference import particle_posterior

pytestmark = pytest.mark.contract


def _no_numerical_execution(*_args) -> Never:
    raise AssertionError("This boundary check must not execute the numerical model")


def test_inference_uses_dynestyx_targets_without_external_sampler_implementations():
    source_root = Path(__file__).resolve().parents[4] / "src/grassmoth/models/ssm/inference"
    forbidden = (
        "cuthbert",
        "dynestyx.inference.particle_mcmc",
        "dynestyx.inference.particle_chain",
        "dynestyx.inference.particle_profiling",
    )
    violations = []
    for source in source_root.rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
            else:
                continue
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for module in modules
                for prefix in forbidden
            ):
                violations.append(f"{source.relative_to(source_root)}:{node.lineno}")
    assert not violations, f"Inference implementation must stay in the app: {violations}"


def test_library_parameter_output_preserves_public_vector_sites_and_chain_order():
    positions = jnp.arange(12.0).reshape(2, 3, 2)
    parameters = ParameterTransform(
        initial_position=jnp.zeros(2),
        unravel=_no_numerical_execution,
        constrain=lambda z: {"beta": 2 * z, "beta_decentered": z},
        log_prior=_no_numerical_execution,
    )
    runtime = ParticleTarget(
        parameters=parameters,
        context=_no_numerical_execution,
        dynamical_model=_no_numerical_execution,
        observations=jnp.zeros((4, 1)),
        times=jnp.arange(4.0),
        state_space=StateSpace(1, (), (0,), (0,), (), ()),
        observation_partition=CompiledObservationPartition((0,), ()),
        point_reading_mask=jnp.ones((4, 1), dtype=bool),
    )
    problem = ParticleProblem(runtime=runtime, site_info={}, public_sites=frozenset({"beta"}))
    samples = extract_grouped_public_samples(positions, bundle=problem, num_chains=2, num_samples=3)
    assert set(samples) == {"beta"}
    np.testing.assert_array_equal(samples["beta"], 2 * positions)
    mcmc = TrajectoryMCMCResult(samples, {}, num_chains=2, num_samples=3)
    np.testing.assert_array_equal(mcmc.get_samples()["beta"], (2 * positions).reshape(6, 2))


def test_supplied_initial_path_does_not_compute_a_predictive_rollout():
    target = Mock(
        spec=ParticleTarget,
        context=Mock(),
        initial_path=Mock(side_effect=_no_numerical_execution),
        replay_path=Mock(side_effect=lambda _context, path: path),
        initial_moments=Mock(return_value=(jnp.zeros(1), jnp.eye(1))),
        log_posterior_from_context=Mock(return_value=(jnp.array(-2.0), jnp.array(-1.0))),
    )
    initial_path = jnp.array([[0.1], [0.2], [0.3]])
    state = _initialize_chain_state(
        jnp.zeros(1),
        observations=jnp.zeros((3, 1)),
        times=jnp.arange(3.0),
        target=target,
        parameter_gradient_fn=jnp.zeros_like,
        parameter_chol=jnp.eye(1),
        initial_latent_delta=jnp.full(3, 0.1),
        param_step_size=0.01,
        param_min_scale=1e-6,
        param_max_scale=1.0,
        initial_latent_trajectory=initial_path,
    )
    target.initial_path.assert_not_called()
    np.testing.assert_array_equal(state.latent_trajectory, initial_path)
    target.log_posterior_from_context.assert_called_once()


def test_fixed_model_samples_replay_deterministics_with_zero_parameter_dimension():
    parameters = ParameterTransform(
        initial_position=jnp.zeros(0),
        unravel=_no_numerical_execution,
        constrain=lambda _: {"lambda": jnp.ones((2, 1)), "internal": jnp.array(5.0)},
        log_prior=_no_numerical_execution,
    )
    samples = extract_constrained_samples(jnp.zeros((3, 0)), parameters, {"lambda"})
    assert set(samples) == {"lambda"}
    np.testing.assert_array_equal(samples["lambda"], np.ones((3, 2, 1)))
    bundle = Mock(
        spec=ParticleProblem,
        runtime=Mock(
            spec=ParticleTarget, initial_position=parameters.initial_position, parameters=parameters
        ),
        public_sites=frozenset({"lambda"}),
    )
    grouped = extract_grouped_public_samples(
        jnp.zeros((2, 3, 0)), bundle=bundle, num_chains=2, num_samples=3
    )
    np.testing.assert_array_equal(grouped["lambda"], np.ones((2, 3, 2, 1)))


def test_loo_uses_joint_emissions_and_omits_only_completely_missing_rows(monkeypatch):
    samples = {"beta": jnp.arange(12.0).reshape(2, 3, 2)}
    factors = jnp.arange(24.0).reshape(2, 3, 4) / -10
    observations = jnp.array([[1.0, 2.0], [jnp.nan, jnp.nan], [3.0, jnp.nan], [jnp.nan, 4.0]])
    mcmc = TrajectoryMCMCResult(samples, {}, num_chains=2, num_samples=3)
    posterior = MCMCPosterior.from_run(
        draws=JointPosteriorDraws(parameters=mcmc.get_samples()),
        diagnostics=ProductionDiagnostics(mcmc=mcmc, observation_log_probs=factors),
    )

    def estimate(idata, *, pointwise):
        assert pointwise is True
        np.testing.assert_array_equal(
            idata.log_likelihood["measurement_row"].values, factors[:, :, [0, 2, 3]]
        )
        np.testing.assert_array_equal(idata.posterior["beta"].values, samples["beta"])
        return Mock(
            spec=import_module("arviz_stats.utils").ELPDDataLOO,
            elpd=-7.0,
            p=0.4,
            se=0.2,
            n_data_points=3,
            pareto_k=Mock(values=np.array([0.2, 0.8, 0.6])),
        )

    # Test the scientific factor boundary without computing PSIS or fitting.
    monkeypatch.setattr(import_module("arviz_stats.loo"), "loo", estimate)
    measured = posterior.get_loo_diagnostics(
        observed_rows=jnp.any(jnp.isfinite(observations), axis=1)
    )
    assert measured is not None
    result, points = measured
    assert [(point.timestep, point.k, point.status) for point in points] == [
        (3, 0.8, "failed"),
        (4, 0.6, "failed"),
        (1, 0.2, "passed"),
    ]
    assert result.elpd_loo == -7.0
    assert result.p_loo == 0.4
    assert result.se == 0.2
    assert result.n_data_points == len(points) == 3
    assert result.n_bad_k == 1
    assert result.n_warn_k == 1


def test_loo_all_missing_rows_have_no_predictive_estimate():
    posterior = MCMCPosterior.from_run(
        draws=JointPosteriorDraws(parameters={}),
        diagnostics=ProductionDiagnostics(
            observation_log_probs=jnp.zeros((2, 3, 4)),
            mcmc=TrajectoryMCMCResult({}, {}, num_chains=2, num_samples=3),
        ),
    )
    assert posterior.get_loo_diagnostics(observed_rows=jnp.zeros(4, dtype=bool)) is None


def test_loo_cannot_reweight_away_an_exact_state_constraint():
    posterior = MCMCPosterior.from_run(
        draws=JointPosteriorDraws(parameters={}),
        diagnostics=ProductionDiagnostics(
            observation_log_probs=jnp.zeros((2, 3, 2)),
            exact_observation_rows=jnp.array([True, False]),
            mcmc=TrajectoryMCMCResult({}, {}, num_chains=2, num_samples=3),
        ),
    )
    assert posterior.get_loo_diagnostics(observed_rows=jnp.ones(2, dtype=bool)) is None


def test_published_particle_buffers_are_frozen_and_do_not_alias_builder_maps():
    from dataclasses import FrozenInstanceError

    samples = {"beta": jnp.zeros(3)}
    posterior = particle_posterior(draws=JointPosteriorDraws(parameters=samples))
    samples["beta"] = jnp.ones(3)
    np.testing.assert_array_equal(posterior.get_samples()["beta"], jnp.zeros(3))
    with pytest.raises(TypeError):
        setitem(posterior.get_samples(), "beta", jnp.ones(3))
    with pytest.raises(FrozenInstanceError):
        posterior.diagnostics.observation_log_probs = jnp.ones((1, 3, 1))
