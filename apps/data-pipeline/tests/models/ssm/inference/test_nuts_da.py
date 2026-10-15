"""Joint NUTS recovery and production posterior contracts."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.reports.posterior_diagnostics import NutsDADiagnostics
from nof1_causal_lab.artifacts.sampling import NutsDASpec, SamplerSpec
from nof1_causal_lab.models.ssm.inference.fit import fit
from nof1_causal_lab.models.ssm.inference.samplers.nuts import NutsDARun, run_nuts_da
from nof1_causal_lab.models.ssm.inference.types import MCMCPosterior
from tests.models.ssm.inference.test_laps_initialization import _nonlinear_fixture
from tests.support.compute import COMPUTE_BUDGET
from tests.support.interval_particle import gaussian_interval_case

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.inference(concern="recovery")
def test_parallel_joint_nuts_recovers_analytic_parameter_state_covariance():
    target, mean, covariance = gaussian_interval_case()
    run = run_nuts_da(
        target,
        sampler=SamplerSpec(
            algorithm=NutsDASpec(),
            num_warmup=250,
            num_samples_per_chain=750,
            num_chains=4,
        ),
        compute_budget=COMPUTE_BUDGET,
        key=jax.random.key(904),
        clock=time.monotonic,
    )
    assert isinstance(run, NutsDARun)
    assert run.paths is not None
    assert run.positions.shape[:2] == run.paths.shape[:2] == (4, 750)
    joint = np.concatenate((run.positions, run.paths[:, :, :, 0]), axis=-1).reshape(-1, mean.size)
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), (joint - mean).T).T
    np.testing.assert_allclose(whitened.mean(axis=0), 0, atol=0.12)
    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(mean.size), atol=0.18)
    assert not np.array_equal(run.positions[0], run.positions[1])
    extra = run.phase_extra_fields["post_warmup"]
    assert extra["diverging"].shape == (4, 750)
    assert float(jnp.mean(extra["diverging"])) < 0.02
    assert run.phase_extra_fields["warmup"]["energy"].shape == (4, 250)
    # Adaptation has ended before any retained draw.
    np.testing.assert_allclose(
        extra["step_size"], np.broadcast_to(run.final_step_size[:, None], (4, 750))
    )
    np.testing.assert_array_equal(extra["tree_depth"], np.ceil(np.log2(extra["num_steps"] + 1)))
    for chain in range(4):
        context = target.context(run.positions[chain, 0], target.times)
        expected = target.observation_log_probs(context, run.paths[chain, 0], target.observations)
        np.testing.assert_allclose(run.observation_log_probs[chain, 0], expected, rtol=2e-6)


@pytest.mark.parametrize("retain_paths", [False, True])
def test_nuts_fit_preserves_nonlinear_model_and_shared_posterior(retain_paths):
    prepared, panel = _nonlinear_fixture()
    result = fit(
        prepared.prior_runtime_bundle,
        panel,
        sampler=SamplerSpec(
            algorithm=NutsDASpec(max_tree_depth=3, initial_step_size=0.01),
            num_warmup=0,
            num_samples_per_chain=4,
            num_chains=2,
            retain_latent_paths=retain_paths,
        ),
        compute_budget=COMPUTE_BUDGET,
        reparam=None,
        clock=time.monotonic,
    )
    assert isinstance(result, MCMCPosterior)
    assert result.evidence.method == "nuts_da"
    assert result.draws.describe().n_draws == 8
    assert all(values.shape[0] == 8 for values in result.get_samples().values())
    assert (result.draws.latent_paths is not None) == retain_paths
    if result.draws.latent_paths is not None:
        assert result.draws.latent_paths.shape == (8, len(panel.times), 1)
    assert np.isfinite(result.diagnostics.observation_log_probs).all()
    assert isinstance(result.diagnostics.sampler_diagnostics, NutsDADiagnostics)
    assert result.diagnostics.phase_extra_fields is not None
    assert result.diagnostics.phase_extra_fields["warmup"]["energy"].shape == (2, 0)
