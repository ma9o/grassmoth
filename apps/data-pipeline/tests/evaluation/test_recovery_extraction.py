"""Recovery summaries share typed scalar and tensor coordinate selection."""

import jax.numpy as jnp
import pytest
from evaluation.fixtures.synthetic_nonlinear import RecoveryTargetSpec
from evaluation.recovery import extraction

from nof1_causal_lab.artifacts.model.parameter import ParameterCoordinate
from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
from nof1_causal_lab.models.ssm.inference.types import (
    JointPosteriorDraws,
    ParticleMCMCPosterior,
    ProductionDiagnostics,
)

pytestmark = [pytest.mark.inference(concern="sampling"), pytest.mark.inference(concern="recovery")]


def test_recovery_and_ess_share_target_coordinates_and_scales(monkeypatch):
    monkeypatch.setattr(
        extraction,
        "RECOVERY_TARGETS",
        {
            "scalar": RecoveryTargetSpec(
                ParameterCoordinate(site_name="scalar_site", indices=()), 2.5, 2.5
            ),
            "matrix": RecoveryTargetSpec(
                ParameterCoordinate(site_name="matrix_site", indices=(1, 0)), 10.0, 2.0
            ),
            "missing": RecoveryTargetSpec(
                ParameterCoordinate(site_name="absent_site", indices=()), 0.0, 1.0
            ),
        },
    )
    mcmc = TrajectoryMCMCResult(
        chain_samples={
            "scalar_site": jnp.array([[1.0, 2.0, 3.0, 4.0]]),
            "matrix_site": jnp.arange(16.0).reshape(1, 4, 2, 2),
        },
        chain_extra_fields={},
        num_chains=1,
        num_samples=4,
    )
    result = ParticleMCMCPosterior.from_run(
        draws=JointPosteriorDraws(mcmc.get_samples()),
        diagnostics=ProductionDiagnostics(
            mcmc=mcmc, observation_log_probs=jnp.zeros((mcmc.num_chains, mcmc.num_samples, 0))
        ),
    )

    recovery = extraction.parameter_recovery(result, elapsed_seconds=2.0)
    ess = extraction.scalar_posterior_ess(result, max_sites=3, elapsed_seconds=2.0)

    assert recovery["site_count"] == ess["site_count"] == 2
    assert recovery["missing_targets"] == {"missing": {"site": "absent_site", "index": ()}}
    assert recovery["sites"]["scalar"]["mean"] == 2.5
    assert recovery["sites"]["matrix"]["mean"] == 8.0
    assert recovery["sites"]["matrix"]["scale_adjusted_abs_error"] == 1.0
    for label, row in recovery["sites"].items():
        assert row["mean"] == ess["sites"][label]["mean"]
        assert row["ess_approx"] == ess["sites"][label]["ess_approx"]
