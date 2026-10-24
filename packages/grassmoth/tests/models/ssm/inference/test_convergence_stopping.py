"""Declared stopping composes the production gate without altering either exact engine."""

import logging
from datetime import UTC, datetime
from time import monotonic

import numpy as np
import pytest

from grassmoth.artifacts.identity import ParameterRef
from grassmoth.artifacts.reports.checks import Evaluated, NumericCriterionEvidence
from grassmoth.artifacts.reports.posterior_diagnostics import ConvergenceReport
from grassmoth.artifacts.sampling import (
    ConvergenceStoppingSpec,
    MarginalParticleGibbsSpec,
    NutsDASpec,
    NutsTrajectorySpec,
    SamplerSpec,
    TwistedParticleAGradSpec,
)
from grassmoth.models.ssm.inference.diagnostics import convergence
from grassmoth.models.ssm.inference.fit import fit
from grassmoth.models.ssm.inference.types import MCMCPosterior
from grassmoth_test_support.compute import COMPUTE_BUDGET, compile_program, prepare_program
from grassmoth_test_support.inference import nonlinear_fit_fixture


@pytest.mark.contract
def test_stopping_spec_is_opt_in_and_has_positive_checkpoints():
    assert SamplerSpec().stop_when_converged is None
    for interval, minimum in ((0, 1), (1, 0), (-1, 5)):
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            ConvergenceStoppingSpec(checkpoint_draws=interval, min_draws=minimum)
    # A minimum above the ceiling, and zero-draw preflight, simply cannot stop early.
    settings = SamplerSpec(
        num_samples_per_chain=0,
        stop_when_converged={
            "checkpoint_draws": 5,
            "min_draws": 10,
        },
    )
    assert settings.stop_when_converged == ConvergenceStoppingSpec(checkpoint_draws=5, min_draws=10)


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(
    "algorithm",
    [
        MarginalParticleGibbsSpec(path_update=TwistedParticleAGradSpec(num_particles=8)),
        NutsDASpec(trajectory=NutsTrajectorySpec(max_tree_depth=2)),
    ],
)
def test_native_checks_and_stopping_keep_nonlinear_draws_and_frozen_tuning(algorithm, monkeypatch):
    prepared, panel = nonlinear_fit_fixture()
    settings = SamplerSpec(
        algorithm=algorithm, num_warmup=40, num_samples_per_chain=12, num_chains=2, seed=19
    )

    def execute(policy):
        result = fit(
            prepared.prior_runtime_bundle,
            panel,
            sampler=settings.revised(stop_when_converged=policy),
            compute_budget=COMPUTE_BUDGET,
            reparam=None,
            clock=monotonic,
            compiler=compile_program,
            prepare_program=prepare_program,
            logger=logging.getLogger(__name__),
            wall_clock=lambda: datetime.now(UTC),
        )
        assert isinstance(result, MCMCPosterior)
        return result

    full = execute(None)
    native = convergence.chain_convergence
    reports = []

    def observe(chains):
        report = native(chains)
        reports.append((chains.num_samples_per_chain, report))
        return report

    monkeypatch.setattr(convergence, "chain_convergence", observe)
    checked = execute(ConvergenceStoppingSpec(checkpoint_draws=1, min_draws=20))
    assert [draws for draws, _ in reports] == list(range(1, 13))
    assert reports[0][1].status != "passed"
    references = {
        **{
            coordinate: (
                binding.elements[element],
                ParameterRef(parameter_id=binding.parameter_id, element_id=element),
            )
            for binding in panel.compiled_dynamical_model.bindings
            for element, coordinate in binding.coordinates.items()
        },
        **dict.fromkeys(panel.compiled_dynamical_model.auxiliary_coordinates),
    }
    assert reports[-1][1] == native(checked.get_inference_diagnostics(references))
    np.testing.assert_array_equal(full.draws.latent_paths, checked.draws.latent_paths)
    for name, value in full.get_samples().items():
        np.testing.assert_array_equal(value, checked.get_samples()[name])

    # Force an eligible gate result to exercise truncation in both concrete adapters.
    monkeypatch.setattr(
        convergence,
        "chain_convergence",
        lambda _chains: ConvergenceReport(
            findings=(
                Evaluated(
                    code="test_gate",
                    subject="recorded_parameter_chains",
                    outcome="passed",
                    evidence=NumericCriterionEvidence(criterion="test", value=1.0),
                ),
            )
        ),
    )
    stopped = execute(ConvergenceStoppingSpec(checkpoint_draws=4, min_draws=8))
    assert stopped.diagnostics.mcmc.num_samples == 8
    assert stopped.draws.describe().n_draws == 16
    np.testing.assert_array_equal(
        stopped.draws.latent_paths.reshape(2, 8, *stopped.draws.latent_paths.shape[1:]),
        full.draws.latent_paths.reshape(2, 12, *full.draws.latent_paths.shape[1:])[:, :8],
    )
    for name, value in full.diagnostics.mcmc.get_samples(group_by_chain=True).items():
        np.testing.assert_array_equal(
            value[:, :8], stopped.diagnostics.mcmc.get_samples(group_by_chain=True)[name]
        )
    for phase in ("warmup", "post_warmup"):
        for name, value in full.diagnostics.phase_extra_fields[phase].items():
            expected = value if phase == "warmup" else value[:, :8]
            np.testing.assert_array_equal(
                expected, stopped.diagnostics.phase_extra_fields[phase][name]
            )

    assert stopped.diagnostics.checkpoint is not None
    resumed = fit(
        prepared.prior_runtime_bundle,
        panel,
        sampler=settings.revised(num_warmup=0, num_samples_per_chain=4),
        continuation=stopped.diagnostics.checkpoint,
        compute_budget=COMPUTE_BUDGET,
        reparam=None,
        clock=monotonic,
        compiler=compile_program,
        prepare_program=prepare_program,
        logger=logging.getLogger(__name__),
        wall_clock=lambda: datetime.now(UTC),
    )
    assert isinstance(resumed, MCMCPosterior)
    assert resumed.draws.latent_paths is not None
    assert resumed.diagnostics.checkpoint is not None
    assert resumed.diagnostics.mcmc.num_samples == 4
    np.testing.assert_array_equal(
        resumed.draws.latent_paths.reshape(2, 4, *resumed.draws.latent_paths.shape[1:]),
        full.draws.latent_paths.reshape(2, 12, *full.draws.latent_paths.shape[1:])[:, 8:],
    )
    for name, value in full.diagnostics.mcmc.get_samples(group_by_chain=True).items():
        np.testing.assert_array_equal(
            value[:, 8:], resumed.diagnostics.mcmc.get_samples(group_by_chain=True)[name]
        )
    np.testing.assert_array_equal(
        full.diagnostics.checkpoint.random_key.values,
        resumed.diagnostics.checkpoint.random_key.values,
    )
