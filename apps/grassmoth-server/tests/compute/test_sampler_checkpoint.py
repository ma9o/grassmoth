"""Sampler continuation survives the server checkpoint codec."""

import logging
import time
from datetime import UTC, datetime

import numpy as np
import pytest

from grassmoth.artifacts.continuation import SamplerCheckpoint
from grassmoth.artifacts.sampling import (
    ConditionalNutsSpec,
    GaussianInitializationSpec,
    LapsInitializationSpec,
    MarginalParticleGibbsSpec,
    MetropolisHastingsSpec,
    ParticleGibbsSpec,
    SamplerSpec,
)
from grassmoth.models.ssm.inference.fit import fit
from grassmoth.models.ssm.inference.types import MCMCPosterior
from grassmoth_server.compute.compilation_cache import compile_staged, prepare_program
from grassmoth_server.study.persistence.codec import pack_result, unpack_result
from grassmoth_test_support.compute import COMPUTE_BUDGET
from grassmoth_test_support.inference import nonlinear_fit_fixture

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize(
    ("method", "initializer"),
    [("mh", "laps"), ("nuts", "gaussian"), ("marginal", "laps"), ("marginal", "gaussian")],
)
@pytest.mark.inference(concern="warmup")
def test_particle_continuation_matches_uninterrupted_fit_after_checkpoint_roundtrip(
    method, initializer
):
    prepared, panel = nonlinear_fit_fixture()
    initialization = (
        LapsInitializationSpec(ensemble_size=32, num_unadjusted_steps=100, num_adjusted_steps=5)
        if initializer == "laps"
        else GaussianInitializationSpec()
    )
    algorithm = (
        MarginalParticleGibbsSpec(initialization=initialization)
        if method == "marginal"
        else ParticleGibbsSpec(
            parameter_kernel=MetropolisHastingsSpec()
            if method == "mh"
            else ConditionalNutsSpec(max_tree_depth=3, initial_step_size=0.01),
            initialization=initialization,
        )
    )
    settings = SamplerSpec(
        algorithm=algorithm,
        num_warmup=40 if method == "marginal" else 5,
        num_samples_per_chain=4,
        num_chains=2,
        num_particles=4,
        retain_latent_paths=True,
    )

    def run(options, checkpoint=None):
        result = fit(
            prepared.prior_runtime_bundle,
            panel,
            sampler=options,
            continuation=checkpoint,
            compute_budget=COMPUTE_BUDGET,
            reparam=None,
            clock=time.monotonic,
            compiler=compile_staged,
            prepare_program=prepare_program,
            logger=logging.getLogger(__name__),
            wall_clock=lambda: datetime.now(UTC),
        )
        assert isinstance(result, MCMCPosterior)
        assert result.diagnostics.checkpoint is not None
        return result

    full = run(settings)
    assert full.diagnostics.sampler_diagnostics.initialization.source == initializer
    first = run(settings.revised(num_samples_per_chain=2))
    checkpoint = SamplerCheckpoint.model_validate(
        unpack_result(pack_result(first.diagnostics.checkpoint))
    )
    second = run(settings.revised(num_warmup=0, num_samples_per_chain=2), checkpoint)

    def combined(before, after):
        # Public draws flatten chain first, so append within each chain.
        return np.concatenate(
            (
                np.asarray(before).reshape((2, 2, *before.shape[1:])),
                np.asarray(after).reshape((2, 2, *after.shape[1:])),
            ),
            axis=1,
        ).reshape((8, *before.shape[1:]))

    for name, values in full.draws.parameters.items():
        np.testing.assert_allclose(
            values,
            combined(first.draws.parameters[name], second.draws.parameters[name]),
            rtol=2e-6,
            atol=2e-6,
        )
    assert full.draws.latent_paths is not None
    assert first.draws.latent_paths is not None
    assert second.draws.latent_paths is not None
    np.testing.assert_allclose(
        full.draws.latent_paths,
        combined(first.draws.latent_paths, second.draws.latent_paths),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        full.diagnostics.observation_log_probs,
        np.concatenate(
            (first.diagnostics.observation_log_probs, second.diagnostics.observation_log_probs),
            axis=1,
        ),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_array_equal(second.final_latent_delta, first.final_latent_delta)
    assert second.diagnostics.checkpoint is not None
    assert full.diagnostics.checkpoint is not None
    assert second.diagnostics.checkpoint.random_key == full.diagnostics.checkpoint.random_key
    assert second.diagnostics.checkpoint.tuning == checkpoint.tuning
