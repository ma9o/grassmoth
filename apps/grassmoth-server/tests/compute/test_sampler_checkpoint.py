"""Sampler continuation survives the server checkpoint codec."""

from dataclasses import replace

import numpy as np
import pytest

from grassmoth.artifacts.continuation import SamplerCheckpoint
from grassmoth.artifacts.sampling import (
    GuidedMethodSpec,
    MarginalParticleGibbsSpec,
    NutsDASpec,
    NutsTrajectorySpec,
    PathUpdateSpec,
    SamplerSpec,
)
from grassmoth.models.ssm.inference.fit import fit
from grassmoth.models.ssm.inference.types import MCMCPosterior
from grassmoth_server.compute.compilation_cache import compile_program, prepare_program
from grassmoth_server.study.persistence.codec import pack_result, unpack_result
from grassmoth_test_support.compute import EXECUTION_CONTEXT
from grassmoth_test_support.inference import nonlinear_fit_fixture
from grassmoth_test_support.values import revise_fixture

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize(
    "algorithm",
    [
        MarginalParticleGibbsSpec(
            path_update=PathUpdateSpec(
                method=GuidedMethodSpec(kind="particle_mgrad", num_particles=4)
            )
        ),
        MarginalParticleGibbsSpec(
            path_update=PathUpdateSpec(
                method=GuidedMethodSpec(kind="twisted_particle_agrad", num_particles=4)
            )
        ),
        NutsDASpec(parameterization="mixed", trajectory=NutsTrajectorySpec(max_tree_depth=3)),
        pytest.param(
            NutsDASpec(
                smoother=PathUpdateSpec(method=GuidedMethodSpec(num_particles=4)),
                parameterization="non_centred",
                trajectory=NutsTrajectorySpec(max_tree_depth=3),
            ),
            id="nuts-smoother",
        ),
    ],
)
@pytest.mark.inference(concern="warmup")
def test_continuation_matches_uninterrupted_fit_after_checkpoint_roundtrip(algorithm):
    prepared, panel = nonlinear_fit_fixture()
    settings = SamplerSpec(
        num_warmup=40,
        num_samples_per_chain=4,
        num_chains=2,
        algorithm=algorithm,
    )

    def run(options, checkpoint=None):
        result = fit(
            prepared.prior_runtime_bundle,
            panel,
            sampler=options,
            continuation=checkpoint,
            execution_context=replace(
                EXECUTION_CONTEXT, compiler=compile_program, prepare_program=prepare_program
            ),
        )
        assert isinstance(result, MCMCPosterior)
        assert result.diagnostics.checkpoint is not None
        return result

    full = run(settings)
    assert full.diagnostics.sampler_diagnostics is not None
    assert full.diagnostics.sampler_diagnostics.method == algorithm.method
    first = run(revise_fixture(settings, num_samples_per_chain=2))
    checkpoint = SamplerCheckpoint.model_validate(
        unpack_result(pack_result(first.diagnostics.checkpoint))
    )
    second = run(revise_fixture(settings, num_warmup=0, num_samples_per_chain=2), checkpoint)

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
