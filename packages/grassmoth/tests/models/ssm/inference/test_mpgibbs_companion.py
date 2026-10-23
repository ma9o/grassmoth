"""Companion moves are exact Metropolis transitions with their own warmup-adapted steps."""

import logging
import time
from datetime import UTC, datetime

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from grassmoth.artifacts.sampling import (
    GaussianInitializationSpec,
    GaussianReferenceTemperingSpec,
    MarginalParticleGibbsSpec,
    SamplerSpec,
)
from grassmoth.models.ssm.inference.joint_coordinates import build_joint_coordinates
from grassmoth.models.ssm.inference.samplers.mpgibbs.companion import (
    build_companion_kernel,
    build_companion_proposal,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import (
    _initialize_chain_state,
    run_marginal_particle_gibbs,
)
from grassmoth.models.ssm.inference.warmup.initialization import (
    ParticleInitialization,
    prepare_particle_initialization,
)
from grassmoth_test_support.compute import COMPUTE_BUDGET, compile_program, prepare_program
from packages.grassmoth.tests.models.ssm.inference.test_joint_coordinates import (
    _correlated_target,
)


@pytest.mark.contract
def test_companion_choice_identifies_the_kernel_and_excludes_tempering():
    identities = {
        SamplerSpec(algorithm=MarginalParticleGibbsSpec(companion=choice)).kernel_identity
        for choice in ("none", "centred_mh", "centred_and_scale_noncentred_mh")
    }
    assert len(identities) == 3
    with pytest.raises(ValidationError, match="tempering"):
        MarginalParticleGibbsSpec(
            companion="centred_mh", tempering=GaussianReferenceTemperingSpec()
        )


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("name", ["centred_mh", "scale_noncentred_mh"])
def test_companion_proposals_are_reversible_with_exact_ratios_and_refreshed_state(name):
    with jax.enable_x64():
        target = _correlated_target()
        coordinates = build_joint_coordinates(target, "non_centred")
        position, path = coordinates.decode(jnp.linspace(-0.3, 0.4, coordinates.dimension))
        initial = _initialize_chain_state(
            position,
            observations=target.observations,
            times=target.times,
            target=target,
            parameter_gradient_fn=jnp.sin,
            initial_latent_delta=jnp.full(target.times.shape, 0.01),
            param_step_size=0.02,
            parameter_chol=jnp.eye(position.size),
            param_min_scale=1e-6,
            param_max_scale=1e3,
            initial_latent_trajectory=path,
            exact_constraints=target.exact_constraints,
        )
        propose = jax.jit(build_companion_proposal(target, name, jnp.sin))
        proposed, ratio = propose(initial, position + 0.04)
        returned, reverse = propose(proposed, position)
        np.testing.assert_allclose(ratio + reverse, 0, atol=1e-9)
        np.testing.assert_allclose(returned.latent_trajectory, initial.latent_trajectory, atol=1e-9)
        assert proposed.latent_trajectory[1, 0] == 0.3
        np.testing.assert_allclose(proposed.parameter_gradient, jnp.sin(proposed.position))
        exact = target.log_posterior(
            proposed.position, proposed.latent_trajectory, target.observations, target.times
        )
        np.testing.assert_allclose(proposed.complete_log_posterior, exact, atol=1e-9)
        if name == "centred_mh":
            np.testing.assert_array_equal(proposed.latent_trajectory, initial.latent_trajectory)
            np.testing.assert_allclose(ratio, exact - initial.complete_log_posterior, atol=1e-9)
            return

        def scientific(point):
            parameters, trajectory = coordinates.decode(point)
            return jnp.concatenate((parameters, trajectory.ravel()[coordinates.free_indices]))

        old_point = coordinates.encode(initial.position, initial.latent_trajectory)
        new_point = coordinates.encode(proposed.position, proposed.latent_trajectory)
        _, old_jacobian = jnp.linalg.slogdet(jax.jacfwd(scientific)(old_point))
        _, new_jacobian = jnp.linalg.slogdet(jax.jacfwd(scientific)(new_point))
        np.testing.assert_allclose(
            ratio,
            exact - initial.complete_log_posterior + new_jacobian - old_jacobian,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            old_point[position.size :], new_point[position.size :], atol=1e-9
        )


@pytest.mark.inference(concern="sampling")
def test_each_companion_adapts_its_own_step_in_warmup_and_freezes_it():
    target = _correlated_target()
    assert [
        move.name
        for move in build_companion_kernel(
            target,
            "centred_and_scale_noncentred_mh",
            diffusion_mask=jnp.zeros(1, dtype=bool),
            parameter_gradient=jnp.zeros_like,
        ).moves
    ] == ["centred_mh"]
    initialization = prepare_particle_initialization(
        target,
        spec=GaussianInitializationSpec(),
        key=jax.random.key(3),
        num_chains=2,
        compute_budget=COMPUTE_BUDGET,
    )
    assert isinstance(initialization, ParticleInitialization)
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_particles=4,
        num_parameter_particles=2,
        param_step_size=0.02,
        parameter_preconditioner_chol=initialization.parameter_chol,
        exact_constraints=target.exact_constraints,
    )
    result = run_marginal_particle_gibbs(
        target,
        kernel=kernel,
        companion=build_companion_kernel(
            target,
            "centred_and_scale_noncentred_mh",
            diffusion_mask=jnp.ones(1, dtype=bool),
            parameter_gradient=kernel.parameter_gradient_fn,
        ),
        compiler=compile_program,
        prepare_program=prepare_program,
        num_warmup=60,
        num_samples=20,
        num_chains=2,
        key=jax.random.key(11),
        adaptation_rate=0.05,
        initialization=initialization,
        compute_budget=COMPUTE_BUDGET,
        clock=time.monotonic,
        wall_clock=lambda: datetime.now(UTC),
        logger=logging.getLogger(__name__),
    )
    for move in ("centred_mh", "scale_noncentred_mh"):
        warmup = np.asarray(result.warmup_chain_extra_fields[f"{move}_step_size"])
        retained = np.asarray(result.chain_extra_fields[f"{move}_step_size"])
        assert warmup.shape == (2, 60)
        assert np.all(np.ptp(warmup, axis=1) > 0)
        np.testing.assert_array_equal(retained, np.repeat(retained[:, :1], 20, axis=1))
        accepted = np.asarray(result.all_chain_extra_fields[f"{move}_accept_prob"])
        assert set(np.unique(accepted)) <= {0.0, 1.0}
        assert np.all(np.isfinite(result.all_chain_extra_fields[f"{move}_jump_rms"]))
    assert not np.array_equal(
        result.warmup_chain_extra_fields["centred_mh_step_size"],
        result.warmup_chain_extra_fields["scale_noncentred_mh_step_size"],
    )
