"""Companion moves are exact parameter transitions with their own warmup-adapted steps."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.artifacts.sampling import (
    DsmcSpec,
    GaussianInitializationSpec,
    GuidedPathSpec,
    MarginalParticleGibbsSpec,
    SamplerSpec,
    UntemperedSpec,
)
from grassmoth.models.ssm.inference.joint_coordinates import build_joint_coordinates
from grassmoth.models.ssm.inference.samplers.mpgibbs.companion import (
    build_companion_kernel,
    build_companion_proposal,
    build_conditional_nuts,
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
from grassmoth_test_support.compute import EXECUTION_CONTEXT, compile_program
from grassmoth_test_support.interval_particle import gaussian_interval_case
from packages.grassmoth.tests.models.ssm.inference.test_joint_coordinates import (
    _correlated_target,
)


@pytest.mark.contract
def test_companion_choice_identifies_the_kernel():
    identities = {
        SamplerSpec(
            algorithm=MarginalParticleGibbsSpec(tempering=UntemperedSpec(companion=choice))
        ).kernel_identity
        for choice in ("none", "centred_mh", "centred_and_scale_noncentred_mh", "centred_nuts")
    }
    assert len(identities) == 4


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("name", ["centred_mh", "scale_noncentred_mh"])
def test_companion_proposals_are_reversible_with_exact_ratios_and_refreshed_state(name):
    target = _correlated_target()
    coordinates = build_joint_coordinates(target, "non_centred", compiler=compile_program)
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
    propose = jax.jit(build_companion_proposal(target, name, jnp.sin, compiler=compile_program))
    proposed, ratio = propose(initial, position + 0.04)
    returned, reverse = propose(proposed, position)
    np.testing.assert_allclose(ratio + reverse, 0, atol=1e-5)
    np.testing.assert_allclose(returned.latent_trajectory, initial.latent_trajectory, atol=1e-5)
    assert proposed.latent_trajectory[1, 0] == 0.3
    np.testing.assert_allclose(proposed.parameter_gradient, jnp.sin(proposed.position))
    exact = target.log_posterior(
        proposed.position, proposed.latent_trajectory, target.observations, target.times
    )
    np.testing.assert_allclose(proposed.complete_log_posterior, exact, atol=1e-5)
    if name == "centred_mh":
        np.testing.assert_array_equal(proposed.latent_trajectory, initial.latent_trajectory)
        np.testing.assert_allclose(ratio, exact - initial.complete_log_posterior, atol=1e-5)
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
        atol=1e-5,
    )
    np.testing.assert_allclose(old_point[position.size :], new_point[position.size :], atol=1e-5)


@pytest.mark.inference(concern="sampling")
@pytest.mark.inference(concern="recovery")
def test_centred_nuts_preserves_the_joint_posterior_and_freezes_its_adaptation():
    target, mean, covariance = gaussian_interval_case()
    warmup, retained = 250, 1500
    nuts = build_conditional_nuts(
        target,
        build_companion_proposal(target, "centred_mh", jnp.zeros_like, compiler=compile_program),
    )
    center = jnp.asarray(mean)
    loading = jnp.asarray(covariance[1:, 0] / covariance[0, 0])
    conditional_chol = jnp.linalg.cholesky(
        jnp.asarray(covariance[1:, 1:])
        - jnp.outer(jnp.asarray(covariance[1:, 0]), jnp.asarray(covariance[0, 1:]))
        / covariance[0, 0]
    )

    def chain(key):
        init_key, run_key = jax.random.split(key)
        state = _initialize_chain_state(
            center[:1],
            observations=target.observations,
            times=target.times,
            target=target,
            parameter_gradient_fn=jnp.zeros_like,
            initial_latent_delta=jnp.full(target.times.shape, 0.1),
            param_step_size=0.02,
            parameter_chol=jnp.sqrt(jnp.asarray(covariance[:1, :1])),
            param_min_scale=1e-6,
            param_max_scale=1e3,
            initial_latent_trajectory=center[1:, None],
            exact_constraints=target.exact_constraints,
        )

        def update(carry, step_key):
            state, tuning = carry
            theta_key, path_key = jax.random.split(step_key)
            state, tuning, record = nuts.step(state, tuning, theta_key)
            # Independently known x|theta completes an exact blocked Gibbs sweep.
            # Its changing path catches a stale NUTS energy/gradient cache.
            path = (
                center[1:]
                + loading * (state.position[0] - center[0])
                + conditional_chol @ jax.random.normal(path_key, (len(mean) - 1,))
            )[:, None]
            draw = jnp.concatenate((state.position, path[:, 0]))
            return (state._replace(latent_trajectory=path), tuning), (
                draw,
                record["centred_nuts_step_size"],
            )

        _, (draws, scales) = jax.lax.scan(
            update,
            (state, nuts.initialize(state, init_key, warmup)),
            jax.random.split(run_key, warmup + retained),
        )
        return draws[warmup:], scales[warmup:]

    draws, scales = jax.jit(jax.vmap(chain))(jax.random.split(jax.random.key(348), 4))
    whitened = np.linalg.solve(
        np.linalg.cholesky(covariance), (np.asarray(draws).reshape(-1, len(mean)) - mean).T
    ).T
    np.testing.assert_allclose(whitened.mean(axis=0), 0.0, atol=0.12)
    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(len(mean)), atol=0.15)
    np.testing.assert_array_equal(scales, np.broadcast_to(scales[:, :1], scales.shape))


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(
    ("companion", "moves", "path_update"),
    [
        (
            "centred_and_scale_noncentred_mh",
            ("centred_mh", "scale_noncentred_mh"),
            DsmcSpec(num_particles=4),
        ),
        (
            "centred_mh",
            ("centred_mh",),
            GuidedPathSpec(method="twisted_particle_agrad", num_particles=4),
        ),
        ("centred_nuts", ("centred_nuts",), DsmcSpec(num_particles=4)),
    ],
)
def test_each_companion_adapts_its_own_step_in_warmup_and_freezes_it(companion, moves, path_update):
    target = _correlated_target()
    assert [
        move.name
        for move in build_companion_kernel(
            target,
            "centred_and_scale_noncentred_mh",
            diffusion_mask=jnp.zeros(1, dtype=bool),
            parameter_gradient=jnp.zeros_like,
            compiler=compile_program,
        ).moves
    ] == ["centred_mh"]
    initialization = prepare_particle_initialization(
        target,
        spec=GaussianInitializationSpec(),
        key=jax.random.key(3),
        num_chains=2,
        execution_context=EXECUTION_CONTEXT,
    )
    assert isinstance(initialization, ParticleInitialization)
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        path_update=path_update,
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
            companion,
            diffusion_mask=jnp.ones(1, dtype=bool),
            parameter_gradient=kernel.parameter_gradient_fn,
            compiler=compile_program,
        ),
        sampler=SamplerSpec(num_warmup=60, num_samples_per_chain=20, num_chains=2),
        key=jax.random.key(11),
        initialization=initialization,
        execution_context=EXECUTION_CONTEXT,
    )
    for move in moves:
        warmup = np.asarray(result.warmup_chain_extra_fields[f"{move}_step_size"])
        retained = np.asarray(result.chain_extra_fields[f"{move}_step_size"])
        assert warmup.shape == (2, 60)
        assert np.all(np.ptp(warmup, axis=1) > 0)
        np.testing.assert_array_equal(retained, np.repeat(retained[:, :1], 20, axis=1))
        accepted = np.asarray(result.all_chain_extra_fields[f"{move}_accept_prob"])
        # Metropolis moves record acceptance indicators; NUTS its acceptance statistic.
        if move != "centred_nuts":
            assert set(np.unique(accepted)) <= {0.0, 1.0}
        assert np.all(np.isfinite(result.all_chain_extra_fields[f"{move}_jump_rms"]))
    if companion == "centred_nuts":
        assert result.chain_extra_fields["diverging"].shape == (2, 20)
        assert np.all(np.asarray(result.chain_extra_fields["tree_depth"]) >= 1)
        return
    if len(moves) == 1:
        return
    assert not np.array_equal(
        result.warmup_chain_extra_fields["centred_mh_step_size"],
        result.warmup_chain_extra_fields["scale_noncentred_mh_step_size"],
    )
