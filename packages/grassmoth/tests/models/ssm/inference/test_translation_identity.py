"""Retained physical references survive finite-precision chart reconstruction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.artifacts.sampling import GuidedPathSpec
from grassmoth.models.ssm.inference.samplers.mpgibbs import kernel as kernel_module
from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import _initialize_chain_state
from grassmoth.models.ssm.inference.samplers.particle.contract import ParticleSmootherResult
from grassmoth.models.ssm.inference.targets.coordinates import TranslationChart
from grassmoth_test_support.compute import compile_program
from grassmoth_test_support.interval_particle import initialize_interval_chain, interval_target

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize(("offset", "value"), [(1e4, 0.1234567), (1e-3, 1.234567e-8)])
@pytest.mark.parametrize("move_one_coordinate", [False, True])
def test_retained_chart_coordinates_preserve_physical_identity_and_caches(
    monkeypatch, offset, value, move_one_coordinate
):
    """Isolate reconstruction from the particle-selection and parameter-selection laws."""
    times = np.arange(2.0)
    target = interval_target(times, np.zeros((2, 1)), times[:, None], ("last",), state_dim=2)
    chart = TranslationChart(jnp.full((2, 1), offset), jnp.zeros(1))
    position = jnp.ones(1)
    physical = jnp.array([[value, -value], [-value, value]])
    # Both scales genuinely lose physical bits under an ordinary float32 round trip.
    rounded = np.asarray(chart.decode(position, chart.encode(position, physical)))
    assert np.any(rounded != np.asarray(physical))

    def selected_path(_target, context, _guide, _spec, _key, reference, **_kwargs):
        selected = reference.at[1, 1].add(0.125) if move_one_coordinate else reference
        scores = context.score_labels(selected)
        # Fix the parameter label to isolate numerical reference reconstruction.
        scores = scores._replace(log_probs=jnp.array([0.0, -jnp.inf]))
        proposed = jnp.broadcast_to(jnp.array([False, True]), reference.shape)
        return ParticleSmootherResult(selected, scores, jnp.zeros(2, dtype=jnp.int32), proposed, {})

    monkeypatch.setattr(kernel_module, "guided_step", selected_path)
    kernel = kernel_module.build_marginal_particle_gibbs_kernel(
        target,
        num_parameter_particles=2,
        path_update=GuidedPathSpec(method="particle_agrad", num_particles=2),
        initial_chart=chart,
    )
    state = _initialize_chain_state(
        position,
        observations=target.observations,
        times=target.times,
        target=target,
        parameter_gradient_fn=kernel.parameter_gradient_fn,
        parameter_chol=kernel.initial_parameter_chol,
        initial_latent_delta=jnp.ones(2),
        param_step_size=0.1,
        param_min_scale=1e-6,
        param_max_scale=1.0,
        initial_latent_trajectory=physical,
        chart=chart,
    )
    updated, diagnostics = jax.jit(kernel.step_fn)(state, jax.random.key(731))
    np.testing.assert_array_equal(updated.position, position)
    np.testing.assert_array_equal(updated.latent_trajectory[:, 0], physical[:, 0])
    np.testing.assert_array_equal(updated.latent_trajectory[0], physical[0])
    if move_one_coordinate:
        assert float(diagnostics["latent_frozen_frac"]) == 0.75
        assert float(diagnostics["latent_move_rms"]) > 0
    else:
        np.testing.assert_array_equal(updated.latent_trajectory, physical)
        assert float(diagnostics["latent_frozen_frac"]) == 1.0
        assert float(diagnostics["latent_move_rms"]) == 0.0
        assert float(diagnostics["latent_move_max_abs"]) == 0.0
        np.testing.assert_array_equal(diagnostics["latent_accepted"], 0)
    complete, trajectory = target.log_posterior_from_context(
        updated.position, updated.latent_context, updated.latent_trajectory, target.observations
    )
    np.testing.assert_array_equal(updated.complete_log_posterior, complete)
    np.testing.assert_array_equal(updated.trajectory_log_prob, trajectory)


def test_charted_profile_components_return_production_physical_paths_and_caches():
    """Component timing executes physical reconstruction and scoring for every chain."""
    times = np.arange(2.0)
    target = interval_target(times, np.array([[0.2], [-0.3]]), times[:, None], ("last",))
    chart = TranslationChart(jnp.array([[-0.85]]), jnp.array([0.3]))
    kernel = kernel_module.build_marginal_particle_gibbs_kernel(
        target,
        num_parameter_particles=2,
        path_update=GuidedPathSpec(method="twisted_particle_agrad", num_particles=4, kappa=1),
        initial_chart=chart,
    )
    starts = jnp.array([[0.45, -0.2, 0.15], [-0.6, 0.3, -0.1]])
    states = jax.vmap(lambda joint: initialize_interval_chain(target, joint, kernel=kernel))(starts)
    states = states._replace(
        chart=jax.tree.map(lambda value: jnp.broadcast_to(value, (2, *value.shape)), chart)
    )
    keys = jax.random.split(jax.random.key(318), 2)
    production, _ = jax.jit(jax.vmap(kernel.step_fn))(states, keys)
    profile = kernel.profile_parts(states, keys, 1, compile_program)
    parameter_outputs = jax.block_until_ready(profile.parameter_move())
    path_outputs = jax.block_until_ready(profile.path_smoother())
    assert parameter_outputs
    # The shared reconstruction returns physical path, path density and joint
    # density last; synchronizing these prevents their work being eliminated.
    path, trajectory, complete = path_outputs[-3:]
    np.testing.assert_allclose(path, production.latent_trajectory, atol=2e-6)
    np.testing.assert_allclose(trajectory, production.trajectory_log_prob, atol=2e-6)
    np.testing.assert_allclose(complete, production.complete_log_posterior, atol=2e-6)
