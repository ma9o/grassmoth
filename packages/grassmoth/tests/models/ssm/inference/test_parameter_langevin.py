"""Bounded pilot proposals preserve exact label laws and reject undefined densities."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from grassmoth.artifacts.sampling import DsmcSpec, GuidedPathSpec
from grassmoth.models.ssm.inference.parameter_transform import ParameterTransform
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import _initialize_chain_state
from grassmoth.models.ssm.inference.samplers.particle.context import build_smoother_context
from grassmoth.models.ssm.inference.samplers.particle.setup import build_smoother_setup
from grassmoth_test_support.interval_particle import interval_target


def _scale_target(*, bounded_prior=False):
    times = np.arange(5.0)
    target = interval_target(times, np.full((5, 1), 20.0), times[:, None], ("last",))

    def context(position, grid):
        bound = target.context(jnp.zeros(1), grid)
        model = eqx.tree_at(
            lambda value: value.observation_model.measurement.manifest_cov,
            bound[0],
            jnp.exp(2 * position[0]).reshape((1, 1)),
        )
        return model, bound[1], bound[2]

    def prior(position):
        if bounded_prior:
            return dist.Uniform(-1.0, 1.0, validate_args=True).log_prob(position[0])
        return dist.HalfNormal(1.0).log_prob(jnp.exp(position[0])) + position[0]

    return replace(
        target,
        context=context,
        parameters=ParameterTransform(
            jnp.zeros(1), lambda z: {"log_scale": z}, lambda z: {"scale": jnp.exp(z)}, prior
        ),
    )


def _state(target, kernel, position, path, *, step=0.006643351):
    return _initialize_chain_state(
        position,
        observations=target.observations,
        times=target.times,
        target=target,
        parameter_gradient_fn=kernel.parameter_gradient_fn,
        parameter_chol=kernel.initial_parameter_chol,
        initial_latent_delta=jnp.full(target.times.shape, 1e-4),
        param_step_size=step,
        param_min_scale=1e-6,
        param_max_scale=1e3,
        initial_latent_trajectory=path,
    )


@pytest.mark.inference(concern="sampling")
def test_large_stale_pilot_score_moves_parameters_without_poisoning_guided_paths():
    target = _scale_target()
    pilot = jnp.zeros((5, 1))
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_parameter_particles=2,
        parameter_proposal="pseudo_langevin",
        parameter_reference_path=pilot,
        parameter_preconditioner_chol=jnp.array([[0.03]]),
        path_update=GuidedPathSpec(method="twisted_particle_agrad", num_particles=8, kappa=1),
    )
    initial = _state(target, kernel, jnp.array([-4.25]), jnp.full((5, 1), 20.0))
    # The old drift would push a log observation scale beyond floating-point support.
    raw_drift = 2 * initial.param_step_size * 0.03**2 * initial.parameter_gradient
    assert float(raw_drift[0]) > 100

    def step(state, key):
        updated, info = kernel.step_fn(state, key)
        return updated, (updated.position, updated.latent_trajectory, info["final_label_log_probs"])

    final, (positions, paths, weights) = jax.jit(
        lambda state: jax.lax.scan(step, state, jax.random.split(jax.random.key(19), 32))
    )(initial)
    assert np.isfinite(positions).all()
    assert np.isfinite(paths).all()
    assert np.isfinite(weights).all()
    assert np.count_nonzero(np.diff(np.asarray(positions[:, 0]))) >= 2
    assert not np.array_equal(final.latent_trajectory, initial.latent_trajectory)
    complete, trajectory = target.log_posterior_from_context(
        final.position, final.latent_context, final.latent_trajectory, target.observations
    )
    np.testing.assert_allclose(final.complete_log_posterior, complete, rtol=2e-5)
    np.testing.assert_allclose(final.trajectory_log_prob, trajectory, rtol=2e-5)


@pytest.mark.inference(concern="sampling")
def test_bounded_label_correction_equals_the_two_gaussian_log_densities():
    target = interval_target(np.array([0.0]), np.array([[np.nan]]), np.array([[0.0]]), ("last",))
    law = dist.MultivariateNormal(jnp.zeros(2), covariance_matrix=jnp.diag(jnp.array([0.01, 0.04])))
    target = replace(
        target,
        parameters=ParameterTransform(
            jnp.zeros(2), lambda z: {"z": z}, lambda z: {"z": z}, law.log_prob
        ),
    )
    chol = jnp.array([[0.7, 0.0], [0.3, 0.2]])
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_parameter_particles=3,
        parameter_proposal="pseudo_langevin",
        parameter_preconditioner_chol=chol,
        path_update=DsmcSpec(num_particles=4),
    )
    state = _state(target, kernel, jnp.array([0.5, -0.3]), jnp.zeros((1, 1)), step=0.3)
    traced = jax.jit(kernel.trace_step_fn)(state, jax.random.key(93))
    candidates = traced.trace.parameters.candidates
    auxiliary = traced.trace.parameters.auxiliary
    gradient = -np.asarray(candidates) / np.array([0.01, 0.04])
    whitened = np.sqrt(0.6) * gradient @ np.asarray(chol)
    assert (np.linalg.norm(whitened, axis=1) > 1).any()
    bounded = whitened / np.maximum(1.0, np.linalg.norm(whitened, axis=1, keepdims=True))
    means = candidates + np.sqrt(0.6) * bounded @ np.asarray(chol).T
    forward = dist.MultivariateNormal(means, scale_tril=np.sqrt(0.6) * chol).log_prob(auxiliary)
    reverse = dist.MultivariateNormal(auxiliary, scale_tril=np.sqrt(0.6) * chol).log_prob(
        candidates
    )
    expected = jax.nn.log_softmax(jax.vmap(law.log_prob)(candidates) + forward - reverse)
    np.testing.assert_allclose(
        traced.trace.parameters.label_log_prior, expected, atol=3e-5, rtol=2e-5
    )


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(
    "path_update",
    [
        DsmcSpec(num_particles=8),
        GuidedPathSpec(method="twisted_particle_agrad", num_particles=8, kappa=1),
    ],
)
def test_out_of_support_candidates_have_zero_mass_and_leave_a_finite_path_update(path_update):
    target = _scale_target(bounded_prior=True)
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_parameter_particles=8,
        parameter_proposal="pseudo_langevin",
        parameter_preconditioner_chol=jnp.array([[100.0]]),
        path_update=path_update,
    )
    state = _state(target, kernel, jnp.zeros(1), jnp.full((5, 1), 20.0), step=0.3)
    traced = jax.jit(kernel.trace_step_fn)(state, jax.random.key(29))
    updated, info = traced.result
    candidates = traced.trace.parameters.candidates
    outside = np.abs(np.asarray(candidates[:, 0])) > 1
    assert outside.any()
    assert (np.abs(np.asarray(candidates[:, 0])) > 100).any()
    assert np.isneginf(np.asarray(traced.trace.parameters.label_log_prior)[outside]).all()
    assert np.isneginf(np.asarray(info["final_label_log_probs"])[outside]).all()
    assert not np.isnan(info["final_label_log_probs"]).any()
    assert not outside[int(info["selected_label"])]
    assert np.isfinite(updated.position).all()
    assert np.isfinite(updated.parameter_gradient).all()
    assert np.isfinite(updated.latent_trajectory).all()
    assert np.isfinite(updated.complete_log_posterior)
    assert np.isfinite(info["amala_grad_norm_mean"])


@pytest.mark.inference(concern="sampling")
def test_nonfinite_trajectory_densities_have_zero_final_selection_mass():
    target = interval_target(np.array([0.0]), np.array([[np.nan]]), np.array([[0.0]]), ("last",))
    static = build_smoother_setup(
        target,
        DsmcSpec(num_particles=4),
        num_parameter_particles=4,
        free_mask=target.independent_mask,
        paid_mix=None,
        diagnostic_metrics=(),
    )
    context = build_smoother_context(
        static,
        jnp.zeros((4, 1)),
        jnp.zeros(4),
        reference=jnp.zeros((1, 1)),
        delta=jnp.ones(1),
        trajectory_dtype=jnp.float32,
    )
    scores = jax.jit(context.label_scores_from_densities)(
        jnp.array([0.0, jnp.nan, jnp.inf, -jnp.inf])
    )
    np.testing.assert_array_equal(scores.log_probs, jnp.array([0.0, -jnp.inf, -jnp.inf, -jnp.inf]))
