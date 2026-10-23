"""Builds the per-step :class:`SmootherContext` for MPGibbs latent smoothers."""

# References:
#   https://arxiv.org/abs/2505.04611 — Corenflos (2025), arXiv:2505.04611:
#     the per-step posterior mixture over the parameter ensemble that the smoothers
#     condition on.
#   Särkkä (2013), Bayesian Filtering and Smoothing,
#     for the Gaussian transition / initial-state moments assembled here.

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.inference.samplers.particle.contract import (
    SmootherContext,
    SmootherSetup,
    TrajectoryLabelScores,
)
from grassmoth.models.ssm.inference.samplers.particle.math import (
    _normalize_log_probs,
    _value_and_grad_by_param,
)

if TYPE_CHECKING:
    from jax.typing import DTypeLike

    from grassmoth.models.ssm.inference.targets.particle import ParticleContext


def build_smoother_context(
    static: SmootherSetup,
    parameter_particles: jnp.ndarray,
    label_correction: jnp.ndarray,
    *,
    reference: jnp.ndarray,
    delta: jnp.ndarray,
    trajectory_dtype: DTypeLike,
) -> SmootherContext:
    """Assemble per-step latent contexts and shared smoother helper closures."""
    latent_context_runtime_fn = static.latent_context_runtime_fn
    log_prior_unc_fn = static.log_prior_unc_fn
    obs_increment_fn = static.obs_increment_fn
    trajectory_log_prob_fn = static.trajectory_log_prob_fn
    runtime_observations = static.runtime_observations
    runtime_times = static.runtime_times
    num_parameter_particles = static.num_parameter_particles
    transition_initial_log_prob_fn = static.transition_initial_log_prob_fn
    transition_value_grad_fn = static.transition_value_grad_fn
    transition_pairwise_log_probs_fn = static.transition_pairwise_log_probs_fn

    x_ref = reference
    latent_dtype = x_ref.dtype
    traj_dtype = trajectory_dtype
    num_steps = int(x_ref.shape[0])
    num_free_particles = static.num_particles - 1

    with jax.named_scope("build_contexts"):
        contexts = jax.vmap(lambda z: latent_context_runtime_fn(z, runtime_times))(
            parameter_particles
        )
        parameter_log_priors = jax.vmap(log_prior_unc_fn)(parameter_particles)
        parameter_log_probs = (parameter_log_priors + label_correction).astype(traj_dtype)
        initial_label_log_probs = _normalize_log_probs(parameter_log_probs)

    def _initial_value_grad_by_param(particle0: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _value_and_grad_by_param(
            contexts,
            particle0,
            transition_initial_log_prob_fn,
            value_dtype=traj_dtype,
            grad_dtype=latent_dtype,
        )

    def _transition_current_value_grad_by_param(
        prev_particle: jnp.ndarray,
        particle_t: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        values, gradients = jax.vmap(
            lambda context: transition_value_grad_fn(
                context, prev_particle, particle_t, time_idx, current_gradient=True
            )
        )(contexts)
        return values.astype(traj_dtype), gradients.astype(latent_dtype)

    def _transition_next_value_grad_by_param(
        particle_t: jnp.ndarray,
        next_particle: jnp.ndarray,
        next_time_idx: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        values, gradients = jax.vmap(
            lambda context: transition_value_grad_fn(
                context, particle_t, next_particle, next_time_idx, current_gradient=False
            )
        )(contexts)
        return values.astype(traj_dtype), gradients.astype(latent_dtype)

    def _pairwise_transition_log_probs(
        prev_particles: jnp.ndarray,
        next_particles: jnp.ndarray,
        seam: jnp.ndarray,
    ) -> jnp.ndarray:
        seam_clamped = jnp.minimum(seam, num_steps - 1)
        real_seam = seam < num_steps
        per_param = jax.vmap(
            lambda context: transition_pairwise_log_probs_fn(
                context,
                prev_particles,
                next_particles,
                seam_clamped,
            )
        )(contexts)
        transition_lp = jnp.moveaxis(per_param, 0, -1)
        return jnp.where(real_seam, transition_lp, 0.0).astype(traj_dtype)

    def _trajectory_log_probs(path: jnp.ndarray) -> jnp.ndarray:
        def _one_context(context: ParticleContext) -> jnp.ndarray:
            return trajectory_log_prob_fn(
                context,
                path,
                runtime_observations,
            )

        return jax.vmap(_one_context)(contexts).astype(traj_dtype)

    def _label_scores_from_densities(trajectory_log_probs: jnp.ndarray) -> TrajectoryLabelScores:
        # The Barker correction belongs only to the label law. Preserve the
        # uncorrected exact densities for the selected chain state's caches.
        return TrajectoryLabelScores(
            _normalize_log_probs(parameter_log_probs + trajectory_log_probs),
            trajectory_log_probs,
            parameter_log_priors + trajectory_log_probs,
        )

    def _score_labels(path: jnp.ndarray) -> TrajectoryLabelScores:
        return _label_scores_from_densities(_trajectory_log_probs(path))

    return SmootherContext(
        contexts=contexts,
        initial_label_log_probs=initial_label_log_probs,
        num_steps=num_steps,
        num_free_particles=num_free_particles,
        num_parameter_particles=num_parameter_particles,
        latent_dtype=latent_dtype,
        traj_dtype=traj_dtype,
        obs_increment_fn=obs_increment_fn,
        runtime_observations=runtime_observations,
        latent_free_mask=static.latent_free_mask,
        amala_delta=jnp.asarray(delta, dtype=latent_dtype),
        drift=static.drift,
        latent_block_coords=static.latent_block_coords,
        paid_mix=static.paid_mix,
        initial_value_grad_by_param=_initial_value_grad_by_param,
        transition_current_value_grad_by_param=_transition_current_value_grad_by_param,
        transition_next_value_grad_by_param=_transition_next_value_grad_by_param,
        pairwise_transition_log_probs=_pairwise_transition_log_probs,
        score_labels=_score_labels,
        label_scores_from_densities=_label_scores_from_densities,
        intervals=static.intervals,
        diagnostic_weights=static.diagnostic_weights,
    )
