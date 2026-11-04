"""Builds the per-step :class:`SmootherContext` for MPGibbs latent smoothers.

Reference: Corenflos (2025), https://arxiv.org/abs/2505.04611. The per-step posterior
mixture over the parameter ensemble that the smoothers condition on.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from grassmoth.models.ssm.inference.samplers.particle.contract import (
    SmootherContext,
    SmootherSetup,
    TrajectoryLabelScores,
)
from grassmoth.models.ssm.inference.samplers.particle.math import _normalize_log_probs
from grassmoth.models.ssm.inference.targets.particle import ParticleContext
from grassmoth.models.ssm.shapes import Array, Float


def build_smoother_context(
    static: SmootherSetup,
    parameter_particles: Float[Array, "K U"],
    label_correction: Float[Array, " K"],
    *,
    reference: Float[Array, "T D"],
    delta: Float[Array, " T"] | Float[Array, "T D"],
    trajectory_dtype: DTypeLike,
    label_factor: Callable[[ParticleContext], jax.Array] | None = None,
) -> SmootherContext:
    """Assemble per-step latent contexts and shared smoother helper closures."""
    latent_context_runtime_fn = static.latent_context_runtime_fn
    log_prior_unc_fn = static.log_prior_unc_fn
    trajectory_log_prob_fn = static.trajectory_log_prob_fn
    runtime_observations = static.runtime_observations
    runtime_times = static.runtime_times
    num_parameter_particles = static.num_parameter_particles

    latent_dtype = reference.dtype
    traj_dtype = trajectory_dtype
    num_steps = int(reference.shape[0])

    with jax.named_scope("build_contexts"):
        contexts = jax.vmap(lambda z: latent_context_runtime_fn(z, runtime_times))(
            parameter_particles
        )
        parameter_log_priors = jax.vmap(log_prior_unc_fn)(parameter_particles)
        parameter_log_probs = (parameter_log_priors + label_correction).astype(traj_dtype)
        active = jnp.isfinite(parameter_log_probs)
        parameter_log_probs = jnp.where(active, parameter_log_probs, -jnp.inf)
        initial_label_log_probs = _normalize_log_probs(parameter_log_probs)
        # A zero-mass label must not inject NaNs into shared gradients or geometry.
        # Its stored coordinates stay unchanged; this inert context is never selected.
        # The valid reference label guarantees a nonempty support.
        representative = jnp.argmax(active)
        contexts = jax.tree.map(
            lambda leaf: jnp.where(
                active.reshape((-1,) + (1,) * (leaf.ndim - 1)), leaf, leaf[representative]
            ),
            contexts,
        )
        guide_position = jnp.sum(
            jnp.where(active[:, None], parameter_particles, 0.0), axis=0
        ) / jnp.sum(active)

        if label_factor is not None:
            factors = jax.vmap(label_factor)(contexts)
            parameter_log_priors = parameter_log_priors + factors
            parameter_log_probs = parameter_log_probs + factors
            initial_label_log_probs = _normalize_log_probs(parameter_log_probs)

    def _trajectory_log_probs(path: Float[Array, "T D"]) -> Float[Array, " K"]:
        def _one_context(context: ParticleContext) -> jnp.ndarray:
            return trajectory_log_prob_fn(
                context,
                path,
                runtime_observations,
            )

        return jax.vmap(_one_context)(contexts).astype(traj_dtype)

    def _label_scores_from_densities(
        trajectory_log_probs: Float[Array, " K"],
    ) -> TrajectoryLabelScores:
        # The Barker correction belongs only to the label law. Preserve the
        # uncorrected exact densities for the selected chain state's caches.
        return TrajectoryLabelScores(
            _normalize_log_probs(parameter_log_probs + trajectory_log_probs),
            trajectory_log_probs,
            parameter_log_priors + trajectory_log_probs,
        )

    def _score_labels(path: Float[Array, "T D"]) -> TrajectoryLabelScores:
        return _label_scores_from_densities(_trajectory_log_probs(path))

    return SmootherContext(
        contexts=contexts,
        guide_position=guide_position,
        initial_label_log_probs=initial_label_log_probs,
        num_steps=num_steps,
        num_parameter_particles=num_parameter_particles,
        traj_dtype=traj_dtype,
        runtime_observations=runtime_observations,
        latent_free_mask=static.latent_free_mask,
        amala_delta=jnp.asarray(delta, dtype=latent_dtype),
        score_labels=_score_labels,
        label_scores_from_densities=_label_scores_from_densities,
        diagnostic_weights=static.diagnostic_weights,
    )
