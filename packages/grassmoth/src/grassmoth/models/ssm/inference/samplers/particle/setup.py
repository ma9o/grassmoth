"""Compile the shared conditional dSMC target for one or several parameter labels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.inference.samplers.particle.contract import SmootherSetup
from grassmoth.models.ssm.inference.samplers.particle.diagnostics import (
    build_particle_diagnostic_flags,
)

if TYPE_CHECKING:
    from grassmoth.artifacts.sampling import DsmcSpec
    from grassmoth.models.ssm.inference.samplers.particle.contract import PaidMixture
    from grassmoth.models.ssm.inference.targets.particle import (
        ParticleContext,
        ParticleTarget,
    )


def build_smoother_setup(
    target: ParticleTarget,
    spec: DsmcSpec,
    *,
    num_particles: int,
    num_parameter_particles: int,
    free_mask: jax.Array,
    paid_mix: PaidMixture | None,
    diagnostic_metrics: frozenset[str],
) -> SmootherSetup:
    """Bind local auxiliary factors while retaining the exact scientific target."""
    local_target = target.tree_target()

    def observation_increment(
        context: ParticleContext, state: jax.Array, index: jax.Array, _observations: jax.Array
    ) -> jax.Array:
        if not target.observation_partition.local:
            return jnp.asarray(0.0, dtype=state.dtype)
        return local_target.observation_increment(context, state, index, local_target.observations)

    return SmootherSetup(
        latent_context_runtime_fn=target.context,
        log_prior_unc_fn=target.log_prior,
        obs_increment_fn=observation_increment,
        trajectory_log_prob_fn=target.path_log_prob,
        runtime_observations=target.observations,
        runtime_times=target.times,
        latent_free_mask=free_mask,
        num_particles=num_particles,
        num_parameter_particles=num_parameter_particles,
        drift=spec.drift,
        latent_block_coords=spec.latent_block_coords,
        paid_mix=paid_mix,
        transition_initial_log_prob_fn=local_target.initial_log_prob,
        transition_value_grad_fn=local_target._transition_value_and_grad,
        transition_pairwise_log_probs_fn=local_target.pairwise_transition_log_prob,
        diagnostic_weights=build_particle_diagnostic_flags(
            diagnostic_metrics=diagnostic_metrics
        ).particle_weights,
        intervals=local_target.intervals,
    )
