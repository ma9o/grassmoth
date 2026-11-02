"""Compile the shared conditional particle target for one or several parameter labels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.models.ssm.inference.samplers.particle.contract import SmootherSetup
from grassmoth.models.ssm.inference.samplers.particle.diagnostics import (
    build_particle_diagnostic_flags,
)

if TYPE_CHECKING:
    from collections.abc import Collection

    import jax

    from grassmoth.artifacts.sampling import ParticleDiagnosticMetric
    from grassmoth.models.ssm.inference.targets.particle import ParticleTarget


def build_smoother_setup(
    target: ParticleTarget,
    *,
    num_parameter_particles: int,
    free_mask: jax.Array,
    diagnostic_metrics: Collection[ParticleDiagnosticMetric],
) -> SmootherSetup:
    """Bind the exact scientific target for every parameter label."""
    return SmootherSetup(
        latent_context_runtime_fn=target.context,
        log_prior_unc_fn=target.log_prior,
        trajectory_log_prob_fn=target.path_log_prob,
        runtime_observations=target.observations,
        runtime_times=target.times,
        latent_free_mask=free_mask,
        num_parameter_particles=num_parameter_particles,
        diagnostic_weights=build_particle_diagnostic_flags(
            diagnostic_metrics=diagnostic_metrics
        ).particle_weights,
    )
