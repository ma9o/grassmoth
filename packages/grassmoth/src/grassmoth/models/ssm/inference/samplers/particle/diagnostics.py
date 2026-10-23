"""Optional diagnostic metric groups shared by the particle samplers."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from grassmoth.compute.programs import scalar_operand

if TYPE_CHECKING:
    from collections.abc import Collection

    import jax

    from grassmoth.artifacts.sampling import ParticleDiagnosticMetric


class ParticleDiagnosticFlags(NamedTuple):
    """Runtime switches for optional particle diagnostic traces."""

    particle_identity: jax.Array
    parameter_movement: jax.Array
    particle_weights: jax.Array


def build_particle_diagnostic_flags(
    *,
    diagnostic_metrics: Collection[ParticleDiagnosticMetric],
) -> ParticleDiagnosticFlags:
    """Build runtime booleans before tracing the sampler."""
    return ParticleDiagnosticFlags(
        particle_identity=scalar_operand(value="particle_identity" in diagnostic_metrics),
        parameter_movement=scalar_operand(value="parameter_movement" in diagnostic_metrics),
        particle_weights=scalar_operand(value="particle_weights" in diagnostic_metrics),
    )


def excluded_particle_diagnostics(
    diagnostic_metrics: Collection[ParticleDiagnosticMetric],
) -> frozenset[str]:
    """Select retained histories on the host without changing the compiled outputs."""
    from grassmoth.artifacts.reports.posterior_diagnostics import ParticleWeightMetric

    groups: dict[ParticleDiagnosticMetric, tuple[str, ...]] = {
        "particle_identity": (
            "selected_particle_per_t",
            "reference_path_hit_rate",
            "selected_particle_unique_count",
        ),
        "parameter_movement": ("parameter_jump_rms",),
        "particle_weights": tuple(metric.history_key for metric in ParticleWeightMetric),
    }
    return frozenset(
        name
        for metric, names in groups.items()
        if metric not in diagnostic_metrics
        for name in names
    )
