"""Optional diagnostic metric groups shared by the particle samplers."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

from grassmoth.compute.programs import scalar_operand

if TYPE_CHECKING:
    from collections.abc import Sequence

    import jax


class ParticleDiagnosticMetric(StrEnum):
    """Optional traces: shared dSMC weights, plus marginalized-sampler movement and identity."""

    PARTICLE_IDENTITY = "particle_identity"
    PARAMETER_MOVEMENT = "parameter_movement"
    PARTICLE_WEIGHTS = "particle_weights"


PARTICLE_DIAGNOSTIC_METRIC_VALUES = tuple(metric.value for metric in ParticleDiagnosticMetric)


class ParticleDiagnosticFlags(NamedTuple):
    """Runtime switches for optional particle diagnostic traces."""

    particle_identity: jax.Array
    parameter_movement: jax.Array
    particle_weights: jax.Array


def resolve_particle_diagnostic_metrics(
    *,
    diagnostic_metrics_all: bool,
    diagnostic_metrics: Sequence[str] | None,
) -> frozenset[str]:
    """Resolve optional particle diagnostic metric groups from config/CLI strings."""
    if diagnostic_metrics_all:
        return frozenset(PARTICLE_DIAGNOSTIC_METRIC_VALUES)
    requested = tuple(diagnostic_metrics or ())
    allowed = set(PARTICLE_DIAGNOSTIC_METRIC_VALUES)
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(
            "particle diagnostic_metrics contains unknown values "
            f"{unknown}; allowed: {sorted(allowed)}."
        )
    return frozenset(requested)


def build_particle_diagnostic_flags(
    *,
    diagnostic_metrics: frozenset[str],
) -> ParticleDiagnosticFlags:
    """Build runtime booleans before tracing the sampler."""
    return ParticleDiagnosticFlags(
        particle_identity=scalar_operand(
            value=ParticleDiagnosticMetric.PARTICLE_IDENTITY.value in diagnostic_metrics
        ),
        parameter_movement=scalar_operand(
            value=ParticleDiagnosticMetric.PARAMETER_MOVEMENT.value in diagnostic_metrics
        ),
        particle_weights=scalar_operand(
            value=ParticleDiagnosticMetric.PARTICLE_WEIGHTS.value in diagnostic_metrics
        ),
    )


def excluded_particle_diagnostics(diagnostic_metrics: frozenset[str]) -> frozenset[str]:
    """Select retained histories on the host without changing the compiled outputs."""
    from grassmoth.artifacts.reports.posterior_diagnostics import ParticleWeightMetric

    groups = {
        ParticleDiagnosticMetric.PARTICLE_IDENTITY: (
            "selected_particle_per_t",
            "reference_path_hit_rate",
            "selected_particle_unique_count",
        ),
        ParticleDiagnosticMetric.PARAMETER_MOVEMENT: ("parameter_jump_rms",),
        ParticleDiagnosticMetric.PARTICLE_WEIGHTS: tuple(
            metric.history_key for metric in ParticleWeightMetric
        ),
    }
    return frozenset(
        name
        for metric, names in groups.items()
        if metric not in diagnostic_metrics
        for name in names
    )
