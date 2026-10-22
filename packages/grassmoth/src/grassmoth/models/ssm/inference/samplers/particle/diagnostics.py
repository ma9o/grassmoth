"""Optional diagnostic metric groups shared by the particle samplers."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence


class ParticleDiagnosticMetric(StrEnum):
    """Optional traces: shared dSMC weights, plus marginalized-sampler movement and identity."""

    PARTICLE_IDENTITY = "particle_identity"
    PARAMETER_MOVEMENT = "parameter_movement"
    PARTICLE_WEIGHTS = "particle_weights"


PARTICLE_DIAGNOSTIC_METRIC_VALUES = tuple(metric.value for metric in ParticleDiagnosticMetric)


class ParticleDiagnosticFlags(NamedTuple):
    """Resolved static switches for optional particle diagnostic traces."""

    particle_identity: bool
    parameter_movement: bool
    particle_weights: bool


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
    """Build static booleans for optional diagnostic groups."""
    return ParticleDiagnosticFlags(
        particle_identity=(ParticleDiagnosticMetric.PARTICLE_IDENTITY.value in diagnostic_metrics),
        parameter_movement=(
            ParticleDiagnosticMetric.PARAMETER_MOVEMENT.value in diagnostic_metrics
        ),
        particle_weights=(ParticleDiagnosticMetric.PARTICLE_WEIGHTS.value in diagnostic_metrics),
    )
