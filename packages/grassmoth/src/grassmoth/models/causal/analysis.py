"""Identify causal queries from one scientific model, retaining every finding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.artifacts.reports.identification import IdentificationReport
from grassmoth.models.causal.identification import (
    check_identifiability,
    get_observed_constructs,
)

if TYPE_CHECKING:
    from grassmoth.models.structure import StructuralSelection


def identify_model(selection: StructuralSelection) -> IdentificationReport:
    """Identify each treatment's effect on the outcome that scopes the selection."""
    dynamical_model_spec, outcome = selection.dynamical_model_spec, selection.outcome
    if outcome is None:
        return IdentificationReport(outcome=None)
    # DynamicalModelSpec permits nonlinear dynamics; a linear-IV argument cannot establish
    # identification for this model, including during partial authoring.
    return check_identifiability(
        dynamical_model_spec.constructs,
        dynamical_model_spec.edges,
        outcome_id=outcome,
        observed_constructs=get_observed_constructs(dynamical_model_spec.constructs),
        iv_allowed=False,
    )
