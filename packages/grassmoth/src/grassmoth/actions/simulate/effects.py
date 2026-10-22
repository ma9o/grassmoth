"""Certify and summarize paired histories without running another simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.reports.checks import NotEvaluated
from grassmoth.artifacts.reports.display_frames import central_frame
from grassmoth.artifacts.scenarios import CausalEffectResult
from grassmoth.models.causal.proofs import (
    CausalCertificationError,
    CertifiedCausalAnalysis,
    certify_identified_estimand,
)

if TYPE_CHECKING:
    from grassmoth.artifacts.identity import ConstructId, GitRef, IndicatorId
    from grassmoth.artifacts.reports.inference import InferenceReportCore
    from grassmoth.artifacts.simulation import SimulationArm, SimulationSpec
    from grassmoth.models.structure import StructuralSelection


def summarize_causal_simulation(
    selection: StructuralSelection,
    design: SimulationSpec,
    *,
    fitted: tuple[GitRef, InferenceReportCore] | None,
    action: SimulationArm,
    reference: SimulationArm,
    state_ids: tuple[ConstructId, ...],
    indicator_ids: tuple[IndicatorId, ...],
) -> CausalEffectResult | NotEvaluated[Literal["causal_effect"]]:
    """Report the question's numeric effect only when identification and exact-fit evidence support it."""
    import jax.numpy as jnp

    from grassmoth.models.ssm.counterfactual.estimands import summarize_draws

    def unavailable(detail: str) -> NotEvaluated[Literal["causal_effect"]]:
        return NotEvaluated(
            code="causal_effect",
            subject="causal_effect",
            reason="CAUSAL_EVALUATION_FAILED",
            detail=detail,
        )

    dynamical_model_spec, outcome = selection.dynamical_model_spec, selection.outcome
    if fitted is None or outcome is None or outcome not in state_ids:
        return unavailable(
            "Causal effects require the question's outcome as a model state and a committed production fit for this model revision."
        )
    model_ref, fit_report = fitted
    try:
        identification = selection.identification
        CertifiedCausalAnalysis(
            dynamical_model_spec=dynamical_model_spec,
            dynamical_model_spec_ref=model_ref,
            identification=identification,
            estimands=tuple(
                certify_identified_estimand(
                    dynamical_model_spec,
                    identification,
                    dynamical_model_spec_ref=model_ref,
                    treatment=dynamical_model_spec.get_construct(target).name,
                    outcome=dynamical_model_spec.get_construct(outcome).name,
                )
                for target in sorted({event.target for event in design.interventions})
            ),
            fitted_dynamical_model_spec=dynamical_model_spec,
            report=fit_report,
        )
    except CausalCertificationError as exc:
        return unavailable(str(exc))
    reference_paths = reference.latent_paths.values
    action_paths = action.latent_paths.values
    if not np.isfinite(reference_paths).all() or not np.isfinite(action_paths).all():
        return unavailable("Non-finite histories do not support numeric causal effects.")
    outcome_index = state_ids.index(outcome)
    differences = action_paths[:, :, outcome_index] - reference_paths[:, :, outcome_index]
    frame = central_frame(differences)
    assert frame is not None, "Finite paired histories have a display range"
    observed = action.observations.values
    reference_observed = reference.observations.values
    manifest_differences = observed[:, -1] - reference_observed[:, -1]
    return CausalEffectResult(
        outcome=outcome,
        labels={construct.id: construct.name for construct in dynamical_model_spec.constructs},
        differences=NumericalArray.from_numpy(differences),
        frame=frame,
        summary=summarize_draws(jnp.asarray(differences[:, -1])),
        reference_mean=float(reference_paths[:, -1, outcome_index].mean()),
        manifest_effects={
            identity: float(manifest_differences[:, index].mean())
            for index, identity in enumerate(indicator_ids)
            if np.isfinite(manifest_differences[:, index]).all()
        },
        warnings=()
        if np.isfinite(manifest_differences).all()
        else (
            "Some indicator contrasts are unavailable because their measurement windows extend before simulation start.",
        ),
    )
