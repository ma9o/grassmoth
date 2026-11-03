"""Report the fit's convergence and unavailable contrasts beside simulation checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from grassmoth.actions.contracts.io import FittedLaw
from grassmoth.artifacts.identity import IndicatorRef
from grassmoth.artifacts.reports.checks import (
    Evaluated,
    NotEvaluated,
    PredictiveAssessment,
    PredictiveSubject,
)
from grassmoth.artifacts.scenarios import CausalEffectResult
from grassmoth.artifacts.simulation import PairedArmSimulation
from grassmoth.models.ssm.inference.diagnostics.convergence import R_HAT_LIMIT, R_HAT_WARNING

if TYPE_CHECKING:
    from grassmoth.actions.contracts.io import PredictiveLaw
    from grassmoth.artifacts.simulation import SimulationEvidence


def _fit_findings(law: PredictiveLaw) -> tuple[PredictiveAssessment, ...]:
    subject = PredictiveSubject(target="whole_model")
    if not isinstance(law, FittedLaw):
        if law.kind == "authored":
            return ()
        return (
            NotEvaluated(
                code="fit_reliability",
                subject=subject,
                reason="NO_RETAINED_CHAINS",
                detail="The current laws do not have one retained fit's convergence report.",
            ),
        )
    convergence = law.report.convergence
    status = convergence.status
    missing = tuple(item for item in convergence.findings if isinstance(item, NotEvaluated))
    measured = tuple(item for item in convergence.findings if isinstance(item, Evaluated))
    reliability: PredictiveAssessment = (
        NotEvaluated(
            code="fit_reliability",
            subject=subject,
            reason="NO_RETAINED_CHAINS",
            detail="; ".join(convergence.messages)
            or "No retained convergence criteria are available.",
        )
        if status == "not_evaluated"
        else Evaluated(
            code="fit_reliability",
            subject=subject,
            outcome=status,
            evidence=tuple(
                item.evidence.revised(
                    note=f"{item.evidence.criterion} {item.outcome} for {item.evidence.note}: "
                    f"{item.evidence.value:g}",
                    band_label=(
                        f"< {item.evidence.upper:g}"
                        if item.evidence.upper is not None
                        else f"≥ {item.evidence.lower:g}"
                    ),
                )
                for item in measured
                if item.outcome == status
            ),
        )
    )
    mixing = tuple(
        item.evidence.revised(
            upper=R_HAT_WARNING,
            note=f"{item.evidence.note}: r_hat is {item.evidence.value:g}; "
            "inspect mixing before relying on this fit.",
            band_label=f"< {R_HAT_WARNING:g}",
        )
        for item in measured
        if item.evidence.criterion == "r_hat" and R_HAT_WARNING <= item.evidence.value < R_HAT_LIMIT
    )
    unavailable: tuple[PredictiveAssessment, ...] = (
        (
            NotEvaluated(
                code="fit_reliability_unavailable",
                subject=subject,
                reason="NO_RETAINED_CHAINS",
                detail="; ".join(item.detail for item in missing),
            ),
        )
        if missing and status == "failed"
        else ()
    )
    mixing_findings: tuple[PredictiveAssessment, ...] = (
        (Evaluated(code="fit_mixing", subject=subject, outcome="failed", evidence=mixing),)
        if mixing
        else ()
    )
    return (reliability, *unavailable, *mixing_findings)


def simulation_context_findings(
    evidence: SimulationEvidence, law: PredictiveLaw
) -> tuple[PredictiveAssessment, ...]:
    """Use the existing assessment structure for fit reliability and unavailable contrasts."""
    arms = evidence.arms
    contrasts: tuple[PredictiveAssessment, ...] = ()
    if isinstance(arms, PairedArmSimulation) and isinstance(arms.causal, CausalEffectResult):
        differences = (
            arms.action.observations.values[:, -1] - arms.reference.observations.values[:, -1]
        )
        contrasts = tuple(
            NotEvaluated(
                code="causal_indicator_contrast",
                subject=PredictiveSubject(target=IndicatorRef(id=identity)),
                reason="NO_OBSERVATION_SUPPORT",
                detail="The indicator contrast is unavailable because its measurement window "
                "extends before simulation start.",
            )
            for column, identity in enumerate(evidence.observation_layout.indicator_ids)
            if not np.isfinite(differences[:, column]).all()
        )
    return (*_fit_findings(law), *contrasts)
