"""Pure action policies announcing retained findings without carrying their evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.public import (
    Assessment,
    ConstructRef,
    FindingSubject,
    NotEvaluated,
    NumericCriterionEvidence,
    PairedArmSimulation,
    PredictiveIndicatorComparison,
    QueryTargetSubject,
)
from nof1_causal_service.study.records import ActionMessage

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from nof1_causal_lab.public import (
        DataComparisonReport,
        DataPreparationResult,
        DataProfileReport,
        FitCheckReport,
        IdentificationReport,
        InferenceReport,
        ModelCheckReport,
        ModelEditResult,
        SimulationReport,
    )


type FindingEvidence = str | NumericCriterionEvidence | tuple[NumericCriterionEvidence, ...]


def _explanation(evidence: FindingEvidence) -> str:
    if isinstance(evidence, str):
        return evidence
    values = evidence if isinstance(evidence, tuple) else (evidence,)
    return "; ".join(
        dict.fromkeys(item.note or f"{item.criterion}: {item.value:g}" for item in values)
    )


def _findings[SubjectT: FindingSubject, EvidenceT: FindingEvidence](
    findings: Iterable[Assessment[SubjectT, EvidenceT]],
    timestamp: datetime,
    *,
    unevaluated: Literal["info", "warning"] = "info",
) -> tuple[ActionMessage, ...]:
    return tuple(
        ActionMessage(
            timestamp=timestamp,
            severity=unevaluated if isinstance(finding, NotEvaluated) else "warning",
            code=finding.code,
            subject=finding.subject,
            detail=finding.detail
            if isinstance(finding, NotEvaluated)
            else _explanation(finding.evidence),
        )
        for finding in findings
        if isinstance(finding, NotEvaluated) or finding.outcome == "failed"
    )


def edit_messages(
    checks: ModelCheckReport,
    identification: IdentificationReport,
    pruning: ModelEditResult,
    timestamp: datetime,
) -> tuple[ActionMessage, ...]:
    """An incomplete edit remains useful; its execution and identification findings warn."""
    queried = {
        finding.subject.target.id
        for finding in checks.question.findings
        if finding.code == "identification" and isinstance(finding.subject, QueryTargetSubject)
    }
    return (
        *_findings(checks.specification, timestamp, unevaluated="warning"),
        *_findings(checks.question.findings, timestamp),
        *(
            ActionMessage(
                timestamp=timestamp,
                severity="warning",
                code="identification",
                subject=ConstructRef(id=target),
                detail="This target's causal effect is not identifiable.",
            )
            for target in identification.non_identifiable
            if target not in queried
        ),
        *(
            (
                ActionMessage(
                    timestamp=timestamp,
                    severity="warning",
                    code="MODEL_COMPONENTS_PRUNED",
                    subject="model",
                    detail="Components outside the question's outcome ancestry were removed.",
                ),
            )
            if any((pruning.constructs, pruning.edges, pruning.parameters, pruning.distributions))
            else ()
        ),
    )


def _data_messages(profile: DataProfileReport, timestamp: datetime) -> tuple[ActionMessage, ...]:
    return _findings(
        (
            *profile.findings,
            *(finding for audit in profile.indicators.values() for finding in audit.findings),
        ),
        timestamp,
    )


def preparation_messages(
    profile: DataProfileReport, extraction: DataPreparationResult, timestamp: datetime
) -> tuple[ActionMessage, ...]:
    """Announce preparation findings while worker results remain in the saved body."""
    failed = sum(worker.status == "failed" for worker in extraction.workers)
    return (
        *_data_messages(profile, timestamp),
        ActionMessage(
            timestamp=timestamp,
            severity="warning" if failed else "info",
            code="EXTRACTION_PARTIAL" if failed else "EXTRACTION_COMPLETED",
            subject="data",
            detail=f"Extraction completed with {failed} failed chunks.",
        ),
        *(
            (
                ActionMessage(
                    timestamp=timestamp,
                    severity="info",
                    code="EXTRACTION_REUSED",
                    subject="data",
                    detail=f"Reused {extraction.extraction_reused} extraction results.",
                ),
            )
            if extraction.extraction_reused
            else ()
        ),
    )


def fit_check_messages(checks: FitCheckReport, timestamp: datetime) -> tuple[ActionMessage, ...]:
    """Announce only the selected record's fit checks; the model edit owns its own findings."""
    return (
        *_data_messages(checks.data, timestamp),
        *_findings(checks.preflight, timestamp, unevaluated="warning"),
        *_findings(checks.question.findings, timestamp),
    )


def inference_messages(report: InferenceReport, timestamp: datetime) -> tuple[ActionMessage, ...]:
    """Retained convergence findings warn without changing completed inference evidence."""
    return _findings(report.core.convergence.findings, timestamp)


def simulation_messages(report: SimulationReport, timestamp: datetime) -> tuple[ActionMessage, ...]:
    """Announce the findings actually evaluated on the generated histories."""
    arms = report.evidence.arms
    return (
        *_findings(report.findings, timestamp),
        *(
            _findings((arms.causal,), timestamp)
            if isinstance(arms, PairedArmSimulation) and isinstance(arms.causal, NotEvaluated)
            else ()
        ),
    )


def comparison_messages(
    report: DataComparisonReport, timestamp: datetime
) -> tuple[ActionMessage, ...]:
    """Announce comparison findings and applicable predictive evaluations once."""
    return tuple(
        message
        for variable in report.variables
        for message in (
            *_findings(variable.findings, timestamp),
            *(
                (
                    _findings((variable.predictive.evaluation,), timestamp)
                    if isinstance(variable.predictive.evaluation, NotEvaluated)
                    else _findings(variable.predictive.evaluation.findings, timestamp)
                )
                if isinstance(variable, PredictiveIndicatorComparison)
                else ()
            ),
        )
    )
