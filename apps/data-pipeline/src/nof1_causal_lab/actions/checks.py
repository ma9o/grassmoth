"""Evaluate inexpensive specification findings without fitting or simulating."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.reports.checks import (
    Evaluated,
    NotEvaluated,
    SpecificationAssessment,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model.question import QuestionSpec
    from nof1_causal_lab.artifacts.observations.dataset import ObservationDataset
    from nof1_causal_lab.artifacts.reports.inference import FitCheckReport
    from nof1_causal_lab.artifacts.reports.model_checks import ModelCheckReport
    from nof1_causal_lab.models.ssm.compile.inputs import (
        FitCompilationResult,
        ModelCompilationResult,
    )
    from nof1_causal_lab.models.structure import StructuralSelection


def check_specification(
    compiled: ModelCompilationResult, inputs: FitCompilationResult
) -> tuple[SpecificationAssessment, ...]:
    """Keep incomplete and unsupported candidates editable and report their readiness."""
    from typing import assert_never

    from nof1_causal_lab.models.ssm.compile.inputs import (
        CompiledDynamicalModel,
        CompiledFitInputs,
        IncompleteModel,
        UnsupportedFit,
    )

    match compiled:
        case IncompleteModel():
            return (
                NotEvaluated(
                    code="model_execution",
                    subject="model",
                    reason="MODEL_INCOMPLETE",
                    detail=compiled.message,
                ),
            )
        case UnsupportedFit():
            return (
                Evaluated(
                    code="model_execution",
                    subject="model",
                    outcome="failed",
                    evidence=compiled.message,
                ),
            )
        case CompiledDynamicalModel():
            pass
        case _:
            assert_never(compiled)
    findings: list[SpecificationAssessment] = [
        Evaluated(
            code="model_execution",
            subject="model",
            outcome="passed",
            evidence="Model definitions, references, and measurements support execution.",
        )
    ]
    match inputs:
        case IncompleteModel() | UnsupportedFit():
            findings.append(
                Evaluated(
                    code="fit_laws", subject="model", outcome="failed", evidence=inputs.message
                )
            )
        case CompiledFitInputs():
            findings.append(
                Evaluated(
                    code="fit_laws",
                    subject="model",
                    outcome="passed",
                    evidence="Parameter laws support the current fitting engine.",
                )
            )
            findings.extend(
                Evaluated(
                    code=diagnostic.code,
                    subject=diagnostic.parameter,
                    outcome="failed",
                    evidence=diagnostic.issue or diagnostic.parameter,
                )
                for diagnostic in inputs.diagnostics
            )
        case _:
            assert_never(inputs)
    return tuple(findings)


def model_check_report(selection: StructuralSelection, question: QuestionSpec) -> ModelCheckReport:
    """Evaluate specification and question findings on the supplied model."""
    from nof1_causal_lab.artifacts.reports.model_checks import ModelCheckReport, QuestionCheckReport
    from nof1_causal_lab.models.causal.question_checks import question_findings
    from nof1_causal_lab.models.ssm.compile.inputs import compile_fit_inputs, compile_model

    compiled = compile_model(selection)
    inputs = compile_fit_inputs(compiled, selection)
    return ModelCheckReport(
        specification=check_specification(compiled, inputs),
        question=QuestionCheckReport(findings=question_findings(question, selection, history=None)),
    )


def fit_check_report(
    selection: StructuralSelection,
    question: QuestionSpec,
    observations: ObservationDataset,
) -> FitCheckReport:
    """Retain input findings after a successful fit has admitted and used these values."""
    from nof1_causal_lab.artifacts.reports.inference import FitCheckReport
    from nof1_causal_lab.artifacts.reports.model_checks import QuestionCheckReport
    from nof1_causal_lab.models.causal.question_checks import question_findings
    from nof1_causal_lab.observations.quality.profile import validate_extraction

    return FitCheckReport(
        data=validate_extraction(selection.dynamical_model_spec, [observations.recorded.frame]),
        preflight=(
            Evaluated(
                code="fit_preflight",
                subject="model",
                outcome="passed",
                evidence="Measurement support, discrete levels, standardization and eligible location laws match the prepared observations.",
            ),
        ),
        question=QuestionCheckReport(
            findings=question_findings(question, selection, history=observations)
        ),
    )
