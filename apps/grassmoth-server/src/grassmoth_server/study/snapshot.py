"""Aggregate reads for one committed model revision."""

from __future__ import annotations

from pydantic import Field, model_validator

from grassmoth.public import (
    DataProfileReport,
    DynamicalModelSpec,
    FitCheckReport,
    GitOid,
    IdentificationReport,
    InferenceReportCore,
    PreparedDataMetadata,
    QuestionCheckReport,
    QuestionSpec,
    SimulationReport,
    SpecificationAssessment,
    Value,
)
from grassmoth_server.study.state import StudyState


class ModelSnapshot(Value):
    """Scientific values selected from recorded action dependencies."""

    question: QuestionSpec | None = None
    dynamical_model_spec: DynamicalModelSpec | None = None
    workspace_id: str = Field(min_length=1)
    commit_id: GitOid
    selected_seq: int = Field(ge=0)
    state: StudyState
    metadata: PreparedDataMetadata | None = None
    profile: DataProfileReport | None = None

    identification: IdentificationReport | None = None
    fit_checks: FitCheckReport | None = None
    fit: InferenceReportCore | None = None
    specification: tuple[SpecificationAssessment, ...] | None = None
    question_checks: QuestionCheckReport | None = None
    simulation: SimulationReport | None = None

    @model_validator(mode="after")
    def validate_ownership(self) -> ModelSnapshot:
        """Require snapshot findings to reference model-owned entities."""
        dynamical_model_spec = self.dynamical_model_spec if self.dynamical_model_spec else None
        indicators = (
            {item.observation.id for item in dynamical_model_spec.indicators}
            if dynamical_model_spec
            else set()
        )
        parameters = (
            {item.id for item in dynamical_model_spec.parameters} if dynamical_model_spec else set()
        )
        findings = self
        if findings.identification:
            if dynamical_model_spec is None:
                raise ValueError("Identification requires its scientific model")
            findings.identification.validate_model(dynamical_model_spec)
        # Counts describe the stored table, including data-quality problems.
        # Undeclared variables are reported by preparation checks, never hidden here.
        if findings.fit_checks and not findings.fit_checks.data.indicators.keys() <= indicators:
            raise ValueError("Validation owner does not exist in the snapshot")
        if findings.fit:
            fit = findings.fit
            marginals = fit.posterior_marginals
            if any(item.subject.parameter_id not in parameters for item in marginals):
                raise ValueError("Posterior finding has no scientific parameter definition")
            if not fit.prior_densities.keys() <= parameters:
                raise ValueError("Prior curve has no scientific parameter definition")
        return self
