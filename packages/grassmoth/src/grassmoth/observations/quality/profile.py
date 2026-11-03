"""Empirical data profiles and model-dependent measurement compatibility checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.artifacts.identity import IndicatorRef
from grassmoth.artifacts.observations.preparation import PreparedDataMetadata
from grassmoth.artifacts.reports.checks import Evaluated
from grassmoth.artifacts.reports.data_quality import (
    DataFinding,
    DataProfileReport,
    IndicatorAudit,
)
from grassmoth.observations.quality.checks import check_construct_correlations
from grassmoth.observations.quality.rules import (
    compute_empirical_profile,
    indicator_findings,
    no_data_validation_result,
)

if TYPE_CHECKING:
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth.artifacts.observations.dataset import ObservationDataset


def validate_extraction(
    dynamical_model_spec: DynamicalModelSpec,
    observations: ObservationDataset,
) -> DataProfileReport:
    """Assess model-dependent compatibility; preparation owns empirical profiles."""
    constructs = {
        indicator.observation.id: construct
        for construct, indicator in dynamical_model_spec.iter_indicators()
    }
    if not any(
        series.history.times.layout[0][0]
        for identity, series in observations.series.items()
        if identity in constructs
    ):
        return no_data_validation_result()
    return DataProfileReport(
        indicators={}, findings=tuple(check_construct_correlations(observations, constructs))
    )


def profile_data(
    observations: ObservationDataset,
    *,
    metadata: PreparedDataMetadata | None = None,
) -> DataProfileReport:
    """Measure admitted data under its own definitions, without a selected model."""
    if not any(series.history.times.layout[0][0] for series in observations.series.values()):
        return no_data_validation_result()
    findings: list[DataFinding] = []
    audits = {
        identity: IndicatorAudit(
            profile=compute_empirical_profile(series),
            findings=indicator_findings(series, observations.time_origin),
        )
        for identity, series in sorted(observations.series.items())
    }
    if isinstance(metadata, PreparedDataMetadata):
        from grassmoth.artifacts.observations.preparation import check_semantic_collisions

        for variable in metadata.preparation.variables:
            for message in check_semantic_collisions(
                variable.extraction.how_to_measure, variable.observation.aggregation
            ):
                audits[variable.observation.id] = audits[variable.observation.id].with_finding(
                    Evaluated(
                        code="scoring_semantics",
                        subject=IndicatorRef(id=variable.observation.id),
                        outcome="failed",
                        evidence=message,
                    )
                )
    return DataProfileReport(indicators=audits, findings=tuple(findings))
