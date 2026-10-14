"""Empirical data profiles and model-dependent measurement compatibility checks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

from nof1_causal_lab.artifacts.identity import IndicatorId, IndicatorRef
from nof1_causal_lab.artifacts.observations.preparation import PreparedDataMetadata
from nof1_causal_lab.artifacts.reports.checks import Evaluated
from nof1_causal_lab.artifacts.reports.data_quality import (
    DataFinding,
    DataProfileReport,
    IndicatorAudit,
)
from nof1_causal_lab.observations.quality.checks import check_construct_correlations
from nof1_causal_lab.observations.quality.rules import (
    compute_empirical_profile,
    indicator_findings,
    no_data_validation_result,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from nof1_causal_lab.artifacts.observations.spec import ResolvedObservationSpec


def validate_extraction(
    dynamical_model_spec: DynamicalModelSpec,
    dataframes: list[pl.DataFrame],
) -> DataProfileReport:
    """Assess model-dependent compatibility; preparation owns empirical profiles."""
    frames = [frame for frame in dataframes if not frame.is_empty()]
    if not frames:
        return no_data_validation_result()
    constructs = {
        indicator.observation.id: construct
        for construct, indicator in dynamical_model_spec.iter_indicators()
    }
    combined = pl.concat(frames, how="vertical").filter(
        pl.col("indicator_id").is_in(list(constructs))
    )
    if combined.is_empty():
        return no_data_validation_result()
    return DataProfileReport(
        indicators={}, findings=tuple(check_construct_correlations(combined, constructs))
    )


def profile_data(
    data: pl.DataFrame,
    *,
    definitions: tuple[ResolvedObservationSpec, ...] = (),
    metadata: PreparedDataMetadata | None = None,
) -> DataProfileReport:
    """Measure recorded data under its own definitions, without a selected model."""
    if data.is_empty():
        return no_data_validation_result()
    definitions = metadata.variables if metadata is not None else definitions
    lookup = {item.id: item for item in definitions}
    identities = {IndicatorId(value) for value in data["indicator_id"].unique()} | set(lookup)
    findings: list[DataFinding] = []
    audits = {
        identity: IndicatorAudit(
            profile=compute_empirical_profile(identity, data),
            findings=indicator_findings(
                identity, data.filter(pl.col("indicator_id") == identity), lookup
            ),
        )
        for identity in sorted(identities)
    }
    if definitions:
        unknown = set(data["indicator_id"].unique()) - set(lookup)
        if unknown:
            findings.append(
                Evaluated(
                    code="undeclared_variables",
                    subject="dataset",
                    outcome="failed",
                    evidence=f"{len(unknown)} observed variables have no definitions in the prepared-data metadata.",
                )
            )
        for variable in definitions:
            levels = (
                variable.ordinal_levels
                if variable.measurement_dtype == "ordinal"
                else variable.categorical_levels
            )
            values = data.filter(pl.col("indicator_id") == variable.id)["value"].drop_nulls()
            if levels and any(
                value < 0 or value >= len(levels) or value != int(value)
                for value in values
                if value is not None and math.isfinite(value)
            ):
                audits[variable.id] = audits[variable.id].with_finding(
                    Evaluated(
                        code="codebook_bounds",
                        subject=IndicatorRef(id=variable.id),
                        outcome="failed",
                        evidence="Observed codes must index the declared codebook",
                    )
                )
            if not values.is_finite().all():
                audits[variable.id] = audits[variable.id].with_finding(
                    Evaluated(
                        code="nonfinite_values",
                        subject=IndicatorRef(id=variable.id),
                        outcome="failed",
                        evidence="Observed numeric values must be finite",
                    )
                )
    if isinstance(metadata, PreparedDataMetadata):
        from nof1_causal_lab.artifacts.observations.preparation import check_semantic_collisions

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
