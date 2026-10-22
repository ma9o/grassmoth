"""Tests for validation: Validate extracted data.

This module tests:
1. validate_extraction() - semantic checks (variance, sample size)
2. Validation checks - timestamps, dtype, coverage, gaps, hallucination, correlations
"""

from datetime import datetime, timedelta
from typing import Any, Literal

import polars as pl
import pytest

from grassmoth.artifacts.identity import IndicatorId, IndicatorRef
from grassmoth.artifacts.model.construct import (
    CausalEdgeSpec,
    ConstructSpec,
    replace_constructs,
)
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.observations.duration import Duration
from grassmoth.artifacts.reports.checks import Evaluated
from grassmoth.artifacts.reports.data_quality import DataProfileReport
from grassmoth.observations.quality.checks import parse_timestamp_series
from grassmoth.observations.quality.profile import profile_data
from grassmoth.observations.quality.profile import validate_extraction as validate_model_data
from grassmoth_test_support.helpers import fixture_entity_id, make_model

pytestmark = pytest.mark.contract


def test_recorded_point_cadence_uses_declared_window_and_only_observed_times():
    definition = (
        _make_spec(model_clock="3h")
        .indicators[0]
        .observation.revised(aggregation="last")
        .resolved(Duration("3h"))
    )
    times = [datetime(2020, 1, 1) + timedelta(hours=3 * index) for index in range(16)]
    data = pl.DataFrame(
        {
            "indicator_id": [definition.id] * 16,
            "value": [float(index % 5) for index in range(16)],
            "anchor_time": times,
            "support_start": times,
            "support_end": times,
        }
    )
    complete = profile_data(data, definitions=(definition,))
    time_findings = {
        item.code: item
        for item in complete.indicators[definition.id].findings
        if item.code in {"insufficient_coverage", "large_timestamp_gap"}
    }
    assert all(
        isinstance(item, Evaluated) and item.outcome == "passed" for item in time_findings.values()
    )
    masked = data.with_columns(
        pl.when(pl.int_range(pl.len()).is_between(4, 11))
        .then(None)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    sparse = profile_data(masked, definitions=(definition,))
    gaps = [item for item in _all_issues(sparse) if item.code == "large_timestamp_gap"]
    assert len(gaps) == 1
    assert "27h" in gaps[0].evidence


@pytest.fixture
def simple_causal_design():
    stress = _make_spec()
    sleep = _make_spec(indicator_name="sleep_hours", construct_name="sleep")
    return stress.with_entities(
        edges=replace_constructs(
            (
                CausalEdgeSpec(
                    id=fixture_entity_id("edge", "stress->sleep"),
                    cause=stress.constructs[0],
                    effect=sleep.constructs[0],
                    description="Stress affects sleep",
                ),
            ),
            (stress.constructs[0], sleep.constructs[0]),
        )
    )


def _create_worker_dfs(records: list[dict[str, Any]]) -> list[pl.DataFrame]:
    """Create DataFrames for validate_extraction from records."""
    df = pl.DataFrame(
        records,
        schema={"indicator_id": pl.Utf8, "value": pl.Utf8, "anchor_time": pl.Utf8},
    )
    return [df]


def validate_extraction(
    dynamical_model_spec: DynamicalModelSpec, dataframes: list[pl.DataFrame]
) -> DataProfileReport:
    """Exercise stored data quality and model compatibility as distinct owners."""
    definitions = tuple(
        item.observation.resolved(
            item.observation.observation_window
            or dynamical_model_spec.measurement_clock
            or Duration("1d")
        )
        for item in dynamical_model_spec.indicators
    )
    windows = {}
    for definition in definitions:
        assert isinstance(definition.observation_window, Duration)
        windows[definition.id] = definition.observation_window.seconds
    dataframes = [
        frame.with_columns(
            (
                parse_timestamp_series(frame["anchor_time"])
                - frame.select(
                    pl.duration(seconds=pl.col("indicator_id").replace_strict(windows))
                ).to_series()
            ).alias("support_start"),
            parse_timestamp_series(frame["anchor_time"]).alias("support_end"),
        )
        for frame in dataframes
    ]
    data = pl.concat(dataframes).with_columns(pl.col("value").cast(pl.Float64, strict=False))
    profile = profile_data(data, definitions=definitions)
    compatibility = validate_model_data(dynamical_model_spec, dataframes)
    return profile.revised(findings=(*profile.findings, *compatibility.findings))


def _all_issues(
    result: DataProfileReport,
) -> tuple[Evaluated[IndicatorRef | Literal["dataset"], str], ...]:
    return tuple(
        finding
        for finding in (
            *result.findings,
            *(finding for audit in result.indicators.values() for finding in audit.findings),
        )
        if isinstance(finding, Evaluated) and finding.outcome == "failed"
    )


def _issues_for_indicator(
    result: DataProfileReport, indicator: IndicatorId
) -> tuple[Evaluated[IndicatorRef | Literal["dataset"], str], ...]:
    return tuple(
        finding
        for finding in _all_issues(result)
        if isinstance(finding.subject, IndicatorRef) and finding.subject.id == indicator
    )


def _make_spec(
    indicator_name="stress_score",
    construct_name="stress",
    dtype="continuous",
    model_clock="1d",
    temporal_status="time_varying",
    extra_indicators=None,
):
    """Create a minimal causal design for testing individual checks."""
    indicators = [
        {
            "id": fixture_entity_id("indicator", indicator_name),
            "construct_id": fixture_entity_id("construct", construct_name),
            "name": indicator_name,
            "measurement_dtype": dtype,
        },
    ]
    if extra_indicators:
        indicators.extend(extra_indicators)

    definitions = []
    for indicator in indicators:
        assert indicator.pop("construct_id") == fixture_entity_id("construct", construct_name)
        indicator.setdefault("construct_polarity", "positive")
        indicator.setdefault("measurement_dtype", "continuous")
        indicator.setdefault("aggregation", "last")
        definitions.append(
            {"observation": indicator, "construct_polarity": indicator.pop("construct_polarity")}
        )
    dynamical_model_spec = make_model([construct_name])
    construct = ConstructSpec.model_validate(
        {
            "id": fixture_entity_id("construct", construct_name),
            "name": construct_name,
            "description": "Validation fixture",
            "role": "endogenous",
            "temporal_status": temporal_status,
            "indicators": definitions,
        }
    )
    return dynamical_model_spec.revised(measurement_clock=model_clock).with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, [construct]),
    )


# ==============================================================================
# UNIT TESTS: validate_extraction
# ==============================================================================


class TestValidateExtraction:
    """Test validate_extraction semantic checks."""

    def test_empty_results_returns_error(self, simple_causal_design):
        """Empty worker results returns error."""
        result = validate_model_data(simple_causal_design, [])
        assert any(i.code == "no_data" for i in _all_issues(result))

    def test_valid_data_no_issues(self, simple_causal_design):
        """Valid data with sufficient variance and sample size passes."""
        records = []
        for i in range(20):
            records.append(
                {
                    "indicator_id": "indicator:3696aef3ff6f446744e5",
                    "value": str(float(i % 5 + 1)),  # 1-5 varying
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
            records.append(
                {
                    "indicator_id": "indicator:9866c549bd1c25f0a5d7",
                    "value": str(6.0 + (i % 3)),  # 6-8 varying
                    "anchor_time": f"2024-01-{i + 1:02d} 08:00",
                }
            )

        worker_results = _create_worker_dfs(records)
        result = validate_extraction(simple_causal_design, worker_results)

        # May have warnings but no errors
        errors = [i for i in _all_issues(result) if i.code == "dtype_violation"]
        assert len(errors) == 0

    def test_missing_indicator_is_warning(self, simple_causal_design):
        """Missing indicator generates warning."""
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "5.0",
                "anchor_time": "2024-01-01 10:00",
            },
            # sleep_hours is missing
        ]
        worker_results = _create_worker_dfs(records)
        result = validate_extraction(simple_causal_design, worker_results)

        # Should have warning for missing sleep_hours
        missing_issues = [i for i in _all_issues(result) if i.code == "missing"]
        assert any(
            isinstance(i.subject, IndicatorRef) and i.subject.id == "indicator:9866c549bd1c25f0a5d7"
            for i in missing_issues
        )

    def test_zero_variance_is_warning(self, simple_causal_design):
        """Constant observed values are a data-quality warning."""
        records = []
        for i in range(20):
            records.append(
                {
                    "indicator_id": "indicator:3696aef3ff6f446744e5",
                    "value": "5.0",  # Constant!
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
            records.append(
                {
                    "indicator_id": "indicator:9866c549bd1c25f0a5d7",
                    "value": str(6.0 + (i % 3)),  # Varying
                    "anchor_time": f"2024-01-{i + 1:02d} 08:00",
                }
            )

        worker_results = _create_worker_dfs(records)
        result = validate_extraction(simple_causal_design, worker_results)

        variance_issues = [i for i in _all_issues(result) if i.code == "variance"]
        assert len(variance_issues) == 1
        assert isinstance(variance_issues[0].subject, IndicatorRef)
        assert variance_issues[0].subject.id == "indicator:3696aef3ff6f446744e5"
        assert variance_issues[0].outcome == "failed"

    def test_data_quality_is_retained_independently_of_construct_time_status(self):
        """A model time status does not reinterpret the stored empirical profile."""
        dynamical_model_spec = _make_spec(model_clock=None, temporal_status="time_invariant")
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "1.0",
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        no_var = [i for i in _all_issues(result) if i.code == "variance"]
        assert len(no_var) == 1
        assert no_var[0].outcome == "failed"

    def test_low_sample_size_is_warning(self, simple_causal_design):
        """Low sample size generates warning."""
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "3.0",
                "anchor_time": "2024-01-01 10:00",
            },
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "4.0",
                "anchor_time": "2024-01-02 10:00",
            },
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "5.0",
                "anchor_time": "2024-01-03 10:00",
            },
            {
                "indicator_id": "indicator:9866c549bd1c25f0a5d7",
                "value": "7.0",
                "anchor_time": "2024-01-01 08:00",
            },
            {
                "indicator_id": "indicator:9866c549bd1c25f0a5d7",
                "value": "7.5",
                "anchor_time": "2024-01-02 08:00",
            },
            {
                "indicator_id": "indicator:9866c549bd1c25f0a5d7",
                "value": "8.0",
                "anchor_time": "2024-01-03 08:00",
            },
        ]

        worker_results = _create_worker_dfs(records)
        result = validate_extraction(simple_causal_design, worker_results)

        # Should be valid (warnings only)

        # But should have low_n warnings
        low_n_warnings = [i for i in _all_issues(result) if i.code == "sample_size"]
        assert len(low_n_warnings) == 2  # Both indicators

    def test_non_numeric_values_are_errors(self, simple_causal_design):
        """Non-numeric values that can't be cast generate error."""
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "high",
                "anchor_time": "2024-01-01 10:00",
            },
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "medium",
                "anchor_time": "2024-01-02 10:00",
            },
            {
                "indicator_id": "indicator:9866c549bd1c25f0a5d7",
                "value": "7.0",
                "anchor_time": "2024-01-01 08:00",
            },
        ]

        worker_results = _create_worker_dfs(records)
        result = validate_extraction(simple_causal_design, worker_results)

        # stress_score should have no_numeric error
        stress_issues = _issues_for_indicator(
            result, fixture_entity_id("indicator", "stress_score")
        )
        assert any(i.code == "no_numeric" for i in stress_issues)

    def test_combined_error_and_warning(self, simple_causal_design):
        """Indicator can have multiple issues."""
        records = [
            # stress_score: no numeric observations
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "not_numeric",
                "anchor_time": "2024-01-01 10:00",
            },
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "not_numeric",
                "anchor_time": "2024-01-02 10:00",
            },
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "not_numeric",
                "anchor_time": "2024-01-03 10:00",
            },
            # sleep_hours: varying but low N
            {
                "indicator_id": "indicator:9866c549bd1c25f0a5d7",
                "value": "7.0",
                "anchor_time": "2024-01-01 08:00",
            },
            {
                "indicator_id": "indicator:9866c549bd1c25f0a5d7",
                "value": "8.0",
                "anchor_time": "2024-01-02 08:00",
            },
            {
                "indicator_id": "indicator:9866c549bd1c25f0a5d7",
                "value": "7.5",
                "anchor_time": "2024-01-03 08:00",
            },
        ]

        worker_results = _create_worker_dfs(records)
        result = validate_extraction(simple_causal_design, worker_results)

        # stress_score has a numeric-data error; sleep retains a low-N warning
        stress_issues = _issues_for_indicator(
            result, fixture_entity_id("indicator", "stress_score")
        )
        issue_types = {i.code for i in stress_issues}
        assert "no_numeric" in issue_types
        assert any(i.outcome == "failed" for i in stress_issues)
        sleep_issues = _issues_for_indicator(result, fixture_entity_id("indicator", "sleep_hours"))
        assert any(i.code == "sample_size" and i.outcome == "failed" for i in sleep_issues)

    def test_only_warnings_is_valid(self, simple_causal_design):
        """is_valid=True when only warnings exist."""
        records = []
        # Only 5 observations but varying
        for i in range(5):
            records.append(
                {
                    "indicator_id": "indicator:3696aef3ff6f446744e5",
                    "value": str(float(i + 1)),
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
            records.append(
                {
                    "indicator_id": "indicator:9866c549bd1c25f0a5d7",
                    "value": str(6.0 + i * 0.5),
                    "anchor_time": f"2024-01-{i + 1:02d} 08:00",
                }
            )

        worker_results = _create_worker_dfs(records)
        result = validate_extraction(simple_causal_design, worker_results)

        issues = _all_issues(result)
        assert len(issues) > 0
        assert all(i.outcome == "failed" for i in issues)


# ==============================================================================
# UNIT TESTS: _check_timestamps
# ==============================================================================


class TestCheckTimestamps:
    """Test timestamp parseability checks."""

    def test_all_parseable_no_issue(self):
        """All parseable timestamps produce no issues."""
        dynamical_model_spec = _make_spec()
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": str(float(i)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        ts_issues = [i for i in _all_issues(result) if i.code == "timestamps"]
        assert len(ts_issues) == 0

    def test_native_datetime_timestamps_are_parseable(self):
        """Already-normalized datetime timestamps should remain parseable."""
        dynamical_model_spec = _make_spec()
        df = pl.DataFrame(
            {
                "indicator_id": ["indicator:3696aef3ff6f446744e5"] * 20,
                "value": [float(i) for i in range(20)],
                "anchor_time": [datetime(2024, 1, i + 1, 10, 0, 0) for i in range(20)],
            }
        )
        result = validate_extraction(dynamical_model_spec, [df])
        ts_issues = [i for i in _all_issues(result) if i.code == "timestamps"]
        assert len(ts_issues) == 0

    def test_all_unparseable_is_error(self):
        """100% unparseable timestamps → error."""
        dynamical_model_spec = _make_spec()
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": str(float(i)),
                "anchor_time": "not-a-date",
            }
            for i in range(20)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        ts_issues = [i for i in _all_issues(result) if i.code == "timestamps"]
        assert len(ts_issues) == 1
        assert ts_issues[0].outcome == "failed"

    def test_majority_unparseable_is_warning(self):
        """Over 50% unparseable timestamps → warning."""
        dynamical_model_spec = _make_spec()
        records = []
        for i in range(20):
            ts = f"2024-01-{i + 1:02d} 10:00" if i < 8 else "garbage"
            records.append(
                {
                    "indicator_id": "indicator:3696aef3ff6f446744e5",
                    "value": str(float(i)),
                    "anchor_time": ts,
                }
            )
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        ts_issues = [i for i in _all_issues(result) if i.code == "timestamps"]
        assert len(ts_issues) == 1
        assert ts_issues[0].outcome == "failed"

    def test_minority_unparseable_no_issue(self):
        """Under 50% unparseable timestamps → no timestamp issue."""
        dynamical_model_spec = _make_spec()
        records = []
        for i in range(20):
            ts = "garbage" if i < 5 else f"2024-01-{i + 1:02d} 10:00"
            records.append(
                {
                    "indicator_id": "indicator:3696aef3ff6f446744e5",
                    "value": str(float(i)),
                    "anchor_time": ts,
                }
            )
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        ts_issues = [i for i in _all_issues(result) if i.code == "timestamps"]
        assert len(ts_issues) == 0


# ==============================================================================
# UNIT TESTS: _check_dtype_range
# ==============================================================================


class TestCheckDtypeRange:
    """Test dtype range conformance checks."""

    def test_binary_valid(self):
        """Binary values in {0, 1} produce no dtype issues."""
        dynamical_model_spec = _make_spec(dtype="binary")
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": v,
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i, v in enumerate(["0", "1", "0", "1", "1", "0", "1", "0", "1", "0"] * 2)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        dtype_issues = [i for i in _all_issues(result) if i.code == "dtype_violation"]
        assert len(dtype_issues) == 0

    def test_binary_violation_is_error(self):
        """Binary values outside {0, 1} → error."""
        dynamical_model_spec = _make_spec(dtype="binary")
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": v,
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i, v in enumerate(["0", "1", "2", "0.5", "1", "0", "1", "0", "1", "0"])
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        dtype_issues = [i for i in _all_issues(result) if i.code == "dtype_violation"]
        assert len(dtype_issues) == 1
        assert dtype_issues[0].outcome == "failed"

    def test_count_negative_is_error(self):
        """Count indicator with negative values → error."""
        dynamical_model_spec = _make_spec(dtype="count")
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": v,
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i, v in enumerate(["3", "5", "-1", "2", "4", "0", "1", "6", "3", "2"])
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        dtype_issues = [i for i in _all_issues(result) if i.code == "dtype_violation"]
        assert any(i.outcome == "failed" for i in dtype_issues)
        assert any("negative" in i.evidence for i in dtype_issues)

    def test_count_fractional_is_error(self):
        """Count indicator with fractional values → error."""
        dynamical_model_spec = _make_spec(dtype="count")
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": v,
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i, v in enumerate(["3", "5", "2.5", "2", "4", "0", "1", "6", "3", "2"])
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        dtype_issues = [i for i in _all_issues(result) if i.code == "dtype_violation"]
        assert any(i.outcome == "failed" for i in dtype_issues)
        assert any("fractional" in i.evidence for i in dtype_issues)

    def test_continuous_outlier_warning(self):
        """Continuous data with extreme outlier → warning."""
        dynamical_model_spec = _make_spec(dtype="continuous")
        values = [str(float(i)) for i in range(20)]
        values[-1] = "1000.0"  # Extreme outlier
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": v,
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i, v in enumerate(values)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        dtype_issues = [i for i in _all_issues(result) if i.code == "dtype_violation"]
        assert len(dtype_issues) == 1
        assert dtype_issues[0].outcome == "failed"

    def test_continuous_no_outlier(self):
        """Continuous data without outliers produces no dtype issues."""
        dynamical_model_spec = _make_spec(dtype="continuous")
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": str(float(i % 10)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        dtype_issues = [i for i in _all_issues(result) if i.code == "dtype_violation"]
        assert len(dtype_issues) == 0


# ==============================================================================
# UNIT TESTS: _check_time_coverage
# ==============================================================================


class TestCheckTimeCoverage:
    """Test time coverage checks."""

    def test_sufficient_coverage_no_issue(self):
        """Enough time span produces no coverage issue."""
        dynamical_model_spec = _make_spec(model_clock="1d")
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": str(float(i % 5)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        cov_issues = [i for i in _all_issues(result) if i.code == "insufficient_coverage"]
        assert len(cov_issues) == 0

    def test_insufficient_coverage_is_warning(self):
        """Short time span → insufficient_coverage warning."""
        dynamical_model_spec = _make_spec(model_clock="1d")
        # Only 3 days of data, need 10 * 24h = 240h
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": str(float(i)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(3)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        cov_issues = [i for i in _all_issues(result) if i.code == "insufficient_coverage"]
        assert len(cov_issues) == 1
        assert cov_issues[0].outcome == "failed"

    def test_data_coverage_is_independent_of_construct_temporal_status(self):
        """The stored data profile retains coverage evidence for every construct role."""
        dynamical_model_spec = _make_spec(model_clock=None, temporal_status="time_invariant")
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": str(float(i)),
                "anchor_time": f"2024-01-0{i + 1} 10:00",
            }
            for i in range(3)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        cov_issues = [i for i in _all_issues(result) if i.code == "insufficient_coverage"]
        assert len(cov_issues) == 1
        assert cov_issues[0].outcome == "failed"

    def test_weekly_granularity_needs_more_span(self):
        """Weekly granularity requires 10 * 168h = 1680h of coverage."""
        dynamical_model_spec = _make_spec(model_clock="1w")
        # 20 days < 70 days needed
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": str(float(i % 5)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        cov_issues = [i for i in _all_issues(result) if i.code == "insufficient_coverage"]
        assert len(cov_issues) == 1


# ==============================================================================
# UNIT TESTS: _check_timestamp_gaps
# ==============================================================================


class TestCheckTimestampGaps:
    """Test timestamp gap detection."""

    def test_no_large_gaps(self):
        """Regular daily data has no large gaps."""
        dynamical_model_spec = _make_spec(model_clock="1d")
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": str(float(i % 5)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        gap_issues = [i for i in _all_issues(result) if i.code == "large_timestamp_gap"]
        assert len(gap_issues) == 0

    def test_large_gap_warning(self):
        """Gap > 5x granularity → warning."""
        dynamical_model_spec = _make_spec(model_clock="1d")
        # 3 observations with a 10-day gap (>5x daily=120h)
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "1.0",
                "anchor_time": "2024-01-01 10:00",
            },
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "2.0",
                "anchor_time": "2024-01-02 10:00",
            },
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "3.0",
                "anchor_time": "2024-01-03 10:00",
            },
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "4.0",
                "anchor_time": "2024-01-04 10:00",
            },
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "5.0",
                "anchor_time": "2024-01-20 10:00",
            },
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "6.0",
                "anchor_time": "2024-01-21 10:00",
            },
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "7.0",
                "anchor_time": "2024-01-22 10:00",
            },
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "8.0",
                "anchor_time": "2024-01-23 10:00",
            },
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "9.0",
                "anchor_time": "2024-01-24 10:00",
            },
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "10.0",
                "anchor_time": "2024-01-25 10:00",
            },
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        gap_issues = [i for i in _all_issues(result) if i.code == "large_timestamp_gap"]
        assert len(gap_issues) == 1
        assert gap_issues[0].outcome == "failed"

    def test_skips_with_few_timestamps(self):
        """Fewer than 3 timestamps skips gap check."""
        dynamical_model_spec = _make_spec(model_clock="1d")
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "1.0",
                "anchor_time": "2024-01-01 10:00",
            },
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": "2.0",
                "anchor_time": "2024-06-01 10:00",
            },
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        gap_issues = [i for i in _all_issues(result) if i.code == "large_timestamp_gap"]
        assert len(gap_issues) == 0


# ==============================================================================
# UNIT TESTS: _check_hallucination_signals
# ==============================================================================


class TestCheckHallucinationSignals:
    """Test hallucination signal detection."""

    def test_clean_data_no_warning(self):
        """Normal data produces no hallucination warnings."""
        dynamical_model_spec = _make_spec(dtype="continuous")
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": str(float(i % 7 + 1)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        hall_issues = [i for i in _all_issues(result) if i.code == "suspicious_pattern"]
        assert len(hall_issues) == 0

    def test_excessive_duplicates_warning(self):
        """Over 50% same value in continuous data → warning."""
        dynamical_model_spec = _make_spec(dtype="continuous")
        # 15 out of 20 are 5.0
        values = ["5.0"] * 15 + ["1.0", "2.0", "3.0", "4.0", "6.0"]
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": v,
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i, v in enumerate(values)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        hall_issues = [i for i in _all_issues(result) if i.code == "suspicious_pattern"]
        assert any("5.0" in i.evidence for i in hall_issues)

    def test_arithmetic_sequence_warning(self):
        """Perfect arithmetic sequence → warning."""
        dynamical_model_spec = _make_spec(dtype="continuous")
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": str(float(i * 2)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        hall_issues = [i for i in _all_issues(result) if i.code == "suspicious_pattern"]
        assert any("arithmetic sequence" in i.evidence for i in hall_issues)

    def test_binary_exempt_from_duplicates(self):
        """Binary data with >50% same value is natural, not flagged."""
        dynamical_model_spec = _make_spec(dtype="binary")
        # 15 out of 20 are 1.0 — normal for binary
        values = ["1"] * 15 + ["0"] * 5
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": v,
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i, v in enumerate(values)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        hall_issues = [
            i
            for i in _all_issues(result)
            if i.code == "suspicious_pattern" and "duplicate" in i.evidence.lower()
        ]
        # No duplicate-based hallucination warning for binary
        assert len(hall_issues) == 0

    def test_count_exempt_from_duplicates(self):
        """Count data with >50% same value is natural, not flagged."""
        dynamical_model_spec = _make_spec(dtype="count")
        # Lots of zeros is typical for count data
        values = ["0"] * 15 + ["1", "2", "3", "4", "5"]
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": v,
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i, v in enumerate(values)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        hall_issues = [
            i
            for i in _all_issues(result)
            if i.code == "suspicious_pattern" and "duplicate" in i.evidence.lower()
        ]
        assert len(hall_issues) == 0


# ==============================================================================
# UNIT TESTS: _check_construct_correlations
# ==============================================================================


class TestCheckConstructCorrelations:
    """Test cross-indicator construct correlation checks."""

    def test_positive_correlation_no_issue(self):
        """Positively correlated indicators within a construct pass."""
        dynamical_model_spec = _make_spec(
            indicator_name="stress_score",
            construct_name="stress",
            extra_indicators=[
                {
                    "id": "indicator:4ff8be7491bd87d28af4",
                    "construct_id": "construct:6b04dc42c531e7091eb8",
                    "name": "stress_self_report",
                    "measurement_dtype": "continuous",
                },
            ],
        )
        records = []
        for i in range(20):
            val = float(i % 5 + 1)
            records.append(
                {
                    "indicator_id": "indicator:3696aef3ff6f446744e5",
                    "value": str(val),
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
            records.append(
                {
                    "indicator_id": "indicator:4ff8be7491bd87d28af4",
                    "value": str(val + 0.5),  # Positively correlated
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        corr_issues = [i for i in _all_issues(result) if i.code == "low_construct_correlation"]
        assert len(corr_issues) == 0

    def test_negative_correlation_warns(self):
        """Negatively correlated indicators → warning."""
        dynamical_model_spec = _make_spec(
            indicator_name="stress_score",
            construct_name="stress",
            extra_indicators=[
                {
                    "id": "indicator:4ff8be7491bd87d28af4",
                    "construct_id": "construct:6b04dc42c531e7091eb8",
                    "name": "stress_self_report",
                    "measurement_dtype": "continuous",
                },
            ],
        )
        records = []
        for i in range(20):
            val = float(i % 5 + 1)
            records.append(
                {
                    "indicator_id": "indicator:3696aef3ff6f446744e5",
                    "value": str(val),
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
            records.append(
                {
                    "indicator_id": "indicator:4ff8be7491bd87d28af4",
                    "value": str(10.0 - val),  # Negatively correlated
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        corr_issues = [i for i in _all_issues(result) if i.code == "low_construct_correlation"]
        assert len(corr_issues) == 1
        assert corr_issues[0].outcome == "failed"
        assert corr_issues[0].subject == "dataset"
        assert "construct:6b04dc42c531e7091eb8" in corr_issues[0].evidence
        assert result.findings[0].subject == "dataset"

    def test_single_indicator_skipped(self):
        """Constructs with only one indicator skip correlation check."""
        dynamical_model_spec = _make_spec()
        records = [
            {
                "indicator_id": "indicator:3696aef3ff6f446744e5",
                "value": str(float(i % 5)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        corr_issues = [i for i in _all_issues(result) if i.code == "low_construct_correlation"]
        assert len(corr_issues) == 0

    def test_insufficient_aligned_skipped(self):
        """Fewer than MIN_ALIGNED_FOR_CFA aligned observations skips check."""
        dynamical_model_spec = _make_spec(
            indicator_name="stress_score",
            construct_name="stress",
            extra_indicators=[
                {
                    "id": "indicator:4ff8be7491bd87d28af4",
                    "construct_id": "construct:6b04dc42c531e7091eb8",
                    "name": "stress_self_report",
                    "measurement_dtype": "continuous",
                },
            ],
        )
        # Different timestamps so nothing aligns
        records = []
        for i in range(20):
            records.append(
                {
                    "indicator_id": "indicator:3696aef3ff6f446744e5",
                    "value": str(float(i)),
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
            records.append(
                {
                    "indicator_id": "indicator:4ff8be7491bd87d28af4",
                    "value": str(float(20 - i)),
                    "anchor_time": f"2024-02-{i + 1:02d} 10:00",  # Different month
                }
            )
        result = validate_extraction(dynamical_model_spec, _create_worker_dfs(records))
        corr_issues = [i for i in _all_issues(result) if i.code == "low_construct_correlation"]
        assert len(corr_issues) == 0
