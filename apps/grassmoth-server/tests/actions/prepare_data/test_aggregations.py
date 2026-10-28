"""Tests for aggregation utility functions.

Covers: _build_agg_expr, _encode_non_continuous, compute_indicators.
"""

from datetime import datetime

import polars as pl
import pytest

from grassmoth.artifacts.identity import IndicatorId
from grassmoth.artifacts.observations.semantics import SummaryOperator
from grassmoth.artifacts.observations.spec import LabeledMeasurement, NumericMeasurement
from grassmoth_server.actions.contracts.preparation import FilePreparationSpec
from grassmoth_server.actions.prepare_data.aggregations import (
    _build_agg_expr,
    _encode_non_continuous,
)
from grassmoth_server.actions.prepare_data.aggregations import (
    compute_indicators as _compute_indicators,
)
from grassmoth_server.actions.prepare_data.context import extraction_context

pytestmark = pytest.mark.contract


def compute_indicators(raw_df, indicators, model_clock, time_col, **kwargs):
    variables = indicators
    if not variables:
        variables = [
            {
                "observation": {
                    "id": "indicator:unused",
                    "name": "unused",
                    "measurement": {"dtype": "continuous"},
                    "aggregation": "mean",
                },
                "extraction": {"kind": "semantic", "how_to_measure": "Unused semantic variable"},
            }
        ]
    preparation = FilePreparationSpec.model_validate(
        {
            "source": {"files": ["source.csv"], **kwargs},
            "definition": {"default_window": model_clock, "variables": variables},
        }
    )
    return _compute_indicators(raw_df, extraction_context(preparation), time_col)


def _make_df(values: list[float]) -> pl.DataFrame:
    """Create a simple DataFrame with a 'value' column."""
    return pl.DataFrame({"value": values})


# =============================================================================
# _build_agg_expr
# =============================================================================


class TestBuildAggExpr:
    @pytest.mark.parametrize(
        ("agg_name", "values", "expected"),
        [
            (SummaryOperator.MEAN, [1.0, 2.0, 3.0], 2.0),
            (SummaryOperator.SUM, [1.0, 2.0, 3.0], 6.0),
            (SummaryOperator.COUNT, [1.0, 2.0, 3.0], 3.0),
            (SummaryOperator.FIRST, [7.0, 2.0, 3.0], 7.0),
            (SummaryOperator.LAST, [7.0, 2.0, 9.0], 9.0),
        ],
    )
    def test_supported_aggregations(self, agg_name, values, expected):
        df = _make_df(values)
        result = df.select(_build_agg_expr(agg_name))
        assert result["value"][0] == pytest.approx(expected)

    def test_std(self):
        df = _make_df([1.0, 2.0, 3.0])
        result = df.select(_build_agg_expr(SummaryOperator.STD))
        assert result["value"][0] == pytest.approx(1.0)  # sample std (ddof=1)

    @pytest.mark.parametrize(
        ("agg_name", "expected"),
        [
            (SummaryOperator.MEAN, 42.0),
            (SummaryOperator.SUM, 42.0),
            (SummaryOperator.COUNT, 1.0),
            (SummaryOperator.FIRST, 42.0),
            (SummaryOperator.LAST, 42.0),
        ],
    )
    def test_single_value(self, agg_name, expected):
        """Aggregating a single value works for the basic scalar reducers."""
        result = _make_df([42.0]).select(_build_agg_expr(agg_name))
        assert result["value"][0] == pytest.approx(expected), f"{agg_name} failed on single value"


# =============================================================================
# _encode_non_continuous
# =============================================================================


class TestEncodeNonContinuous:
    def test_binary_true_false(self):
        df = pl.DataFrame(
            {
                "indicator_id": [
                    "indicator:869e1d0209fb25a7fc06",
                    "indicator:869e1d0209fb25a7fc06",
                    "indicator:869e1d0209fb25a7fc06",
                ],
                "value": ["true", "false", "yes"],
            }
        )
        result = _encode_non_continuous(
            df, {"indicator:869e1d0209fb25a7fc06": NumericMeasurement(dtype="binary")}
        )
        values = result.sort("value")["value"].to_list()
        # "false" -> "0.0", "true" -> "1.0", "yes" -> "1.0"
        assert "0.0" in values
        assert "1.0" in values

    def test_ordinal_numeric_codes_passthrough(self):
        df = pl.DataFrame(
            {
                "indicator_id": [
                    "indicator:036fd134b9ad32d7a2ca",
                    "indicator:036fd134b9ad32d7a2ca",
                    "indicator:036fd134b9ad32d7a2ca",
                ],
                "value": ["0", "1", "2"],
            }
        )
        result = _encode_non_continuous(
            df,
            {
                "indicator:036fd134b9ad32d7a2ca": LabeledMeasurement(
                    dtype="ordinal", levels=("low", "medium", "high")
                )
            },
        )
        vals = sorted(float(v) for v in result["value"].to_list())
        assert vals == [0.0, 1.0, 2.0]

    def test_ordinal_out_of_range_code_becomes_null(self):
        df = pl.DataFrame(
            {
                "indicator_id": [
                    "indicator:036fd134b9ad32d7a2ca",
                    "indicator:036fd134b9ad32d7a2ca",
                ],
                "value": ["2", "3"],
            }
        )
        result = _encode_non_continuous(
            df,
            {
                "indicator:036fd134b9ad32d7a2ca": LabeledMeasurement(
                    dtype="ordinal", levels=("low", "medium", "high")
                )
            },
        ).sort("value", nulls_last=True)
        assert result["value"].to_list() == ["2", None]

    def test_continuous_passthrough(self):
        df = pl.DataFrame(
            {
                "indicator_id": [
                    "indicator:c5b118ae552981435d7b",
                    "indicator:c5b118ae552981435d7b",
                ],
                "value": [70.5, 80.2],
            }
        )
        result = _encode_non_continuous(
            df, {"indicator:c5b118ae552981435d7b": NumericMeasurement(dtype="continuous")}
        )
        assert result["value"].to_list() == [70.5, 80.2]

    def test_empty_dtype_lookup(self):
        df = pl.DataFrame({"indicator_id": ["indicator:1f4c67cecb9238ee1a80"], "value": [1.0]})
        result = _encode_non_continuous(df, {})
        assert result["value"].to_list() == [1.0]

    def test_mixed_indicators(self):
        """Only non-continuous indicators should be encoded; continuous left unchanged."""
        df = pl.DataFrame(
            {
                "indicator_id": [
                    "indicator:869e1d0209fb25a7fc06",
                    "indicator:c5b118ae552981435d7b",
                ],
                "value": ["true", "70.5"],
            }
        )
        result = _encode_non_continuous(
            df,
            {
                "indicator:869e1d0209fb25a7fc06": NumericMeasurement(dtype="binary"),
                "indicator:c5b118ae552981435d7b": NumericMeasurement(dtype="continuous"),
            },
        )
        mood_row = result.filter(pl.col("indicator_id") == "indicator:869e1d0209fb25a7fc06")
        weight_row = result.filter(pl.col("indicator_id") == "indicator:c5b118ae552981435d7b")
        assert float(mood_row["value"][0]) == 1.0
        assert weight_row["value"][0] == "70.5"


# =============================================================================
# compute_indicators
# =============================================================================


def _make_raw_df() -> pl.DataFrame:
    """Create a raw DataFrame spanning 3 days with heart_rate and steps."""
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 8, 0),
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 18, 0),
                datetime(2024, 1, 2, 9, 0),
                datetime(2024, 1, 2, 15, 0),
                datetime(2024, 1, 3, 10, 0),
            ],
            "heart_rate": [72.0, 85.0, 68.0, 74.0, 90.0, 70.0],
            "steps": [1000, 3000, 500, 2000, 4000, 1500],
        }
    )


class TestComputeIndicators:
    @pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
    def test_single_mean(self, time_unit):
        """Mean of heart_rate across 3 daily ticks."""
        df = _make_raw_df().with_columns(pl.col("timestamp").cast(pl.Datetime(time_unit)))
        indicators = [
            {
                "observation": {
                    "measurement": {"dtype": "continuous"},
                    "id": IndicatorId("indicator:c21b43949b3712e734c8"),
                    "name": "avg_hr",
                    "aggregation": "mean",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["heart_rate"],
                    "how_to_measure": "avg_hr",
                },
            },
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        assert result.columns == ["indicator_id", "value", "timestamp"]
        assert result["indicator_id"].to_list() == ["indicator:c21b43949b3712e734c8"] * 3
        # Day 1: mean(72, 85, 68) = 75.0
        values = [float(v) for v in result["value"].to_list()]
        assert abs(values[0] - 75.0) < 0.01

    def test_sum_aggregation(self):
        """Sum of steps across daily ticks."""
        df = _make_raw_df()
        indicators = [
            {
                "observation": {
                    "measurement": {"dtype": "continuous"},
                    "id": IndicatorId("indicator:a0ce08437c19d06aafd1"),
                    "name": "total_steps",
                    "aggregation": "sum",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["steps"],
                    "how_to_measure": "total_steps",
                },
            },
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        values = [float(v) for v in result["value"].to_list()]
        # Day 1: 1000+3000+500=4500, Day 2: 2000+4000=6000, Day 3: 1500
        assert values == [4500.0, 6000.0, 1500.0]

    def test_multiple_indicators(self):
        """Two computed indicators in one call."""
        df = _make_raw_df()
        indicators = [
            {
                "observation": {
                    "measurement": {"dtype": "continuous"},
                    "id": IndicatorId("indicator:c21b43949b3712e734c8"),
                    "name": "avg_hr",
                    "aggregation": "mean",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["heart_rate"],
                    "how_to_measure": "avg_hr",
                },
            },
            {
                "observation": {
                    "measurement": {"dtype": "continuous"},
                    "id": IndicatorId("indicator:a0ce08437c19d06aafd1"),
                    "name": "total_steps",
                    "aggregation": "sum",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["steps"],
                    "how_to_measure": "total_steps",
                },
            },
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        # 3 ticks * 2 indicators = 6 rows
        assert len(result) == 6
        assert set(result["indicator_id"].to_list()) == {
            "indicator:c21b43949b3712e734c8",
            "indicator:a0ce08437c19d06aafd1",
        }

    def test_output_schema(self):
        """Output columns are exactly {indicator, value, timestamp} as Utf8."""
        df = _make_raw_df()
        indicators = [
            {
                "observation": {
                    "measurement": {"dtype": "continuous"},
                    "id": IndicatorId("indicator:c21b43949b3712e734c8"),
                    "name": "avg_hr",
                    "aggregation": "mean",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["heart_rate"],
                    "how_to_measure": "avg_hr",
                },
            },
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        assert result.columns == ["indicator_id", "value", "timestamp"]
        assert result.schema["indicator_id"] == pl.Utf8
        assert result.schema["value"] == pl.Utf8
        assert result.schema["timestamp"] == pl.Utf8

    def test_empty_indicators(self):
        """Empty indicator list returns empty DataFrame with correct schema."""
        df = _make_raw_df()
        result = compute_indicators(df, [], "1d", "timestamp")
        assert result.columns == ["indicator_id", "value", "timestamp"]
        assert len(result) == 0

    def test_unsupported_aggregation_rejected_at_observation_boundary(self):
        """Computed extraction cannot bypass the owned summary vocabulary."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 18, 0),
                ],
                "hr": [70.0, 75.0, 80.0],  # increasing → positive slope
            }
        )
        indicators = [
            {
                "observation": {
                    "measurement": {"dtype": "continuous"},
                    "id": IndicatorId("indicator:cacaf060b8afc7a952d9"),
                    "name": "hr_trend",
                    "aggregation": "trend",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["hr"],
                    "how_to_measure": "hr_trend",
                },
            }
        ]
        with pytest.raises(ValueError, match="aggregation"):
            compute_indicators(df, indicators, "1d", "timestamp")

    def test_missing_source_column(self):
        """Missing source column is skipped with warning, not crash."""
        df = _make_raw_df()
        indicators = [
            {
                "observation": {
                    "measurement": {"dtype": "continuous"},
                    "id": IndicatorId("indicator:938f71ec0999bbbe342c"),
                    "name": "missing",
                    "aggregation": "mean",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["nonexistent_col"],
                    "how_to_measure": "missing",
                },
            },
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        assert len(result) == 0

    def test_null_source_values(self):
        """Source column with nulls aggregates correctly."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 18, 0),
                ],
                "hr": [72.0, None, 68.0],
            }
        )
        indicators = [
            {
                "observation": {
                    "measurement": {"dtype": "continuous"},
                    "id": IndicatorId("indicator:c21b43949b3712e734c8"),
                    "name": "avg_hr",
                    "aggregation": "mean",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["hr"],
                    "how_to_measure": "avg_hr",
                },
            }
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        assert len(result) == 1
        # mean(72, 68) = 70.0 (null ignored)
        assert abs(float(result["value"][0]) - 70.0) < 0.01

    def test_first_ignores_leading_nulls(self):
        """Point aggregations should use the first observed value, not the first row."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 9, 0),
                    datetime(2024, 1, 1, 10, 0),
                ],
                "care_setting": [None, "home", "clinic"],
            }
        )
        indicators = [
            {
                "observation": {
                    "id": IndicatorId("indicator:c0486b3cc6b5559e95d0"),
                    "name": "first_setting",
                    "measurement": {"dtype": "categorical", "levels": ["home", "clinic"]},
                    "aggregation": "first",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["care_setting"],
                    "how_to_measure": "first_setting",
                },
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["value"].to_list() == ["home"]

    def test_categorical_last_preserves_string_value(self):
        """Direct categorical computed indicators should preserve raw labels."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                ],
                "care_setting": ["home", "clinic"],
            }
        )
        indicators = [
            {
                "observation": {
                    "id": IndicatorId("indicator:c664114dcb2ded460d6e"),
                    "name": "last_setting",
                    "measurement": {"dtype": "categorical", "levels": ["home", "clinic"]},
                    "aggregation": "last",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["care_setting"],
                    "how_to_measure": "last_setting",
                },
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["value"].to_list() == ["clinic"]

    def test_ordinal_last_encodes_label_to_numeric_code(self):
        """Direct ordinal computed indicators should emit canonical integer codes."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                ],
                "mood_label": ["bad", "good"],
            }
        )
        indicators = [
            {
                "observation": {
                    "id": IndicatorId("indicator:5dc4b94693df7e0aef53"),
                    "name": "closing_mood",
                    "measurement": {"dtype": "ordinal", "levels": ["bad", "ok", "good"]},
                    "aggregation": "last",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["mood_label"],
                    "how_to_measure": "closing_mood",
                },
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["value"].to_list() == ["2"]

    def test_count_aggregation_counts_non_null_string_values(self):
        """Count aggregations should not null out string source columns before counting."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 9, 0),
                    datetime(2024, 1, 1, 10, 0),
                ],
                "message_text": ["alpha", None, "beta"],
            }
        )
        indicators = [
            {
                "observation": {
                    "id": IndicatorId("indicator:6a123c22171bbe49ee54"),
                    "name": "text_events",
                    "measurement": {"dtype": "count"},
                    "aggregation": "count",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["message_text"],
                    "how_to_measure": "text_events",
                },
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["value"].to_list() == ["2"]

    def test_computed_rule_multi_column_formula(self):
        """Computed rules can deterministically derive window values from multiple columns."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 2, 9, 0),
                ],
                "systolic_bp": [120.0, 150.0, 110.0],
                "diastolic_bp": [80.0, 90.0, 70.0],
            }
        )
        indicators = [
            {
                "observation": {
                    "id": IndicatorId("indicator:46df40a36557ed795b7b"),
                    "name": "map",
                    "measurement": {"dtype": "continuous"},
                    "aggregation": "mean",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["systolic_bp", "diastolic_bp"],
                    "computed_rule": "mean(diastolic_bp + (systolic_bp - diastolic_bp) / 3)",
                    "how_to_measure": "map",
                },
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        values = [float(value) for value in result["value"].to_list()]
        assert values[0] == pytest.approx((80 + (120 - 80) / 3 + 90 + (150 - 90) / 3) / 2)
        assert values[1] == pytest.approx(70 + (110 - 70) / 3)

    def test_computed_rule_filtered_count_preserves_zero_vs_null(self):
        """Filtered deterministic counts should distinguish observed zero from no observation."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 14, 0),
                    datetime(2024, 1, 2, 9, 0),
                    datetime(2024, 1, 3, 10, 0),
                ],
                "event_type": ["med_admin", "med_admin", "note", "med_admin", "note"],
                "admin_status": ["missed", "taken", None, "taken", None],
            }
        )
        indicators = [
            {
                "observation": {
                    "id": IndicatorId("indicator:097d80d6767b143a71b4"),
                    "name": "missed_doses",
                    "measurement": {"dtype": "count"},
                    "aggregation": "sum",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["event_type", "admin_status"],
                    "computed_rule": 'None if count_true(event_type == "med_admin") == 0 else sum(1 if (event_type == "med_admin" and admin_status == "missed") else 0)',
                    "how_to_measure": "missed_doses",
                },
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["timestamp"].to_list() == [
            "2024-01-01T00:00:00",
            "2024-01-02T00:00:00",
            "2024-01-03T00:00:00",
        ]
        assert result["value"].to_list() == ["1", "0", None]

    def test_direct_aggregations_materialize_empty_support_windows(self):
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 2, 10, 0),
                    datetime(2024, 1, 4, 10, 0),
                ],
                "event_id": ["a", None, "b"],
                "score": [5.0, None, 7.0],
            }
        )
        indicators = [
            {
                "observation": {
                    "id": IndicatorId("indicator:e8bcfd4304d709a5cce3"),
                    "name": "event_count",
                    "measurement": {"dtype": "count"},
                    "aggregation": "count",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["event_id"],
                    "how_to_measure": "event_count",
                },
            },
            {
                "observation": {
                    "id": IndicatorId("indicator:422841bed8b563beeec0"),
                    "name": "score_mean",
                    "measurement": {"dtype": "continuous"},
                    "aggregation": "mean",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["score"],
                    "how_to_measure": "score_mean",
                },
            },
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        counts = result.filter(pl.col("indicator_id") == "indicator:e8bcfd4304d709a5cce3")
        means = result.filter(pl.col("indicator_id") == "indicator:422841bed8b563beeec0")
        assert counts["timestamp"].to_list() == [
            "2024-01-01T00:00:00",
            "2024-01-02T00:00:00",
            "2024-01-03T00:00:00",
            "2024-01-04T00:00:00",
        ]
        assert counts["value"].to_list() == ["1", "0", None, "1"]
        assert means["value"].to_list() == ["5.0", None, None, "7.0"]

    def test_computed_rule_count_materializes_empty_support_windows(self):
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 2, 10, 0),
                    datetime(2024, 1, 4, 10, 0),
                ],
                "title": ["stress spike", "ordinary update", "another ordinary update"],
            }
        )
        indicators = [
            {
                "observation": {
                    "id": IndicatorId("indicator:42b9030df461ee342451"),
                    "name": "stress_mentions",
                    "measurement": {"dtype": "count"},
                    "aggregation": "count",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["title"],
                    "computed_rule": 'count_true(contains(lower(coalesce(title, "")), "stress"))',
                    "how_to_measure": "stress_mentions",
                },
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["timestamp"].to_list() == [
            "2024-01-01T00:00:00",
            "2024-01-02T00:00:00",
            "2024-01-03T00:00:00",
            "2024-01-04T00:00:00",
        ]
        assert result["value"].to_list() == ["1", "0", None, "0"]

    def test_computed_rule_window_flag_cannot_claim_point_semantics(self):
        """An any-over-window flag is not the last observed point measurement."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 2, 8, 0),
                    datetime(2024, 1, 2, 12, 0),
                    datetime(2024, 1, 3, 8, 0),
                    datetime(2024, 1, 3, 12, 0),
                ],
                "spo2_pct": [95.0, 94.0, 91.0, 95.0, None, None],
            }
        )
        indicators = [
            {
                "observation": {
                    "id": IndicatorId("indicator:86e4453f8f098e1007ef"),
                    "name": "low_spo2",
                    "measurement": {"dtype": "binary"},
                    "aggregation": "last",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["spo2_pct"],
                    "computed_rule": "1 if any(spo2_pct < 92) else (0 if count_non_null(spo2_pct) > 0 else None)",
                    "how_to_measure": "low_spo2",
                },
            }
        ]

        with pytest.raises(ValueError, match="no supported observation semantics"):
            compute_indicators(df, indicators, "1d", "timestamp")

    def test_computed_rule_cannot_bypass_summary_validation_with_a_dict(self):
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "reading": [1.0]})
        with pytest.raises(ValueError, match="produces 'sum' but aggregation is 'mean'"):
            compute_indicators(
                df,
                [
                    {
                        "observation": {
                            "id": IndicatorId("indicator:reading"),
                            "name": "reading",
                            "measurement": {"dtype": "continuous"},
                            "aggregation": "mean",
                        },
                        "extraction": {
                            "kind": "computed",
                            "source_columns": ["reading"],
                            "computed_rule": "sum(reading)",
                            "how_to_measure": "reading",
                        },
                    }
                ],
                "1d",
                "timestamp",
            )

    def test_computed_rule_contains_any_literal_list(self):
        """contains_any() should accept literal string lists in computed rules."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 2, 9, 0),
                ],
                "title_url": [
                    "https://facebook.com/some-post",
                    "https://example.com",
                    "https://reddit.com/r/polars",
                ],
            }
        )
        indicators = [
            {
                "observation": {
                    "id": IndicatorId("indicator:cce51902cd8da81b8457"),
                    "name": "social_media_hits",
                    "measurement": {"dtype": "count"},
                    "aggregation": "count",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["title_url"],
                    "computed_rule": 'count_true(contains_any(title_url, ["facebook.com", "reddit.com"]))',
                    "how_to_measure": "social_media_hits",
                },
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["timestamp"].to_list() == ["2024-01-01T00:00:00", "2024-01-02T00:00:00"]
        assert result["value"].to_list() == ["1", "1"]

    def test_computed_rule_nested_contains_any_with_if_else(self):
        """Nested computed rules should handle contains_any() list literals inside bool expressions."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 2, 9, 0),
                    datetime(2024, 1, 2, 14, 0),
                    datetime(2024, 1, 3, 10, 0),
                ],
                "title": [
                    "Stress management",
                    None,
                    "ordinary browsing",
                    None,
                    "nothing relevant",
                ],
                "title_url": [
                    None,
                    "https://example.com",
                    "https://burnout.example/article",
                    "https://example.com/other",
                    None,
                ],
            }
        )
        indicators = [
            {
                "observation": {
                    "id": IndicatorId("indicator:1d23877d34bd268f1d4d"),
                    "name": "stress_content_count",
                    "measurement": {"dtype": "count"},
                    "aggregation": "count",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["timestamp", "title", "title_url"],
                    "computed_rule": 'None if count_non_null(timestamp) == 0 else count_true(contains_any(lower(coalesce(title, "")), ["stress", "burnout"]) or contains_any(lower(coalesce(title_url, "")), ["stress", "burnout"]))',
                    "how_to_measure": "stress_content_count",
                },
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["timestamp"].to_list() == [
            "2024-01-01T00:00:00",
            "2024-01-02T00:00:00",
            "2024-01-03T00:00:00",
        ]
        assert result["value"].to_list() == ["1", "1", "0"]

    def test_timestamp_format_matches_bucket_by_clock(self):
        """Computed timestamps match the ISO format from bucket_by_clock."""
        df = _make_raw_df()
        indicators = [
            {
                "observation": {
                    "measurement": {"dtype": "continuous"},
                    "id": IndicatorId("indicator:c21b43949b3712e734c8"),
                    "name": "avg_hr",
                    "aggregation": "mean",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["heart_rate"],
                    "how_to_measure": "avg_hr",
                },
            },
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        # Should be ISO format: YYYY-MM-DDTHH:MM:SS
        assert result["timestamp"][0] == "2024-01-01T00:00:00"

    def test_timezone_aware_string_timestamps(self):
        """UTC-suffixed string timestamps should aggregate without parse errors."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    "2025-03-03T08:00:00Z",
                    "2025-03-03T12:00:00Z",
                    "2025-03-04T09:00:00Z",
                ],
                "heart_rate": [72.0, 84.0, 90.0],
            }
        )
        indicators = [
            {
                "observation": {
                    "measurement": {"dtype": "continuous"},
                    "id": IndicatorId("indicator:c21b43949b3712e734c8"),
                    "name": "avg_hr",
                    "aggregation": "mean",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["heart_rate"],
                    "how_to_measure": "avg_hr",
                },
            },
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["timestamp"].to_list() == ["2025-03-03T00:00:00", "2025-03-04T00:00:00"]
        assert [float(v) for v in result["value"].to_list()] == [78.0, 90.0]

    def test_hourly_clock(self):
        """Hourly model_clock produces every support tick in the observed span."""
        df = _make_raw_df()
        indicators = [
            {
                "observation": {
                    "measurement": {"dtype": "continuous"},
                    "id": IndicatorId("indicator:c21b43949b3712e734c8"),
                    "name": "avg_hr",
                    "aggregation": "mean",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["heart_rate"],
                    "how_to_measure": "avg_hr",
                },
            },
        ]
        result = compute_indicators(df, indicators, "1h", "timestamp")
        assert len(result) == 51
        assert result["timestamp"][0] == "2024-01-01T08:00:00"
        assert result["timestamp"][-1] == "2024-01-03T10:00:00"
        assert result["value"].null_count() == 45

    def test_indicator_specific_observation_window_overrides_model_clock(self):
        """Computed indicators bucket by their own support window, not the global clock."""
        df = _make_raw_df()
        indicators = [
            {
                "observation": {
                    "measurement": {"dtype": "continuous"},
                    "id": IndicatorId("indicator:f56b7b4807d8c6627ccb"),
                    "name": "weekly_steps",
                    "aggregation": "sum",
                    "observation_window": "1w",
                },
                "extraction": {
                    "kind": "computed",
                    "source_columns": ["steps"],
                    "how_to_measure": "weekly_steps",
                },
            },
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert len(result) == 1
        assert result["timestamp"].to_list() == ["2024-01-01T00:00:00"]
        assert float(result["value"][0]) == 12000.0

    def test_col_name_parameter(self):
        """_build_agg_expr with custom col_name works correctly."""
        df = pl.DataFrame({"heart_rate": [72.0, 85.0, 68.0]})
        expr = _build_agg_expr(SummaryOperator.MEAN, "heart_rate")
        result = df.select(expr)
        assert abs(result["value"][0] - 75.0) < 0.01


@pytest.mark.parametrize(
    ("fields", "aggregation", "expected"),
    [
        ({}, "last", [None, 4.0, None, None, None, None, 8.0]),
        ({"fill_null": "forward"}, "last", [None, 4.0, 4.0, 4.0, 4.0, 4.0, 8.0]),
        (
            {"fill_null": "forward", "fill_null_limit": 1},
            "last",
            [None, 4.0, 4.0, None, None, None, 8.0],
        ),
        (
            {"fill_null": "forward", "fill_null_limit": 0},
            "last",
            [None, 4.0, None, None, None, None, 8.0],
        ),
        ({"fill_null": "backward"}, "last", [4.0, 4.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
        (
            {"fill_null": "backward", "fill_null_limit": 1},
            "last",
            [4.0, 4.0, None, None, None, 8.0, 8.0],
        ),
        ({"fill_null": "min"}, "last", [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 8.0]),
        ({"fill_null": "max"}, "last", [8.0, 4.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
        ({"fill_null": "mean"}, "last", [6.0, 4.0, 6.0, 6.0, 6.0, 6.0, 8.0]),
        ({"fill_null": "zero"}, "last", [0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 8.0]),
        ({"fill_null": "one"}, "last", [1.0, 4.0, 1.0, 1.0, 1.0, 1.0, 8.0]),
        ({"fill_null": 2.5}, "last", [2.5, 4.0, 2.5, 2.5, 2.5, 2.5, 8.0]),
        ({}, "sum", [None, 6.0, None, None, None, None, 8.0]),
        ({"fill_null": 0}, "sum", [0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 8.0]),
        ({"fill_null": 0}, "count", [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ({"fill_null": "forward"}, "mean", [None, 3.0, 3.0, 3.0, 3.0, 3.0, 8.0]),
    ],
)
def test_fill_null_uses_polars_after_window_aggregation(fields, aggregation, expected):
    from grassmoth.artifacts.observations.preparation import (
        ComputedExtractionSpec,
        DataVariableSpec,
    )
    from grassmoth.artifacts.observations.spec import AuthoredObservationSpec

    indicator = DataVariableSpec(
        observation=AuthoredObservationSpec(
            observation_window=None,
            id="indicator:record",
            name="record",
            measurement=NumericMeasurement(
                dtype="count" if aggregation == "count" else "continuous"
            ),
            aggregation=aggregation,
        ),
        extraction=ComputedExtractionSpec(
            how_to_measure="Read the recorded value", source_columns=("reading",), **fields
        ),
    )
    raw = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 7),
                datetime(2026, 1, 1),
                datetime(2026, 1, 2),
                datetime(2026, 1, 4),
                datetime(2026, 1, 2, 12),
            ],
            "reading": [8.0, None, 2.0, None, 4.0],
        }
    )
    result = compute_indicators(raw, [indicator.model_dump(mode="json")], "1d", "timestamp")
    assert result["value"].cast(pl.Float64).to_list() == expected
    # Conversions precede filling, which also covers explicit null readings.
    if fields == {"fill_null": "forward"} and aggregation == "last":
        converted = indicator.revised(
            extraction=indicator.extraction.revised(computed_rule="last(reading) / 2")
        )
        result = compute_indicators(raw, [converted.model_dump(mode="json")], "1d", "timestamp")
        assert result["value"].cast(pl.Float64).to_list() == [None, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0]
