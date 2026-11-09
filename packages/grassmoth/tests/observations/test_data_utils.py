"""Tests for storage/paths.py dataframe utility functions."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from grassmoth.artifacts.observations.dataset import SelectedObservations
from grassmoth.artifacts.observations.spec import NumericMeasurement, ResolvedObservationSpec
from grassmoth.observations.rows import pivot_to_wide
from grassmoth_test_support.helpers import observation_dataset

pytestmark = pytest.mark.contract


def _project(rows, grid=(0.0, 1.0, 2.0)):
    origin = datetime(2024, 1, 1, tzinfo=UTC)
    if rows.schema["anchor_time"].is_numeric():
        rows = rows.with_columns(
            pl.Series(
                "anchor_time", [origin + timedelta(days=value) for value in rows["anchor_time"]]
            )
        )
    variables = tuple(
        ResolvedObservationSpec(
            id=identity,
            name=identity,
            measurement=NumericMeasurement(dtype="continuous"),
            aggregation="last",
            observation_window="1d",
        )
        for identity in rows["indicator_id"].unique(maintain_order=True)
    )
    observations = observation_dataset(
        rows.with_columns(
            pl.col("anchor_time").alias("support_start"), pl.col("anchor_time").alias("support_end")
        ),
        variables,
        time_origin=origin,
    )
    selected = observations.select(variables)
    assert isinstance(selected, SelectedObservations)
    return pivot_to_wide(selected, np.asarray(grid))


# =============================================================================
# pivot_to_wide
# =============================================================================


class TestPivotToWide:
    def test_basic_pivot(self):
        """Simple long-to-wide conversion."""
        df = pl.DataFrame(
            {
                "anchor_time": [1.0, 2.0, 1.0, 2.0],
                "indicator_id": [
                    "indicator:1f4c67cecb9238ee1a80",
                    "indicator:1f4c67cecb9238ee1a80",
                    "indicator:3316cd345d83d02fe3fc",
                    "indicator:3316cd345d83d02fe3fc",
                ],
                "value": [10.0, 20.0, 30.0, 40.0],
            }
        )
        wide = _project(df)
        assert "time" in wide.columns
        assert "indicator:1f4c67cecb9238ee1a80" in wide.columns
        assert "indicator:3316cd345d83d02fe3fc" in wide.columns
        assert wide.height == 3

    def test_empty_dataframe(self):
        """Empty input preserves the declared model grid."""
        df = pl.DataFrame({"anchor_time": [], "indicator_id": [], "value": []})
        result = _project(df)
        assert result["time"].to_list() == [0.0, 1.0, 2.0]

    def test_datetime_to_fractional_days(self):
        """Datetime timestamps are converted to fractional days from t0."""
        t0 = datetime(2024, 1, 1)
        t1 = t0 + timedelta(days=1)
        t2 = t0 + timedelta(days=2)
        df = pl.DataFrame(
            {
                "anchor_time": [t0, t1, t2],
                "indicator_id": [
                    "indicator:1f4c67cecb9238ee1a80",
                    "indicator:1f4c67cecb9238ee1a80",
                    "indicator:1f4c67cecb9238ee1a80",
                ],
                "value": [1.0, 2.0, 3.0],
            }
        )
        wide = _project(df)
        assert "time" in wide.columns
        times = wide["time"].to_list()
        assert abs(times[0]) < 0.001  # t0 should be 0
        assert abs(times[1] - 1.0) < 0.001  # t1 should be ~1 day
        assert abs(times[2] - 2.0) < 0.001  # t2 should be ~2 days

    def test_sorted_by_time(self):
        """Output should be sorted by time."""
        df = pl.DataFrame(
            {
                "anchor_time": [3.0, 1.0, 2.0],
                "indicator_id": [
                    "indicator:1f4c67cecb9238ee1a80",
                    "indicator:1f4c67cecb9238ee1a80",
                    "indicator:1f4c67cecb9238ee1a80",
                ],
                "value": [30.0, 10.0, 20.0],
            }
        )
        wide = _project(df, grid=(0.0, 1.0, 2.0, 3.0))
        times = wide["time"].to_list()
        assert times == sorted(times)

    def test_missing_values_as_null(self):
        """Indicators without values at certain times should be null."""
        df = pl.DataFrame(
            {
                "anchor_time": [1.0, 2.0, 2.0],
                "indicator_id": [
                    "indicator:1f4c67cecb9238ee1a80",
                    "indicator:1f4c67cecb9238ee1a80",
                    "indicator:3316cd345d83d02fe3fc",
                ],
                "value": [10.0, 20.0, 30.0],
            }
        )
        wide = _project(df)
        # y has no value at t=1, so it should be null
        y_at_t1 = wide.filter(pl.col("time") == 1.0)["indicator:3316cd345d83d02fe3fc"].to_list()
        assert y_at_t1[0] is None

    def test_string_timestamps_parsed(self):
        """String timestamps should be parsed to datetime."""
        df = pl.DataFrame(
            {
                "anchor_time": ["2024-01-01", "2024-01-02"],
                "indicator_id": [
                    "indicator:1f4c67cecb9238ee1a80",
                    "indicator:1f4c67cecb9238ee1a80",
                ],
                "value": [1.0, 2.0],
            }
        )
        wide = _project(df)
        assert "time" in wide.columns
        times = wide["time"].to_list()
        assert abs(times[0]) < 0.001
        assert abs(times[1] - 1.0) < 0.001

    def test_string_values_cast_to_float(self):
        """String values should be cast to Float64."""
        df = pl.DataFrame(
            {
                "anchor_time": [1.0, 2.0],
                "indicator_id": [
                    "indicator:1f4c67cecb9238ee1a80",
                    "indicator:1f4c67cecb9238ee1a80",
                ],
                "value": ["10.5", "20.3"],
            }
        )
        wide = _project(df)
        assert wide["indicator:1f4c67cecb9238ee1a80"].dtype == pl.Float64
        assert abs(wide["indicator:1f4c67cecb9238ee1a80"][1] - 10.5) < 0.001

    def test_duplicate_values_rejected_at_admission(self):
        """The series owner rejects duplicate coordinates before pivoting."""
        df = pl.DataFrame(
            {
                "anchor_time": [1.0, 1.0, 2.0],
                "indicator_id": [
                    "indicator:1f4c67cecb9238ee1a80",
                    "indicator:1f4c67cecb9238ee1a80",
                    "indicator:1f4c67cecb9238ee1a80",
                ],
                "value": [10.0, 20.0, 30.0],
            }
        )
        with pytest.raises(ValueError, match="duplicate"):
            _project(df)

    def test_single_indicator(self):
        """Minimal case with just one indicator."""
        df = pl.DataFrame(
            {
                "anchor_time": [1.0],
                "indicator_id": ["indicator:1f4c67cecb9238ee1a80"],
                "value": [42.0],
            }
        )
        wide = _project(df)
        assert wide.height == 3
        assert "indicator:1f4c67cecb9238ee1a80" in wide.columns
        assert wide["indicator:1f4c67cecb9238ee1a80"][1] == 42.0


class TestPivotToWideSparsity:
    """Pivoting preserves sparse measurements for the data profiler."""

    def test_pivot_warns_on_sparse_matrix(self, caplog):
        """Sparse multi-granularity data triggers a warning."""
        import logging

        rows = []
        for h in range(24):
            rows.append(
                {
                    "indicator_id": "indicator:f0e6ea8ca02efe08067d",
                    "value": float(h),
                    "anchor_time": h,
                }
            )
        rows.append(
            {"indicator_id": "indicator:4ba776ce45cadd77efac", "value": 5.0, "anchor_time": 0}
        )
        rows.append(
            {"indicator_id": "indicator:683acb864872efccc82b", "value": 9.0, "anchor_time": 0}
        )

        raw = pl.DataFrame(rows)
        logger = logging.getLogger("grassmoth.observations.rows")
        logger.propagate = True
        with caplog.at_level(logging.WARNING, logger="grassmoth.observations.rows"):
            wide = _project(raw, grid=tuple(range(24)))

        assert wide.height == 24
        assert np.isnan(wide.to_numpy()).sum() == 46

    def test_pivot_no_warning_on_complete_matrix(self, caplog):
        """Complete data should not trigger sparsity warning."""
        import logging

        rows = []
        for t in range(10):
            rows.append(
                {
                    "indicator_id": "indicator:c66a68cb408e1c918650",
                    "value": float(t),
                    "anchor_time": t,
                }
            )
            rows.append(
                {
                    "indicator_id": "indicator:963e8fd77d3423ee1b6c",
                    "value": float(t * 2),
                    "anchor_time": t,
                }
            )

        raw = pl.DataFrame(rows)
        with caplog.at_level(logging.WARNING, logger="grassmoth.observations.rows"):
            _project(raw, grid=tuple(range(10)))

        assert not any("Sparse" in msg for msg in caplog.messages)


class TestPivotToWideTimezoneStrings:
    def test_utc_string_timestamps_parsed(self):
        """UTC timestamps with timezone suffix should parse to fractional days."""
        df = pl.DataFrame(
            {
                "anchor_time": ["2024-01-01T00:00:00Z", "2024-01-02T12:00:00Z"],
                "indicator_id": [
                    "indicator:1f4c67cecb9238ee1a80",
                    "indicator:1f4c67cecb9238ee1a80",
                ],
                "value": [1.0, 2.0],
            }
        )
        wide = _project(df, grid=(0.0, 1.5))

        assert wide.schema["time"] == pl.Float64
        times = wide["time"].to_list()
        assert abs(times[0]) < 0.001
        assert abs(times[1] - 1.5) < 0.001
