"""Source rows group into complete support windows, keeping empty windows in the span."""

from datetime import datetime

import polars as pl
import pytest

from grassmoth.actions.prepare_data.rows import support_windows

pytestmark = pytest.mark.contract

# =============================================================================
# support_windows
# =============================================================================


def _make_events_df(timestamps: list[str], values: list[int] | None = None) -> pl.DataFrame:
    """Create a simple events DataFrame with a timestamp column."""
    n = len(timestamps)
    return pl.DataFrame(
        {
            "timestamp": [datetime.fromisoformat(t) for t in timestamps],
            "value": values or list(range(n)),
        }
    )


class TestSupportWindows:
    def test_daily_bucketing(self):
        df = _make_events_df(
            [
                "2024-01-01T08:00:00",
                "2024-01-01T12:00:00",
                "2024-01-01T18:00:00",
                "2024-01-02T09:00:00",
                "2024-01-02T15:00:00",
            ]
        )
        ticks = support_windows(df, "1d", "timestamp")
        assert len(ticks) == 2
        assert ticks[0][0] == "2024-01-01T00:00:00"
        assert len(ticks[0][1]) == 3  # 3 events on day 1
        assert len(ticks[1][1]) == 2  # 2 events on day 2

    def test_hourly_bucketing(self):
        df = _make_events_df(
            [
                "2024-01-01T08:15:00",
                "2024-01-01T08:45:00",
                "2024-01-01T09:30:00",
            ]
        )
        ticks = support_windows(df, "1h", "timestamp")
        assert len(ticks) == 2  # 08:xx and 09:xx
        assert len(ticks[0][1]) == 2
        assert len(ticks[1][1]) == 1

    def test_string_timestamp_column(self):
        """String timestamps should be auto-parsed."""
        df = pl.DataFrame(
            {
                "timestamp": ["2024-01-01T10:00:00", "2024-01-02T10:00:00"],
                "value": [1, 2],
            }
        )
        ticks = support_windows(df, "1d", "timestamp")
        assert len(ticks) == 2

    def test_timezone_aware_string_timestamp_column(self):
        """UTC-suffixed string timestamps should bucket without parse errors."""
        df = pl.DataFrame(
            {
                "timestamp": ["2025-03-03T10:00:00Z", "2025-03-04T10:00:00Z"],
                "value": [1, 2],
            }
        )

        ticks = support_windows(df, "1d", "timestamp")

        assert [tick_id for tick_id, _ in ticks] == [
            "2025-03-03T00:00:00",
            "2025-03-04T00:00:00",
        ]

    def test_materializes_empty_windows_between_observed_ticks(self):
        df = _make_events_df(
            [
                "2024-01-01T08:00:00",
                "2024-01-03T10:00:00",
            ]
        )

        ticks = support_windows(df, "1d", "timestamp")

        assert [tick_id for tick_id, _ in ticks] == [
            "2024-01-01T00:00:00",
            "2024-01-02T00:00:00",
            "2024-01-03T00:00:00",
        ]
        assert [len(events) for _, events in ticks] == [1, 0, 1]

    def test_chronological_ordering(self):
        """Ticks should be sorted chronologically."""
        df = _make_events_df(
            [
                "2024-01-03T10:00:00",
                "2024-01-01T10:00:00",
                "2024-01-02T10:00:00",
            ]
        )
        ticks = support_windows(df, "1d", "timestamp")
        tick_ids = [t[0] for t in ticks]
        assert tick_ids == sorted(tick_ids)

    def test_empty_df(self):
        df = pl.DataFrame(schema={"timestamp": pl.Datetime, "value": pl.Int64})
        ticks = support_windows(df, "1d", "timestamp")
        assert ticks == []

    def test_no_tick_column_in_output(self):
        """The internal __tick__ column should not appear in output DataFrames."""
        df = _make_events_df(["2024-01-01T10:00:00", "2024-01-01T15:00:00"])
        ticks = support_windows(df, "1d", "timestamp")
        assert "__tick__" not in ticks[0][1].columns
