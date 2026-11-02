"""Support windows are formatted as worker text for semantic extraction."""

from datetime import datetime

import polars as pl
import pytest

from grassmoth_server.actions.prepare_data.window_text import format_window_chunk

pytestmark = pytest.mark.contract


class TestFormatWindowChunk:
    def test_basic_formatting(self):
        events = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 8, 0), datetime(2024, 1, 1, 12, 30)],
                "action": ["searched python", "viewed stackoverflow"],
            }
        )
        chunk = [("2024-01-01", events)]
        text = format_window_chunk(chunk, "timestamp", ["action"])
        assert "## Window Start: 2024-01-01" in text
        assert "searched python" in text
        assert "viewed stackoverflow" in text
        assert "08:00" in text
        assert "12:30" in text

    def test_multiple_windows(self):
        events1 = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 8, 0)],
                "action": ["event1"],
            }
        )
        events2 = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 2, 9, 0)],
                "action": ["event2"],
            }
        )
        chunk = [("2024-01-01", events1), ("2024-01-02", events2)]
        text = format_window_chunk(chunk, "timestamp", ["action"])
        assert "## Window Start: 2024-01-01" in text
        assert "## Window Start: 2024-01-02" in text
        assert "event1" in text
        assert "event2" in text

    def test_truncation(self):
        """Events exceeding max_events_per_window should be truncated."""
        n = 50
        events = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, h % 24, m) for h, m in zip(range(n), range(n), strict=True)
                ],
                "action": [f"event_{i}" for i in range(n)],
            }
        )
        chunk = [("2024-01-01", events)]
        text = format_window_chunk(chunk, "timestamp", ["action"], max_events_per_window=20)
        assert "showing 20 of 50 events" in text

    def test_truncation_is_deterministic(self):
        n = 50
        events = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, h % 24, m) for h, m in zip(range(n), range(n), strict=True)
                ],
                "action": [f"event_{i}" for i in range(n)],
            }
        )
        chunk = [("2024-01-01", events)]

        text_a = format_window_chunk(chunk, "timestamp", ["action"], max_events_per_window=20)
        text_b = format_window_chunk(chunk, "timestamp", ["action"], max_events_per_window=20)

        assert text_a == text_b

    def test_no_truncation_under_limit(self):
        events = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, i, 0) for i in range(5)],
                "action": [f"event_{i}" for i in range(5)],
            }
        )
        chunk = [("2024-01-01", events)]
        text = format_window_chunk(chunk, "timestamp", ["action"], max_events_per_window=300)
        assert "showing" not in text

    def test_multiple_display_columns(self):
        events = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 8, 0)],
                "col1": ["val1"],
                "col2": ["val2"],
            }
        )
        chunk = [("2024-01-01", events)]
        text = format_window_chunk(chunk, "timestamp", ["col1", "col2"])
        assert "val1" in text
        assert "val2" in text
