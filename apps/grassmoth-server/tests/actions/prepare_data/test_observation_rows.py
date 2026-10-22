"""Tests for storage/paths.py dataframe utility functions."""

import polars as pl
import pytest

from grassmoth.artifacts.identity import IndicatorId
from grassmoth.artifacts.observations.spec import ResolvedObservationSpec

pytestmark = pytest.mark.contract

# =============================================================================
# pivot_to_wide
# =============================================================================


class TestAnnotateObservationRows:
    def test_adds_observation_metadata_from_indicator_specs(self):
        from grassmoth_server.actions.prepare_data.rows import annotate_observation_rows

        raw = pl.DataFrame(
            {
                "indicator_id": ["indicator:3696aef3ff6f446744e5"],
                "value": ["4.0"],
                "timestamp": ["2024-01-01T00:00:00Z"],
            }
        )
        variables = (
            ResolvedObservationSpec(
                id=IndicatorId("indicator:3696aef3ff6f446744e5"),
                name="stress_score",
                measurement_dtype="continuous",
                aggregation="mean",
                observation_window="1d",
            ),
        )

        annotated = annotate_observation_rows(raw, variables)

        assert "timestamp" not in annotated.columns
        assert annotated["anchor_time"][0] == "2024-01-02T00:00:00"
        assert annotated["support_kind"][0] == "interval"
        assert annotated["summary_operator"][0] == "mean"
        assert annotated["anchor_policy"][0] == "support_end"
        assert annotated["support_start"][0] == "2024-01-01T00:00:00"
        assert annotated["support_end"][0] == "2024-01-02T00:00:00"

    def test_uses_indicator_specific_observation_window_when_present(self):
        from grassmoth_server.actions.prepare_data.rows import annotate_observation_rows

        raw = pl.DataFrame(
            {
                "indicator_id": ["indicator:8172ff8b9182b2e869c5"],
                "value": ["4.0"],
                "timestamp": ["2024-01-01T00:00:00Z"],
            }
        )
        variables = (
            ResolvedObservationSpec(
                id=IndicatorId("indicator:8172ff8b9182b2e869c5"),
                name="weekly_stress_score",
                measurement_dtype="continuous",
                aggregation="mean",
                observation_window="1w",
            ),
        )

        annotated = annotate_observation_rows(raw, variables)

        assert "timestamp" not in annotated.columns
        assert annotated["anchor_time"][0] == "2024-01-08T00:00:00"
        assert annotated["observation_window"][0] == "1w"
        assert annotated["support_start"][0] == "2024-01-01T00:00:00"
        assert annotated["support_end"][0] == "2024-01-08T00:00:00"

    def test_point_last_observations_anchor_at_window_end(self):
        from grassmoth_server.actions.prepare_data.rows import annotate_observation_rows

        raw = pl.DataFrame(
            {
                "indicator_id": ["indicator:5dc4b94693df7e0aef53"],
                "value": ["4.0"],
                "timestamp": ["2024-01-01T00:00:00Z"],
            }
        )
        variables = (
            ResolvedObservationSpec(
                id=IndicatorId("indicator:5dc4b94693df7e0aef53"),
                name="closing_mood",
                measurement_dtype="continuous",
                aggregation="last",
                observation_window="1d",
            ),
        )

        annotated = annotate_observation_rows(raw, variables)

        assert annotated["support_kind"][0] == "point"
        assert annotated["summary_operator"][0] == "last"
        assert annotated["anchor_policy"][0] == "support_end"
        assert annotated["anchor_time"][0] == "2024-01-02T00:00:00"
        assert annotated["support_start"][0] == "2024-01-01T00:00:00"
        assert annotated["support_end"][0] == "2024-01-02T00:00:00"
