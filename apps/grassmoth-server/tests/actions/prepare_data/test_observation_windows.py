"""Preparation and simulation resolve the same calendar support, without invented counts."""

from datetime import datetime

import polars as pl
import pytest

from grassmoth.artifacts.observations.duration import CalendarDuration
from grassmoth.artifacts.observations.rows import validate_observation_rows
from grassmoth.artifacts.observations.semantics import AnchorPolicy
from grassmoth.artifacts.observations.spec import (
    ResolvedObservationSpec,
)
from grassmoth.artifacts.observations.time import ObservationInstant
from grassmoth.observations.rows import observation_window_boundaries
from grassmoth_server.actions.contracts.preparation import FilePreparationSpec
from grassmoth_server.actions.prepare_data.aggregations import compute_indicators
from grassmoth_server.actions.prepare_data.context import extraction_context
from grassmoth_server.actions.prepare_data.rows import (
    annotate_observation_rows,
)

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    ("window", "start", "missing", "end", "days"),
    [
        ("1mo", "2023-02-01", "2023-03-01", "2023-04-01", 28),
        ("1mo", "2024-02-01", "2024-03-01", "2024-04-01", 29),
        ("1y", "2024-01-01", "2025-01-01", "2026-01-01", 366),
        ("2y", "1936-01-01", "1938-01-01", "1940-01-01", 731),
        ("2y", "1938-01-01", "1940-01-01", "1942-01-01", 730),
        ("3mo", "2024-01-01", "2024-04-01", "2024-07-01", 91),
        ("18mo", "2024-01-01", "2025-07-01", "2027-01-01", 547),
        ("1 year 6 months", "2024-01-01", "2025-07-01", "2027-01-01", 547),
        ("3y", "2024-01-01", "2027-01-01", "2030-01-01", 1096),
        ("7y", "2019-01-01", "2026-01-01", "2033-01-01", 2557),
        ("7d", "2024-02-29", "2024-03-07", "2024-03-14", 7),
    ],
)
def test_period_counts_keep_exact_bounds_and_missing_windows(window, start, missing, end, days):
    preparation = FilePreparationSpec.model_validate(
        {
            "source": {"files": ["counts.csv"], "start": start, "end": end},
            "definition": {
                "default_window": "1d",
                "variables": [
                    {
                        "observation": {
                            "id": "indicator:count",
                            "name": "Count",
                            "measurement_dtype": "count",
                            "aggregation": "sum",
                            "observation_window": window,
                        },
                        "extraction": {
                            "kind": "computed",
                            "how_to_measure": "Retain each reported period count.",
                            "source_columns": ["count"],
                        },
                    }
                ],
            },
        }
    )
    context = extraction_context(preparation)
    variable = context.indicators[0].observation.resolved(context.window(context.indicators[0]))
    raw = pl.DataFrame({"timestamp": [start, missing], "count": [10, None]})
    rows = validate_observation_rows(
        annotate_observation_rows(compute_indicators(raw, context, "timestamp"), (variable,)),
        (variable,),
    )
    assert rows["value"].to_list() == [10, None]
    assert (rows["support_end"][0] - rows["support_start"][0]).days == days
    assert rows["support_start"][0] == datetime.fromisoformat(start)

    # Simulation resolves these same windows from their end anchors.
    lower, upper = ObservationInstant.support_bounds(
        pl.col("anchor_time"), variable.observation_window, anchor_policy=AnchorPolicy.SUPPORT_END
    )
    simulated = rows.select(lower.alias("support_start"), upper.alias("support_end"))
    assert simulated.equals(rows.select("support_start", "support_end"))
    if isinstance(variable.observation_window, CalendarDuration):
        boundaries = observation_window_boundaries(
            variable.observation_window, datetime.fromisoformat(start), datetime.fromisoformat(end)
        )
        assert boundaries == tuple(datetime.fromisoformat(value) for value in (start, missing, end))
        # A calendar count cannot be attached to a partial reporting period.
        unaligned = pl.DataFrame({"anchor_time": [datetime.fromisoformat(start).replace(day=2)]})
        assert unaligned.select(lower, upper.alias("end")).row(0) == (None, None)
        if variable.observation_window.months > 12:
            interior_year = datetime.fromisoformat(start).replace(year=int(start[:4]) + 1)
            unaligned_year = pl.DataFrame({"anchor_time": [interior_year]})
            assert unaligned_year.select(lower, upper.alias("end")).row(0) == (None, None)


def test_mixed_indicator_windows_keep_their_own_calendar_bounds():
    windows = ("1mo", "3mo", "2y", "7d")
    variables = tuple(
        ResolvedObservationSpec.model_validate(
            {
                "id": "indicator:" + window,
                "name": window,
                "measurement_dtype": "count",
                "aggregation": "sum",
                "observation_window": window,
            }
        )
        for window in windows
    )
    raw = pl.DataFrame(
        {
            "indicator_id": [variable.id for variable in variables],
            "value": [10, 20, 30, 40],
            "timestamp": ["1938-01-01"] * len(variables),
        }
    )
    rows = validate_observation_rows(annotate_observation_rows(raw, variables), variables)
    by_indicator = {row["indicator_id"]: row for row in rows.to_dicts()}
    for window, end in zip(
        windows, ("1938-02-01", "1938-04-01", "1940-01-01", "1938-01-08"), strict=True
    ):
        assert by_indicator["indicator:" + window]["support_start"] == datetime(1938, 1, 1)
        assert by_indicator["indicator:" + window]["support_end"] == datetime.fromisoformat(end)
