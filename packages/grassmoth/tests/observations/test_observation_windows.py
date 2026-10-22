"""Preparation and simulation resolve the same calendar support, without invented counts."""

import pytest
from pydantic import TypeAdapter, ValidationError

from grassmoth.artifacts.observations.duration import CalendarDuration
from grassmoth.artifacts.observations.spec import (
    ObservationWindow,
    canonical_observation_window,
)

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("source", ["0mo", "0y", "-2y", "1.5mo", "2y1d", "mo", ""])
def test_calendar_window_parser_rejects_invalid_periods(source):
    with pytest.raises(ValidationError):
        TypeAdapter(ObservationWindow).validate_python(source)


@pytest.mark.parametrize(
    ("source", "canonical"),
    [
        ("1y", "12mo"),
        ("7y", "84mo"),
        ("003mo", "3mo"),
        ("2Y", "24mo"),
        ("2 years", "24mo"),
        ("1y6mo", "18mo"),
    ],
)
def test_calendar_window_wire_spelling_and_measurement_equivalence(source, canonical):
    adapter = TypeAdapter(ObservationWindow)
    parsed = adapter.validate_python(source)
    assert isinstance(parsed, CalendarDuration)
    assert adapter.validate_python(parsed) is parsed
    assert adapter.dump_json(parsed) == f'"{source}"'.encode()
    assert adapter.validate_json(adapter.dump_json(parsed)) == parsed
    assert canonical_observation_window(parsed) == canonical_observation_window(canonical)
    assert canonical_observation_window("1y") != canonical_observation_window("365d")
