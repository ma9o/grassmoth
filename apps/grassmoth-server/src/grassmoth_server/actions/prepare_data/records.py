"""Canonical extracted observation rows stored in the panel."""

from typing_extensions import TypedDict

from grassmoth.public import IndicatorId


class ObservationRecord(TypedDict):
    """Canonical serialized extraction observation row."""

    indicator_id: IndicatorId
    value: str | int | float | bool | None
    anchor_time: str | None  # noqa: V107 -- Polars reads this row key when constructing the observation frame.
    support_kind: str | None
    summary_operator: str | None
    anchor_policy: str | None
    observation_window: str | None
    support_start: str | None  # noqa: V107 -- Polars reads this row key when constructing the observation frame.
    support_end: str | None  # noqa: V107 -- Polars reads this row key when constructing the observation frame.
