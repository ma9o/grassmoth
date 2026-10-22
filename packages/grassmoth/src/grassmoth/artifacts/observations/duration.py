"""Parse positive fixed and calendar durations without conflating their units."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

from dateutil.relativedelta import relativedelta
from pydantic_core import core_schema
from pytimeparse2 import parse

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema

type DurationUnit = Literal["s", "m", "h", "d", "w"]
_UNIT_SECONDS: dict[DurationUnit, int] = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
_DURATION_RE = re.compile(r"([0-9]+)(s|m|h|d|w)")


@dataclass(frozen=True, init=False)
class Duration:
    """A fixed interval; days and weeks mean exactly 86400 and 604800 seconds."""

    source: str
    count: int
    unit: DurationUnit

    def __init__(self, source: str) -> None:
        """Parse a positive integer duration with a seconds, minutes, hours, days, or weeks suffix.

        Args:
            source: Compact duration such as ``30m`` or ``7d``.

        Raises:
            ValueError: The spelling does not match a supported integer duration or
                the duration is zero.
        """
        match = _DURATION_RE.fullmatch(source)
        if match is None:
            raise ValueError(
                f"Invalid duration: {source!r}. Expected <int><unit> with s, m, h, d, w"
            )
        count = int(match[1])
        if count == 0:
            raise ValueError("Duration must be positive (got 0)")
        unit = next(unit for unit in _UNIT_SECONDS if unit == match[2])
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "unit", unit)

    @property
    def seconds(self) -> int:
        """Duration expressed as an integer number of seconds."""
        return self.count * _UNIT_SECONDS[self.unit]

    @property
    def days(self) -> float:
        """Lower the exact interval to the numerical engine's day axis."""
        return self.seconds / 86400

    @property
    def polars_interval(self) -> str:
        """Fixed-unit spelling, including Polars' Monday alignment for week buckets."""
        return self.source

    @override
    def __str__(self) -> str:
        return self.source

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: object, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Accept parsed durations or duration strings and serialize them with their original spelling."""
        return _duration_schema(cls)


@dataclass(frozen=True, init=False)
class CalendarDuration:
    """Whole calendar months, resolved at UTC boundaries anchored to January 1970."""

    source: str
    months: int

    def __init__(self, source: str) -> None:
        """Parse calendar units with pytimeparse2 and retain exact dateutil month counts.

        Args:
            source: Calendar duration such as ``3mo``, ``2 years`` or ``1y6mo``.

        Raises:
            ValueError: The duration is not a positive, whole number of calendar months.
        """
        parsed = parse(source, as_timedelta=True)
        if not isinstance(parsed, relativedelta) or parsed != relativedelta(
            years=parsed.years, months=parsed.months
        ):
            raise ValueError(f"Invalid calendar duration: {source!r}. Use whole months or years")
        months = parsed.years * 12 + parsed.months
        if months <= 0:
            raise ValueError("Calendar duration must be positive")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "months", months)

    @property
    def polars_interval(self) -> str:
        """Lower parsed calendar months to Polars without approximating elapsed days."""
        return f"{self.months}mo"

    @override
    def __str__(self) -> str:
        return self.source

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: object, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Parse strings at the scalar boundary and retain the same string wire contract."""
        return _duration_schema(cls)


def _duration_schema(owner: type[Duration] | type[CalendarDuration]) -> CoreSchema:
    parsed = core_schema.no_info_after_validator_function(owner, core_schema.str_schema())
    return core_schema.json_or_python_schema(
        json_schema=parsed,
        python_schema=core_schema.union_schema([core_schema.is_instance_schema(owner), parsed]),
        serialization=core_schema.plain_serializer_function_ser_schema(
            str, return_schema=core_schema.str_schema()
        ),
    )


def parse_observation_window(source: str) -> Duration | CalendarDuration:
    """Parse the fixed-clock spelling or a library-owned calendar expression."""
    return Duration(source) if _DURATION_RE.fullmatch(source) else CalendarDuration(source)


__all__ = ["CalendarDuration", "Duration", "parse_observation_window"]
