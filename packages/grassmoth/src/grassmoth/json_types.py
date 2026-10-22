"""Named types for JSON-safe values at transport boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | JsonArray | JsonObject
type JsonArray = Sequence[JsonValue]
type JsonObject = Mapping[str, JsonValue]
