"""Shared configuration for owned scientific and API values."""

from collections.abc import Mapping
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    field_serializer,
    field_validator,
    model_validator,
)

from nof1_causal_lab.artifacts.immutability import freeze


def _serialization_collections(value: object) -> object:
    """Give Pydantic ordinary containers while retaining each field's serializer."""
    if isinstance(value, Mapping):
        return {key: _serialization_collections(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_serialization_collections(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_serialization_collections(item) for item in value)
    return value


class Value(BaseModel):
    """An owned value has frozen fields, rejects extra fields and shares output presence.

    Collections and NumPy buffers are detached and recursively frozen at
    construction. Other foreign objects retain their native ownership constraints.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        json_schema_serialization_defaults_required=True,
    )

    @field_validator("*", mode="after")
    @classmethod
    def own_collections(cls, value: object) -> object:
        """Freeze incoming collections before Pydantic assigns them to the immutable value."""
        return freeze(value)

    @field_serializer("*", mode="wrap")
    def serialize_collections(  # noqa: ANN201 -- A wrap serializer return annotation replaces every owned field's schema.
        self, value: object, handler: SerializerFunctionWrapHandler
    ):
        """Convert frozen collections to serialization containers before invoking Pydantic's handler."""
        return handler(_serialization_collections(value))

    def revised(self, **changes: object) -> Self:
        """Validate a new value at its concrete owner, retaining no derived caches."""
        return self.model_validate(
            {**{name: getattr(self, name) for name in type(self).model_fields}, **changes}
        )

    @model_validator(mode="before")
    @classmethod
    def parse_projection(cls, value: object) -> object:
        """Serialized read-only projections never become independently supplied state."""
        if isinstance(value, Mapping) and cls.model_computed_fields:
            return {
                key: item for key, item in value.items() if key not in cls.model_computed_fields
            }
        return value
