"""Shared authored text whose content survives whitespace normalization."""

from typing import Annotated

from pydantic import AfterValidator, AnyUrl, StringConstraints, TypeAdapter, WithJsonSchema

NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _parse_reference_url(value: str) -> str:
    """Parse a cited URL once and retain its canonical spelling as an ordinary string."""
    return str(AnyUrl(value))


ReferenceUrl = Annotated[
    str,
    AfterValidator(_parse_reference_url),
    WithJsonSchema(TypeAdapter(AnyUrl).json_schema()),
]
