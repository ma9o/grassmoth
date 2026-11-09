"""Validated arbitrary variants for test fixtures, outside production value APIs."""

from grassmoth.artifacts.base import Value


def revise_fixture[T: Value](value: T, /, **changes: object) -> T:
    """Parse a test variant while retaining owner validation and collection ownership."""
    return type(value).model_validate(
        {**{name: getattr(value, name) for name in type(value).model_fields}, **changes}
    )
