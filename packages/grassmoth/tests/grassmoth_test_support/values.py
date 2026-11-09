"""Validated arbitrary variants for test fixtures, outside production value APIs."""

from grassmoth.artifacts.base import Value


def revise_fixture[T: Value](value: T, /, **changes: object) -> T:
    """Parse a test variant while retaining owner validation and collection ownership."""
    aliases = {name: field.alias or name for name, field in type(value).model_fields.items()}
    return type(value).model_validate(
        {
            **{alias: getattr(value, name) for name, alias in aliases.items()},
            **{aliases.get(name, name): changed for name, changed in changes.items()},
        }
    )
