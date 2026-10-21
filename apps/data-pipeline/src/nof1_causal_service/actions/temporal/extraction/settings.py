"""Resolve extraction settings against configured defaults."""

from __future__ import annotations


def first_config_value[T](*values: T | None) -> T | None:
    """Return the first non-None configuration choice, preserving false and zero values."""
    for value in values:
        if value is not None:
            return value
    return None
