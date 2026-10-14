"""Detach owned collections and numeric buffers at their construction boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def freeze(value: object) -> object:
    """Own nested collections; NumPy buffers use immutable bytes as their storage."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, tuple | list):
        return tuple(freeze(item) for item in value)
    if isinstance(value, frozenset | set):
        return frozenset(freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Owned NumPy buffers require a non-object dtype")
        return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
    return value


def freeze_fields(self: DataclassInstance) -> None:
    """Initialize every field of an owned frozen dataclass before publication."""
    for field in fields(self):
        object.__setattr__(self, field.name, freeze(getattr(self, field.name)))
