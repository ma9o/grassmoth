"""Exact numerical values and shared vector selections owned by scientific fields."""

from __future__ import annotations

import hashlib
import io
import math
from functools import cached_property
from typing import Annotated, Self, override

import numpy as np
from pydantic import Field, FiniteFloat, model_serializer, model_validator

from grassmoth.artifacts.base import Value


class NumericalArray(Value):
    """An immutable NPY buffer carried as a MessagePack binary value.

    The NPY header owns dtype and shape. Buffers use little-endian, row-major
    storage; numerical decoding belongs to the boundary that consumes them.
    """

    npy: bytes = Field(
        description=(
            "Lossless NPY bytes, including dtype and dimensions, in MessagePack. "
            "JSON action responses replace the entire NumericalArray with nested lists "
            "in that shape (a scalar for a zero-dimensional array); non-finite values are null."
        ),
        json_schema_extra={"tsType": "Uint8Array"},
    )

    @model_serializer  # noqa: V105 -- Pydantic invokes the registered binary serialization hook.
    def serialize_buffer(self):  # noqa: ANN201 -- Preserve the owned NPY field schema, including its binary TypeScript type.
        """Serialize only the buffer, independently of decoded and layout caches."""
        return {"npy": self.npy}

    @override
    def __eq__(self, other: object) -> bool:
        """Decoded caches do not participate in numerical value identity."""
        return isinstance(other, NumericalArray) and self.npy == other.npy

    @classmethod
    def from_numpy(cls, values: np.ndarray) -> Self:
        """Own a portable, lossless encoding of a numerical value."""
        array = np.asarray(values)
        if array.dtype.kind not in "bifu" or array.dtype.itemsize not in (1, 2, 4, 8):
            raise TypeError(
                "Numerical buffers require boolean, integer or floating dtypes up to 64 bits"
            )
        buffer = io.BytesIO()
        np.save(
            buffer,
            np.asarray(array, dtype=array.dtype.newbyteorder("<"), order="C"),
            allow_pickle=False,
        )
        return cls(npy=buffer.getvalue())

    @cached_property
    def identity(self) -> str:
        """Content identity used only by numerical storage and serialization."""
        return hashlib.sha256(self.npy).hexdigest()

    @cached_property
    def values(self) -> np.ndarray:
        """View immutable bytes directly so restored owners share numerical storage."""
        shape, dtype, offset = self._header
        return np.ndarray(
            shape,
            dtype=dtype,
            buffer=self.npy,
            offset=offset,
            order="C",
        )

    @property
    def layout(self) -> tuple[tuple[int, ...], np.dtype]:
        """Read dimensions and dtype without materializing numerical elements."""
        shape, dtype, _ = self._header
        return shape, dtype

    @cached_property
    def _header(self) -> tuple[tuple[int, ...], np.dtype, int]:
        """Parse the storage layout once at the numerical value boundary."""
        stream = io.BytesIO(self.npy)
        version = np.lib.format.read_magic(stream)
        if version not in ((1, 0), (2, 0)):
            raise ValueError("Numerical arrays require NPY version 1 or 2")
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, fortran_order, dtype = reader(stream)
        if dtype.kind not in "bifu" or dtype.itemsize not in (1, 2, 4, 8):
            raise ValueError("NPY values must have a supported numerical dtype")
        if fortran_order or dtype.str[0] not in "<|":
            raise ValueError("Numerical arrays require little-endian, row-major storage")
        if len(self.npy) - stream.tell() != math.prod(shape) * dtype.itemsize:
            raise ValueError("NPY payload does not match its declared shape and dtype")
        return shape, dtype, stream.tell()

    @model_validator(mode="after")
    def parse_layout(self) -> Self:
        """Establish the binary array layout once, without reading numerical elements."""
        _ = self.layout
        return self


class ArrayVector(Value):
    """A vector view of an owned numerical value; the wire codec shares its buffer."""

    array: NumericalArray
    indices: tuple[Annotated[int, Field(ge=0)] | None, ...]
    mask: ArrayVector | None = None
    start: int = Field(default=0, ge=0)
    stop: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def one_vector_axis(self) -> Self:
        """Exactly one axis remains free in a stored vector selection."""
        if self.indices.count(None) != 1:
            raise ValueError("A vector selection must have exactly one free axis")
        if self.stop is not None and self.stop < self.start:
            raise ValueError("A vector range must be ordered")
        shape, _ = self.array.layout
        if len(self.indices) != len(shape) or any(
            index is not None and index >= size
            for index, size in zip(self.indices, shape, strict=True)
        ):
            raise ValueError("Vector coordinates exceed the owned buffer")
        extent = shape[self.indices.index(None)]
        if self.start > extent or (self.stop is not None and self.stop > extent):
            raise ValueError("Vector range exceeds the owned buffer")
        if self.mask is not None and self.mask.length != self.length:
            raise ValueError("Vector mask must align with its values")
        return self

    @property
    def length(self) -> int:
        """The selected vector's extent, proven at construction."""
        shape, _ = self.array.layout
        return (
            self.stop if self.stop is not None else shape[self.indices.index(None)]
        ) - self.start


type ScalarValues = tuple[FiniteFloat | None, ...] | ArrayVector


def resolve_vector(value: ScalarValues) -> tuple[float | None, ...]:
    """Select stored scalar values, applying the recorded observation mask."""
    if not isinstance(value, ArrayVector):
        return value
    indices = tuple(slice(None) if index is None else index for index in value.indices)
    values = value.array.values[indices][value.start : value.stop]
    if value.mask is not None:
        values = np.where(np.asarray(resolve_vector(value.mask), dtype=bool), values, np.nan)
    return tuple(float(item) if np.isfinite(item) else None for item in values)
