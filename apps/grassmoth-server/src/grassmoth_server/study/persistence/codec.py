"""MessagePack persistence and transport for domain-owned numerical values.

NPY payloads are ordinary binary values. Extension 42 references an earlier binary
value by its zero-based traversal index, encoded as an unsigned 64-bit big-endian
integer. Decoding restores shared bytes before scientific owners parse the result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

import msgpack
from pydantic_core import to_jsonable_python

if TYPE_CHECKING:
    import numpy as np
    from pydantic import BaseModel

    from grassmoth.public import JsonObject

_BUFFER_REFERENCE = 42


def _metadata(value: object) -> object:
    if isinstance(value, bytes):
        return value
    if isinstance(value, Mapping):
        return {str(key): _metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_metadata(item) for item in value]
    return to_jsonable_python(value)


def pack_payload(value: object) -> bytes:
    """Encode a payload and deduplicate array buffers in the binary envelope."""
    buffers: dict[bytes, int] = {}

    def share(item: object) -> object:
        if isinstance(item, bytes):
            if item in buffers:
                return msgpack.ExtType(_BUFFER_REFERENCE, buffers[item].to_bytes(8, "big"))
            buffers[item] = len(buffers)
            return item
        if isinstance(item, Mapping):
            return {key: share(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [share(child) for child in item]
        return item

    return cast("bytes", msgpack.packb(share(value), use_bin_type=True))


def result_payload(
    result: BaseModel, *, array_loader: Callable[[str], np.ndarray] | None = None
) -> Mapping[str, object]:
    """Serialize each numerical field at its owner, without a result-wide buffer map."""
    values = result.model_dump(
        mode="python",
        context={
            "binary_arrays": True,
            "distribution_array_loader": array_loader,
        },
    )
    return {key: _metadata(value) for key, value in values.items()}


def pack_result(
    result: BaseModel, *, array_loader: Callable[[str], np.ndarray] | None = None
) -> bytes:
    """Retain the complete action body, sharing equal buffers across domain views."""
    return pack_payload(result_payload(result, array_loader=array_loader))


def unpack_result(payload: bytes) -> Mapping[str, object]:
    """Restore shared binary values before parsing the operation-owned schema."""
    buffers: list[bytes] = []

    def restore(value: object) -> object:
        if isinstance(value, bytes):
            buffers.append(value)
            return value
        if isinstance(value, msgpack.ExtType):
            if value.code != _BUFFER_REFERENCE or len(value.data) != 8:
                raise ValueError("Unknown numerical MessagePack extension")
            index = int.from_bytes(value.data, "big")
            if index >= len(buffers):
                raise ValueError("Numerical buffer reference must select an earlier value")
            return buffers[index]
        if isinstance(value, dict):
            return {key: restore(item) for key, item in value.items()}
        if isinstance(value, list):
            return [restore(item) for item in value]
        return value

    return cast("Mapping[str, object]", restore(msgpack.unpackb(payload, raw=False)))


def pack_envelope(metadata: JsonObject, body: bytes) -> bytes:
    """Append the saved body unchanged; envelope metadata contains no binary values."""
    packer = msgpack.Packer(use_bin_type=True)
    return cast(
        "bytes",
        packer.pack_map_header(len(metadata) + 1)
        + b"".join(packer.pack(key) + packer.pack(value) for key, value in metadata.items())
        + packer.pack("body")
        + body,
    )
