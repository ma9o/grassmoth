"""MessagePack persistence and transport for domain-owned numerical values.

NPY payloads are logical binary values. Extension 42 references an earlier value
by its zero-based traversal index, encoded as an unsigned 64-bit big-endian integer.
Large values are arrays containing an extension-43 total byte length followed by
bounded binary chunks. Their chunks count as one logical value for references.
Decoding restores shared bytes before scientific owners parse the result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from io import BytesIO
from typing import TYPE_CHECKING, cast

import msgpack
import orjson
from pydantic_core import to_jsonable_python

from grassmoth.public import NumericalArray

if TYPE_CHECKING:
    import numpy as np
    from pydantic import BaseModel

    from grassmoth.public import JsonObject

_BUFFER_REFERENCE = 42
_BUFFER_CHUNKS = 43
_BINARY_CHUNK_BYTES = 64 * 1024**2


def _metadata(value: object) -> object:
    if isinstance(value, bytes):
        return value
    if isinstance(value, Mapping):
        return {str(key): _metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_metadata(item) for item in value]
    return to_jsonable_python(value)


def write_payload(value: object, write_bytes: Callable[[bytes], object]) -> None:
    """Stream shared buffers with at most one bounded MessagePack chunk copy."""
    buffers: dict[bytes, int] = {}
    identities: dict[int, tuple[bytes, int]] = {}
    packer = msgpack.Packer(use_bin_type=True)

    def write(item: object) -> None:
        if isinstance(item, bytes):
            prior = identities.get(id(item))
            if prior is not None:
                write_bytes(
                    packer.pack(msgpack.ExtType(_BUFFER_REFERENCE, prior[1].to_bytes(8, "big")))
                )
                return
            index = buffers.get(item)
            first = index is None
            if index is None:
                index = len(buffers)
                buffers[item] = index
            # Keep equal-but-distinct owners alive and remember their aliases;
            # repeated views must not compare multi-GB buffers for every visit.
            identities[id(item)] = (item, index)
            if not first:
                write_bytes(
                    packer.pack(msgpack.ExtType(_BUFFER_REFERENCE, index.to_bytes(8, "big")))
                )
            elif len(item) <= _BINARY_CHUNK_BYTES:
                write_bytes(packer.pack(item))
            else:
                count = (len(item) + _BINARY_CHUNK_BYTES - 1) // _BINARY_CHUNK_BYTES
                write_bytes(packer.pack_array_header(count + 1))
                write_bytes(
                    packer.pack(msgpack.ExtType(_BUFFER_CHUNKS, len(item).to_bytes(8, "big")))
                )
                view = memoryview(item)
                for start in range(0, len(item), _BINARY_CHUNK_BYTES):
                    write_bytes(packer.pack(view[start : start + _BINARY_CHUNK_BYTES]))
        elif isinstance(item, Mapping):
            write_bytes(packer.pack_map_header(len(item)))
            for key, child in item.items():
                write_bytes(packer.pack(key))
                write(child)
        elif isinstance(item, (list, tuple)):
            write_bytes(packer.pack_array_header(len(item)))
            for child in item:
                write(child)
        else:
            write_bytes(packer.pack(item))

    write(value)


def pack_payload(value: object) -> bytes:
    """Encode a transport body with the same shared-buffer wire format as disk writes."""
    output = BytesIO()
    write_payload(value, output.write)
    return output.getvalue()


def result_payload(
    result: BaseModel,
    *,
    array_loader: Callable[[str], np.ndarray] | None = None,
    thin_fit_draws: bool = False,
) -> Mapping[str, object]:
    """Serialize each numerical field at its owner, without a result-wide buffer map."""
    values = result.model_dump(
        mode="python",
        context={
            "binary_arrays": True,
            "distribution_array_loader": array_loader,
            "thin_fit_draws": thin_fit_draws,
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
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], msgpack.ExtType)
            and value[0].code == _BUFFER_CHUNKS
        ):
            marker, *chunks = value
            if (
                len(marker.data) != 8
                or not chunks
                or any(not isinstance(chunk, bytes) or not chunk for chunk in chunks)
            ):
                raise ValueError("Invalid numerical buffer chunks")
            restored = b"".join(chunks)
            if len(restored) != int.from_bytes(marker.data, "big"):
                raise ValueError("Numerical buffer chunks do not match their byte length")
            buffers.append(restored)
            return restored
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


def json_result(payload: bytes) -> bytes:
    """Render the served envelope as strict JSON, expanding NPY values in place.

    Shared buffers are restored by the wire decoder. Non-finite numbers become
    null, including masked observations and unavailable numerical diagnostics.
    """

    def expand(value: object) -> object:
        if isinstance(value, Mapping):
            if set(value) == {"npy"}:
                array = NumericalArray(npy=cast("bytes", value["npy"])).values
                return array.item() if array.ndim == 0 else array
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return orjson.dumps(expand(unpack_result(payload)), option=orjson.OPT_SERIALIZE_NUMPY)


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
