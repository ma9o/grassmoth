"""Content-addressed numerical values for native distribution constructor trees."""

import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from nof1_causal_lab.artifacts.arrays import NumericalArray
from nof1_causal_lab.storage import backend as storage


def encode_array(values: np.ndarray) -> tuple[str, bytes]:
    """Own the portable numerical NPY encoding used by storage and action transport."""
    value = NumericalArray.from_numpy(values)
    return value.identity, value.npy


def decode_array(identity: str, payload: bytes) -> np.ndarray:
    """Verify content identity before loading a numerical value."""
    value = NumericalArray(npy=payload)
    if value.identity != identity:
        raise ValueError("Stored numerical array failed its content identity check")
    return value.values


def write_array(directory: str, values: np.ndarray) -> str:
    """Persist a content-addressed NumPy array only when its encoded content is not already stored."""
    return write_owned_array(directory, NumericalArray.from_numpy(values))


def write_owned_array(directory: str, value: NumericalArray) -> str:
    """Persist an already encoded numerical owner without another copy or hash."""
    identity = value.identity
    storage.makedirs(directory)
    path = storage.join(directory, f"{identity}.npy")
    if storage.is_remote():
        with storage.open_file(path, "wb") as stream:
            stream.write(value.npy)
    else:
        with NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(value.npy)
                stream.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return identity


def read_array(directory: str, identity: str) -> np.ndarray:
    """Load a numerical array by a validated SHA-256 identity and verify its encoded content."""
    return decode_array(identity, read_array_bytes(directory, identity))


def read_array_bytes(directory: str, identity: str) -> bytes:
    """Read a temporary NPY buffer without expanding its numerical values."""
    if re.fullmatch(r"[0-9a-f]{64}", identity) is None:
        raise ValueError("Invalid numerical array identity")
    with storage.open_file(storage.join(directory, f"{identity}.npy"), "rb") as stream:
        return stream.read()
