"""Content-addressed numerical values for native distribution constructor trees."""

import numpy as np

from grassmoth.public import NumericalArray


def decode_array(identity: str, payload: bytes) -> np.ndarray:
    """Verify content identity before loading a numerical value."""
    value = NumericalArray(npy=payload)
    if value.identity != identity:
        raise ValueError("Stored numerical array failed its content identity check")
    return value.values
