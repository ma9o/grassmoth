"""Producing coordinate parameterizations for native energies, masks and continuation state."""

import hashlib
import json
from typing import Annotated, Literal

from pydantic import Field, computed_field

from grassmoth.artifacts.base import Value


class CentredParameterizationRecord(Value):
    """The original frozen affine parameterization, reconstructed from the pinned target and seed."""

    kind: Literal["centred"] = "centred"


class NonCentredParameterizationRecord(Value):
    """Exact conditional innovations in every unpinned independent path coordinate."""

    kind: Literal["non_centred"] = "non_centred"


class MixedParameterizationRecord(Value):
    """Centred free coordinates, in the full checkpoint path's time/state axis order."""

    kind: Literal["mixed"] = "mixed"
    centred_mask: tuple[tuple[bool, ...], ...]

    @computed_field
    @property
    def mask_identity(self) -> str:
        """Content identity of the actual mask, including its time and state dimensions."""
        return hashlib.sha256(
            json.dumps(self.centred_mask, separators=(",", ":")).encode()
        ).hexdigest()

    @computed_field
    @property
    def equals_non_centred(self) -> bool:
        """Whether the available-reading policy selects no centred free coordinate."""
        return not any(any(row) for row in self.centred_mask)


type JointParameterizationRecord = Annotated[
    CentredParameterizationRecord | NonCentredParameterizationRecord | MixedParameterizationRecord,
    Field(discriminator="kind"),
]
