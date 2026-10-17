"""Producing coordinate charts for native energies, masks and continuation state."""

import hashlib
import json
from typing import Annotated, Literal

from pydantic import Field, computed_field

from nof1_causal_lab.artifacts.base import Value


class CentredChartRecord(Value):
    """The original frozen affine chart, reconstructed from the pinned target and seed."""

    kind: Literal["centred"] = "centred"


class InnovationChartRecord(Value):
    """Exact conditional innovations in every unpinned independent path coordinate."""

    kind: Literal["innovation"] = "innovation"


class MixedChartRecord(Value):
    """Centred free coordinates, in the full checkpoint path's time/state axis order."""

    kind: Literal["mixed"] = "mixed"
    centred_mask: tuple[tuple[bool, ...], ...]

    @computed_field
    @property
    def mask_identity(self) -> str:
        """Content identity of the actual mask, including its time and state dimensions."""
        return hashlib.sha256(json.dumps(self.centred_mask, separators=(",", ":")).encode()).hexdigest()

    @computed_field
    @property
    def equals_innovation(self) -> bool:
        """Whether the available-reading policy selects no centred free coordinate."""
        return not any(any(row) for row in self.centred_mask)


type JointChartRecord = Annotated[
    CentredChartRecord | InnovationChartRecord | MixedChartRecord, Field(discriminator="kind")
]
