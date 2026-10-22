"""A saved data producer and an optional selection within its observation histories."""

from __future__ import annotations

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value


class DataRef[RevisionT, IndexT: int | None](Value):
    """A prepared-data or simulation gitref with a selection of recorded histories.

    An integer selects one history by its zero-based index: zero for prepared user
    data, or less than the saved replicate count for a simulation. None selects
    every retained history when the consuming action permits it.
    """

    revision: RevisionT
    replicate_index: IndexT = Field(ge=0)
