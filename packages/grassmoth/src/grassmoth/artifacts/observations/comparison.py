"""Immutable source provenance paired with its admitted observation history."""

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.observations.dataset import ObservationDataset
from grassmoth.artifacts.observations.reference import DataRef


class Dataset(Value):
    """One admitted history, identified by its immutable source."""

    source: DataRef[GitOid, int]
    observations: ObservationDataset
