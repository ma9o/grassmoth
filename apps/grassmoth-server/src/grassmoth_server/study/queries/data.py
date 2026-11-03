"""Read prepared or simulated observation histories directly from their saved producer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from grassmoth.public import (
    DataRef,
    GitOid,
    ObservationDataset,
    PrepareDataOutput,
    PreparedDataMetadata,
    SimulateOutput,
)
from grassmoth_server.study.errors import StudyLookupError

if TYPE_CHECKING:
    from datetime import datetime

    from grassmoth.public import ResolvedObservationSpec
    from grassmoth_server.study.persistence.store import ArtifactStore


@dataclass(frozen=True)
class DataHistory:
    """One selected history hydrated at the storage boundary, without a new artifact."""

    source: DataRef[GitOid, int]
    observations: ObservationDataset
    metadata: PreparedDataMetadata | None

    @property
    def variables(self) -> tuple[ResolvedObservationSpec, ...]:
        """Observation definitions belonging to this selected history."""
        return self.observations.variables

    @property
    def time_origin(self) -> datetime:
        """Calendar instant of model day zero."""
        return self.observations.time_origin


@dataclass(frozen=True)
class PreparedDataSourceRef:
    """The artifact tree retaining a prepared observation output."""

    revision: GitOid


@dataclass(frozen=True)
class SimulatedDataSourceRef:
    """The saved result retaining a simulated observation output."""

    revision: GitOid


def resolve_data_source(
    store: ArtifactStore, revision: GitOid
) -> PreparedDataSourceRef | SimulatedDataSourceRef:
    """Discriminate a data source from its tree or applied action, without decoding its output."""
    from grassmoth_server.study.persistence.git_objects import object_tree
    from grassmoth_server.study.persistence.history import StudyRepository
    from grassmoth_server.study.records import Applied

    tree = object_tree(store.repo, revision)
    if "meta.json" in tree:
        store.read_meta("panel", revision)
        return PreparedDataSourceRef(revision)
    record = StudyRepository(store.workspace_id, repository_path=Path(store.repo.path)).record(
        revision
    )
    attempt = record.record.attempt
    if not isinstance(attempt.outcome, Applied):
        raise StudyLookupError("Data must select an applied prepare_data or simulate call")
    if attempt.action == "prepare_data":
        panel = next(
            (info for info in attempt.outcome.effects.produced if info.artifact_id == "panel"),
            None,
        )
        if panel is None:
            raise StudyLookupError("The selected preparation produced no observation history")
        return PreparedDataSourceRef(panel.revision)
    if attempt.action == "simulate":
        return SimulatedDataSourceRef(attempt.outcome.result)
    raise StudyLookupError("Data must select an applied prepare_data or simulate call")


def read_data_source(store: ArtifactStore, revision: GitOid) -> PrepareDataOutput | SimulateOutput:
    """Decode the output selected by the shared data-source resolver."""
    source = resolve_data_source(store, revision)
    if isinstance(source, PreparedDataSourceRef):
        return _read_prepared_output(store, source.revision)
    return store.read_result(source.revision, SimulateOutput)


def _read_prepared_output(store: ArtifactStore, revision: GitOid) -> PrepareDataOutput:
    from grassmoth_server.study.persistence.codec import unpack_result
    from grassmoth_server.study.persistence.git_objects import read_file

    return PrepareDataOutput.model_validate(
        unpack_result(read_file(store.repo, revision, "result.msgpack"))
    )


def read_data_history(store: ArtifactStore, source: DataRef[GitOid, int]) -> DataHistory:
    """Select exactly one recorded history; prepared user data has only index zero."""
    data = read_data_source(store, source.revision)
    return _read_history(source, data)


def _read_history(
    source: DataRef[GitOid, int], data: PrepareDataOutput | SimulateOutput
) -> DataHistory:
    if isinstance(data, PrepareDataOutput):
        if source.replicate_index != 0:
            raise StudyLookupError("Prepared user data has one history; replicate_index must be 0")
        return DataHistory(source, data.observations(), data.metadata)
    draws = data.report.evidence.draws
    if not 0 <= source.replicate_index < draws:
        raise StudyLookupError(f"Simulation replicate must be between 0 and {draws - 1}")
    return DataHistory(source, data.observations(source.replicate_index), None)


def select_data_histories(
    source: DataRef[GitOid, int | None], data: PrepareDataOutput | SimulateOutput
) -> tuple[DataHistory, ...]:
    """Select histories from an already loaded producer body."""
    indices = (
        (source.replicate_index,)
        if source.replicate_index is not None
        else (0,)
        if isinstance(data, PrepareDataOutput)
        else range(data.report.evidence.draws)
    )
    return tuple(
        _read_history(
            DataRef[GitOid, int](revision=source.revision, replicate_index=index),
            data,
        )
        for index in indices
    )
