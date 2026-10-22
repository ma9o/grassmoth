"""Read prepared or simulated observation histories directly from their saved producer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from grassmoth.public import (
    DataRef,
    GitOid,
    ModelTime,
    ObservationDataset,
    ObservationInstant,
    PreparedDataMetadata,
)
from grassmoth_server.study.errors import StudyLookupError

if TYPE_CHECKING:
    from datetime import datetime

    from grassmoth.public import ResolvedObservationSpec, SimulationEvidence
    from grassmoth_server.actions.prepare_data.records import ObservationRecord
    from grassmoth_server.study.persistence.store import ArtifactStore


def read_simulation_observations(
    report: SimulationEvidence,
    replicate: int,
) -> pl.DataFrame:
    """Read emitted observations on their recorded dates."""
    if not 0 <= replicate < report.draws:
        raise StudyLookupError(f"Simulation replicate must be between 0 and {report.draws - 1}")
    times = np.asarray(report.times)
    layout = report.observation_layout
    starts = layout.support_start_times.values
    ends = layout.support_end_times.values
    mask = layout.mask.values
    if (
        starts.shape != (len(times), len(report.observation_layout.indicator_ids))
        or ends.shape != starts.shape
    ):
        raise ValueError("Simulation support does not match the recorded observation layout")
    observations = report.arms.action.observations.values
    if observations.shape != (
        report.draws,
        len(times),
        len(report.observation_layout.indicator_ids),
    ):
        raise ValueError("Simulation observations do not match the recorded draws and design")
    values = observations[replicate]
    if mask.shape != observations.shape or mask.dtype != np.bool_:
        raise ValueError("Simulation observation mask must align with the recorded draws")
    observed = mask[replicate]
    if not np.isfinite(values[observed]).all():
        raise ValueError("Simulation replicate has non-finite emissions at observed times")
    if np.isinf(values).any():
        raise ValueError("Simulation replicate contains infinite values")
    if not (np.isfinite(starts[observed]).all() and np.isfinite(ends[observed]).all()):
        raise ValueError("Observed simulation values must have finite support boundaries")

    origin = ObservationInstant(report.time_origin)

    def _timestamp(day: float) -> str | None:
        if np.isnan(day):
            return None
        return ModelTime(float(day)).at(origin).value.isoformat(timespec="microseconds")

    rows: list[ObservationRecord] = [
        {
            "indicator_id": identity,
            "value": None if not observed[t, i] else float(values[t, i]),
            "anchor_time": _timestamp(time),
            "support_kind": layout.variables[i].support_kind.value,
            "summary_operator": layout.variables[i].summary_operator.value,
            "anchor_policy": layout.variables[i].anchor_policy.value,
            "observation_window": str(layout.variables[i].observation_window),
            "support_start": _timestamp(starts[t, i]),
            "support_end": _timestamp(ends[t, i]),
        }
        for t, time in enumerate(times)
        for i, identity in enumerate(report.observation_layout.indicator_ids)
    ]
    # Emissions already use the model's numeric codes, including unobserved
    # category levels. Extraction's label encoding must not run a second time.
    return (
        pl.DataFrame(rows, schema=ObservationDataset.row_schema() | {"value": pl.Float64})
        .with_columns(
            pl.col("anchor_time", "support_start", "support_end")
            .str.to_datetime(format="%+")
            .dt.replace_time_zone(None)
        )
        .sort("indicator_id", "anchor_time")
    )


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


def read_data_source(
    store: ArtifactStore, revision: GitOid
) -> PreparedDataMetadata | SimulationEvidence:
    """Resolve a prepared panel tree or an applied data-producing action commit."""
    from grassmoth.public import SimulateOutput
    from grassmoth_server.study.persistence.git_objects import object_tree
    from grassmoth_server.study.persistence.history import StudyRepository
    from grassmoth_server.study.queries.lineage import read_data_metadata
    from grassmoth_server.study.records import Applied

    tree = object_tree(store.repo, revision)
    if "meta.json" in tree:
        store.read_meta("panel", revision)
        return read_data_metadata(store, revision)
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
        return read_data_metadata(store, panel.revision)
    if attempt.action == "simulate":
        result = store.read_result(attempt.outcome.result, SimulateOutput)
        return result.report.evidence
    raise StudyLookupError("Data must select an applied prepare_data or simulate call")


def panel_revision(store: ArtifactStore, revision: GitOid) -> GitOid:
    """Locate the parquet tree of an already selected prepared-data source."""
    from grassmoth_server.study.persistence.git_objects import object_tree
    from grassmoth_server.study.persistence.history import StudyRepository
    from grassmoth_server.study.records import Applied

    if "meta.json" in object_tree(store.repo, revision):
        return revision
    attempt = (
        StudyRepository(store.workspace_id, repository_path=Path(store.repo.path))
        .record(revision)
        .record.attempt
    )
    assert attempt.action == "prepare_data"
    assert isinstance(attempt.outcome, Applied)
    return next(
        info.revision for info in attempt.outcome.effects.produced if info.artifact_id == "panel"
    )


def read_data_history(store: ArtifactStore, source: DataRef[GitOid, int]) -> DataHistory:
    """Select exactly one recorded history; prepared user data has only index zero."""
    data = read_data_source(store, source.revision)
    return _read_history(store, source, data)


def _read_history(
    store: ArtifactStore,
    source: DataRef[GitOid, int],
    data: PreparedDataMetadata | SimulationEvidence,
) -> DataHistory:
    if isinstance(data, PreparedDataMetadata):
        if source.replicate_index != 0:
            raise StudyLookupError("Prepared user data has one history; replicate_index must be 0")
        return DataHistory(
            source,
            ObservationDataset.from_frame(
                store.read_parquet_file(
                    "panel", panel_revision(store, source.revision), "panel.parquet"
                ),
                data.variables,
                time_origin=data.time_origin,
            ),
            data,
        )
    return DataHistory(
        source,
        ObservationDataset.from_frame(
            read_simulation_observations(data, source.replicate_index),
            data.observation_layout.variables,
            time_origin=data.time_origin,
        ),
        None,
    )


def read_data_histories(
    store: ArtifactStore, source: DataRef[GitOid, int | None]
) -> tuple[DataHistory, ...]:
    """Read the selected histories with each simulation array loaded once."""
    data = read_data_source(store, source.revision)
    indices = (
        (source.replicate_index,)
        if source.replicate_index is not None
        else (0,)
        if isinstance(data, PreparedDataMetadata)
        else range(data.draws)
    )
    return tuple(
        _read_history(
            store,
            DataRef[GitOid, int](revision=source.revision, replicate_index=index),
            data,
        )
        for index in indices
    )
