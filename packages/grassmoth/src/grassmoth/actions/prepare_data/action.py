"""Extract a source table's observations and complete the preparation with its findings."""

import polars as pl

from grassmoth.actions.contracts.io import ExtractedValue, PrepareDataOutput, SourceTable
from grassmoth.actions.prepare_data.aggregations import compute_indicators
from grassmoth.actions.prepare_data.materialization import (
    materialize_panel,
    recorded_observations,
)
from grassmoth.actions.prepare_data.rows import (
    annotate_observation_rows,
    extracted_rows,
    prepared_time_origin,
)
from grassmoth.artifacts.observations.dataset import ObservationDataset
from grassmoth.artifacts.observations.preparation import (
    DataPreparationResult,
    DataPreparationSpec,
    PreparedDataMetadata,
)
from grassmoth.observations.quality.profile import profile_data
from grassmoth.study.projections.visuals import observation_history


def prepare_data(
    definition: DataPreparationSpec,
    source: SourceTable,
    extracted: tuple[ExtractedValue, ...],
    extraction: DataPreparationResult,
) -> PrepareDataOutput:
    """Compute and import the deterministic variables, add the extracted values, and profile them.

    ``extracted`` holds the semantic variables' values, one per support window, including
    explicit nulls for windows without source values.
    """
    windowed = pl.concat((compute_indicators(source, definition), extracted_rows(extracted)))
    rows = pl.concat(
        (
            annotate_observation_rows(windowed, definition.observation_schema()),
            recorded_observations(source, definition),
        )
    )
    observed = materialize_panel(rows, definition)
    if observed.is_empty():
        raise ValueError("Extraction produced no observations")
    metadata = PreparedDataMetadata(
        preparation=definition, time_origin=prepared_time_origin(observed, source.start)
    )
    return prepared_output(
        ObservationDataset.from_frame(
            observed, metadata.variables, time_origin=metadata.time_origin
        ),
        metadata,
        extraction,
    )


def prepared_output(
    observations: ObservationDataset,
    metadata: PreparedDataMetadata,
    extraction: DataPreparationResult,
) -> PrepareDataOutput:
    """Retain complete histories and quality findings for the parsed panel."""
    panel = observations.recorded.frame
    return PrepareDataOutput(
        data={
            variable.id: observation_history(
                metadata.time_origin,
                panel.filter(pl.col("indicator_id") == variable.id).sort("anchor_time"),
            )
            for variable in metadata.variables
        },
        metadata=metadata,
        profile=profile_data(panel, definitions=metadata.variables, metadata=metadata),
        extraction=extraction,
    )
