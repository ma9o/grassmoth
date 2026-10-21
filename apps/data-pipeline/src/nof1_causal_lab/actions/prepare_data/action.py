"""Complete a preparation from parsed observations and extraction evidence."""

from nof1_causal_lab.actions.contracts.bodies import PrepareDataOutput
from nof1_causal_lab.artifacts.observations.dataset import ObservationDataset
from nof1_causal_lab.artifacts.observations.preparation import (
    DataPreparationResult,
    PreparedDataMetadata,
)
from nof1_causal_lab.observations.quality.profile import profile_data
from nof1_causal_lab.study.projections.visuals import observation_history


def prepare_data(
    observations: ObservationDataset,
    metadata: PreparedDataMetadata,
    extraction: DataPreparationResult,
) -> PrepareDataOutput:
    """Retain complete histories and quality findings for the parsed panel."""
    import polars as pl

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
