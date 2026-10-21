"""Construct fitting inputs from compiled laws and a bound observation panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nof1_causal_lab.models.ssm.compile.inputs import CompiledFitInputs
from nof1_causal_lab.models.ssm.preparation.binding import (
    BoundPanel,
    PanelPreparationFailure,
    bind_panel,
)

if TYPE_CHECKING:
    from datetime import datetime

    from nof1_causal_lab.artifacts.observations.dataset import ObservationDataset
    from nof1_causal_lab.models.ssm.compile.inputs import CompilationFailure, FitCompilationResult


@dataclass(frozen=True)
class PreparedFit:
    """Fit laws and observations bound to the same compiled model."""

    inputs: CompiledFitInputs
    panel: BoundPanel


def prepare_fit(
    inputs: FitCompilationResult,
    observations: ObservationDataset,
    *,
    time_origin: datetime,
) -> PreparedFit | CompilationFailure | PanelPreparationFailure:
    """Forward compilation failures or bind the resolved fitting inputs once."""
    if not isinstance(inputs, CompiledFitInputs):
        return inputs
    panel = bind_panel(
        observations,
        compiled_dynamical_model=inputs.compiled_dynamical_model,
        time_origin=time_origin,
    )
    if isinstance(panel, PanelPreparationFailure):
        return panel
    return PreparedFit(inputs, panel)
