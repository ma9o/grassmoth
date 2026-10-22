"""Resolve user-data extraction instructions against one pinned scientific model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.public import DataPreparationSpec, DataVariableSpec
from grassmoth_server.actions.contracts.preparation import FilePreparationSpec, FileSourceRef
from grassmoth_server.study.errors import StudyLookupError
from grassmoth_server.study.persistence.store import ArtifactStore, read_model

if TYPE_CHECKING:
    from grassmoth.public import GitOid
    from grassmoth_server.actions.contracts.inputs import PrepareDataInput


def resolve_preparation(
    store: ArtifactStore, request: PrepareDataInput[GitOid, FileSourceRef]
) -> FilePreparationSpec:
    """Resolve extraction instructions against the selected model's observation schema.

    Args:
        store: Artifact store containing the referenced model revision.
        request: Pinned source, model, and per-indicator extraction instructions.

    Returns:
        Preparation recipe with observation definitions and clock supplied by the model.

    Raises:
        StudyLookupError: The model has no measurement clock, no observations, or
            extraction keys that do not exactly match its observation IDs.
    """
    dynamical_model_spec = read_model(store, request.dynamical_model_spec_ref)
    if dynamical_model_spec.measurement_clock is None:
        raise StudyLookupError("prepare_data requires a model with a measurement clock")
    identities = {indicator.observation.id for indicator in dynamical_model_spec.indicators}
    if not identities or set(request.extraction) != identities:
        raise StudyLookupError(
            "Extraction instructions must name each model observation ID exactly once"
        )
    return FilePreparationSpec(
        source=request.source,
        definition=DataPreparationSpec(
            default_window=dynamical_model_spec.measurement_clock,
            variables=tuple(
                DataVariableSpec(
                    observation=indicator.observation,
                    extraction=request.extraction[indicator.observation.id],
                )
                for indicator in dynamical_model_spec.indicators
            ),
            context=request.context,
        ),
    )
