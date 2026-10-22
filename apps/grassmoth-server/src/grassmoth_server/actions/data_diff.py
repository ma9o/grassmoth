"""Read-only comparison of saved observation datasets and simulation replicates."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from grassmoth.public import SimulationEvidence, data_diff
from grassmoth_server.actions.contracts.requests import SimulateRequest

if TYPE_CHECKING:
    from grassmoth.public import DataDiffOutput, DataRef, Dataset, GitOid, IndicatorId
    from grassmoth_server.actions.contracts.inputs import DataSelection
    from grassmoth_server.actions.contracts.requests import DataDiffRequest


def read_data_diff(workspace_id: str, request: DataDiffRequest[GitOid]) -> DataDiffOutput:
    """Load existing data only; never read a model for generation, fit, or write artifacts."""
    from grassmoth.public import Dataset
    from grassmoth_server.study.persistence.history import StudyRepository
    from grassmoth_server.study.persistence.store import ArtifactStore, read_model
    from grassmoth_server.study.queries.data import read_data_histories, read_data_source

    store = ArtifactStore(workspace_id)
    input_indicators: set[IndicatorId] = set()

    @cache
    def load(source: DataRef[GitOid, int | None]) -> tuple[Dataset, ...]:
        producer = read_data_source(store, source.revision)
        if isinstance(producer, SimulationEvidence):
            recorded = StudyRepository(workspace_id).record(source.revision).record.attempt.request
            assert isinstance(recorded, SimulateRequest)
            dynamical_model_spec = read_model(store, recorded.input.dynamical_model_spec_ref)
            input_indicators.update(
                indicator.observation.id
                for construct in dynamical_model_spec.constructs
                if construct.role == "exogenous"
                for indicator in construct.indicators
            )
        return tuple(
            Dataset.from_observations(history.source, history.observations)
            for history in read_data_histories(store, source)
        )

    def selection(value: DataSelection[GitOid]) -> tuple[Dataset, ...]:
        return tuple(dataset for source in value for dataset in load(source))  # pyright: ignore[reportArgumentType] -- Value freezes DataRef; Pydantic's stub declares __hash__=None before its frozen metaclass installs the hash.

    left, right = selection(request.input.left_ref), selection(request.input.right_ref)
    return data_diff(left, right, input_indicators=frozenset(input_indicators))
