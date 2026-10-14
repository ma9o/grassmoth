"""Read-only comparison of saved observation datasets and simulation replicates."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from nof1_causal_lab.actions.contracts.bodies import DataDiffOutput
from nof1_causal_lab.actions.contracts.requests import SimulateRequest
from nof1_causal_lab.artifacts.reports.data_comparison import DataComparisonReport
from nof1_causal_lab.artifacts.simulation import SimulationEvidence
from nof1_causal_lab.models.diagnostics.history_comparison import compare_data_variables

if TYPE_CHECKING:
    from nof1_causal_lab.actions.contracts.requests import DataDiffRequest
    from nof1_causal_lab.artifacts.identity import GitOid, IndicatorId
    from nof1_causal_lab.artifacts.observations.comparison import Dataset
    from nof1_causal_lab.artifacts.observations.reference import DataRef, DataSelection


def _compute_data_diff(workspace_id: str, request: DataDiffRequest[GitOid]) -> DataDiffOutput:
    """Load existing data only; never read a model for generation, fit, or write artifacts."""
    from nof1_causal_lab.study.persistence.history import StudyRepository
    from nof1_causal_lab.study.persistence.store import ArtifactStore, read_model
    from nof1_causal_lab.study.projections.observations import read_dataset
    from nof1_causal_lab.study.queries.data import read_data_histories, read_data_source

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
            read_dataset(history.source, history.observations)
            for history in read_data_histories(store, source)
        )

    def selection(value: DataSelection[GitOid]) -> tuple[Dataset, ...]:
        return tuple(dataset for source in value for dataset in load(source))  # pyright: ignore[reportArgumentType] -- Value freezes DataRef; Pydantic's stub declares __hash__=None before its frozen metaclass installs the hash.

    left, right = selection(request.input.left_ref), selection(request.input.right_ref)
    return DataDiffOutput(
        report=DataComparisonReport(
            left=tuple(item.source for item in left),
            right=tuple(item.source for item in right),
            variables=compare_data_variables(left, right, input_indicators=input_indicators),
        ),
    )


def read_data_diff(workspace_id: str, request: DataDiffRequest[GitOid]) -> DataDiffOutput:
    """Compute the comparison once during its owning action."""
    return _compute_data_diff(workspace_id, request)
