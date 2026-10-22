"""Compare supplied observation histories without generation or persistence."""

from grassmoth.actions.contracts.bodies import DataDiffOutput
from grassmoth.artifacts.identity import IndicatorId
from grassmoth.artifacts.observations.comparison import Dataset
from grassmoth.artifacts.reports.data_comparison import DataComparisonReport
from grassmoth.models.diagnostics.history_comparison import compare_data_variables


def data_diff(
    left: tuple[Dataset, ...],
    right: tuple[Dataset, ...],
    *,
    input_indicators: frozenset[IndicatorId],
) -> DataDiffOutput:
    """Retain the exact selections and their empirical comparison findings."""
    return DataDiffOutput(
        report=DataComparisonReport(
            left=tuple(item.source for item in left),
            right=tuple(item.source for item in right),
            variables=compare_data_variables(left, right, input_indicators=input_indicators),
        )
    )
