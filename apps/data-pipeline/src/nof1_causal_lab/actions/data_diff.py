"""Compare supplied observation histories without generation or persistence."""

from nof1_causal_lab.actions.contracts.bodies import DataDiffOutput
from nof1_causal_lab.artifacts.identity import IndicatorId
from nof1_causal_lab.artifacts.observations.comparison import Dataset
from nof1_causal_lab.artifacts.reports.data_comparison import DataComparisonReport
from nof1_causal_lab.models.diagnostics.history_comparison import compare_data_variables


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
