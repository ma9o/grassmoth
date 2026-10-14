"""Compute data-only findings while preparing observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.observations.reference import DataRef
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.study.persistence.store import ArtifactStore
from nof1_causal_lab.study.queries.data import read_data_history
from nof1_causal_lab.study.state import apply_effects

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.observations.preparation import DataPreparationResult
    from nof1_causal_lab.artifacts.reports.data_quality import DataProfileReport
    from nof1_causal_lab.study.records import Applied
    from nof1_causal_lab.study.state import StudyState


def read_data_profile(store: ArtifactStore, source: DataRef[GitOid, int]) -> DataProfileReport:
    """Compute the empirical data profile for one exact replicate selection."""
    from nof1_causal_lab.observations.quality.profile import profile_data

    def render() -> DataProfileReport:
        history = read_data_history(store, source)
        return profile_data(
            history.observations.recorded.frame,
            definitions=history.variables,
            metadata=history.metadata,
        )

    return render()


def evaluate_data_checks(
    workspace_id: str,
    state: StudyState,
    applied: Applied[DataPreparationResult],
) -> DataProfileReport:
    """Return the prepared panel's report for publication with its action."""
    selected = apply_effects(state, applied.effects.produced, applied.effects.retracted)
    panel = selected.get("panel")
    if panel is None:
        raise IncompleteModelError("Data preparation produced no usable observations")
    return read_data_profile(
        ArtifactStore(workspace_id),
        DataRef[GitOid, int](revision=panel.revision, replicate_index=0),
    )
