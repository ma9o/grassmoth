"""Study state: artifact versions selected by successful attempts in submission order.

Each artifact records its pinned input OIDs, which identify the facts used by
execution. StudyState projects the indexed attempts, independently of their
publication order and Git ancestry.

These are pydantic models (frozen) rather than dataclasses because they
cross serialization boundaries verbatim: Temporal update/activity payloads
(via the pydantic data converter) and the tool-server facade's JSON API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import Field

from grassmoth.public import DataRef, GitOid, Value
from grassmoth_server.actions.contracts.identity import (
    ARTIFACT_IDS,
    ArtifactId,
    ScientificActionId,
)


class ArtifactFiles(Value):
    """Execution staging or an uploaded input owns its physical files."""

    kind: Literal["files"] = "files"


class ArtifactResult(Value):
    """A published scientific artifact selects a field of its owning action result."""

    kind: Literal["result"] = "result"
    result: GitOid


type ArtifactSource = Annotated[ArtifactFiles | ArtifactResult, Field(discriminator="kind")]


class ArtifactRecord(Value):
    """An immutable artifact revision with its producer and exact input dependencies.

    ``derived_from`` pins the versions used to compute the payload. Initial
    models pin their question; edits also pin their model parent. The producing
    activity stamps ``created_at`` outside deterministic workflow execution.
    """

    artifact_id: ArtifactId
    revision: GitOid
    derived_from: Mapping[ArtifactId, GitOid] = Field(default_factory=dict)
    produced_by: str | None = None
    created_at: str = ""
    source: ArtifactSource = Field(default_factory=ArtifactFiles)


class StudyState(Value):
    """Study state projects the artifacts selected by a sequence of attempts.

    ``current`` maps artifact id → the revision info that is *current* for the
    study. Absent key = the artifact does not exist (either never produced,
    or produced-when-nonempty semantics withheld it).
    """

    current: Mapping[ArtifactId, ArtifactRecord] = Field(default_factory=dict)
    data: DataRef[GitOid, int] | None = None

    def get(self, artifact_id: ArtifactId) -> ArtifactRecord | None:
        """Look up the selected artifact revision, or ``None`` if that artifact is absent."""
        return self.current.get(artifact_id)

    def has(self, artifact_id: ArtifactId) -> bool:
        """Check whether the state selects a revision of the requested artifact kind."""
        return artifact_id in self.current

    def matches_inputs(self, output: ArtifactId, *inputs: ArtifactId) -> bool:
        """Whether an artifact pins exactly the selected versions of these inputs."""
        info = self.get(output)
        return info is not None and all(
            (selected := self.get(artifact_id)) is not None
            and info.derived_from.get(artifact_id) == selected.revision
            for artifact_id in inputs
        )

    def with_artifacts(self, infos: Sequence[ArtifactRecord]) -> StudyState:
        """Return a new state with ``infos`` installed as current versions."""
        merged = dict(self.current)
        for info in infos:
            merged[info.artifact_id] = info
        return self.revised(current={aid: merged[aid] for aid in ARTIFACT_IDS if aid in merged})

    def without(self, artifact_ids: list[ArtifactId]) -> StudyState:
        """Return a new state with the given artifacts removed from ``current``.

        Used for produced-when-nonempty semantics: re-running a stage whose
        previous revision produced an optional artifact, but whose new run did
        not, must retract the old revision — otherwise downstream stages would
        silently consume a payload derived from superseded inputs.
        """
        removed = set(artifact_ids)
        return self.revised(
            current={aid: info for aid, info in self.current.items() if aid not in removed}
        )


class RetractedArtifact(Value):
    """A current artifact removed by an action, with the finding that caused it."""

    artifact_id: ArtifactId
    reason_ref: str


def validate_lineage(state: StudyState, action: ScientificActionId) -> str | None:
    """Require an existing question for its consumers; publication owns question exclusivity."""
    if action == "edit_question":
        return None
    return None if state.has("question") else "Set the study question first"


def apply_effects(
    state: StudyState,
    produced: Sequence[ArtifactRecord],
    retracted: Sequence[RetractedArtifact] | None = None,
) -> StudyState:
    """Install produced versions and retractions into a new state."""
    next_state = state.with_artifacts(produced)
    if retracted:
        next_state = next_state.without([item.artifact_id for item in retracted])
    return next_state


def is_stale(state: StudyState, artifact_id: ArtifactId) -> bool:
    """Whether an artifact's input chain references superseded versions.

    Findings belong to complete saved action results. This query follows only the
    immutable execution facts selected by the journal.
    """
    if artifact_id == "model":
        return False
    return _staleness(state, artifact_id, frozenset())


def _staleness(state: StudyState, artifact_id: ArtifactId, visiting: frozenset[ArtifactId]) -> bool:
    info = state.get(artifact_id)
    if info is None or artifact_id in visiting:
        return False
    marked = visiting.union((artifact_id,))
    for input_id in info.derived_from:
        # Preparation retains its model as provenance. Model edits do not alter
        # recorded data; compatibility is assessed against each consuming model.
        if artifact_id == "panel" and input_id == "model":
            continue
        current = state.get(input_id)
        if current is None or not state.matches_inputs(artifact_id, input_id):
            return True
        if input_id != "model" and _staleness(state, input_id, marked):
            return True
    return False
