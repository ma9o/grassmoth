"""Correlated scientific attempts; Git publication and attempt metadata have one owner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Self, cast
from uuid import UUID

from pydantic import AwareDatetime, Field

from nof1_causal_lab.actions.contracts.effects import ActionEffects
from nof1_causal_lab.actions.contracts.requests import (
    DataDiffRequest,
    EditModelRequest,
    EditQuestionRequest,
    FitRequest,
    ModelDiffRequest,
    PrepareDataRequest,
    ScientificActionRequest,
    SimulateRequest,
)
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model.dynamical_model_spec import ModelEditResult
from nof1_causal_lab.artifacts.observations.preparation import DataPreparationResult, FileSourceRef
from nof1_causal_lab.artifacts.reports.checks import FindingSubject
from nof1_causal_lab.artifacts.reports.inference import InferenceEvidence
from nof1_causal_lab.artifacts.simulation import ModelSimulationResult
from nof1_causal_lab.study.state import StudyState

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from nof1_causal_lab.artifacts.observations.reference import DataRef


class ActionMessage(Value):
    """A label emitted by an attempt; measurements belong in its scientific result."""

    kind: Literal["log"] = "log"
    timestamp: AwareDatetime
    severity: Literal["debug", "info", "warning", "error"]
    code: str
    subject: FindingSubject
    detail: str


class Applied[ResultT](Value):
    """A successful action result and the artifact changes it proposes to publish."""

    status: Literal["applied"] = Field(
        default="applied", description="Discriminator identifying an applied outcome."
    )
    result: ResultT = Field(
        description=(
            "Complete saved result reference, or the transient evidence awaiting completion."
        )
    )
    effects: ActionEffects = Field(
        description="Produced and retracted artifacts and retained report references."
    )


class Rejected(Value):
    """An expected refusal to execute or publish a scientific request."""

    status: Literal["rejected"] = Field(
        default="rejected", description="Discriminator identifying a rejected outcome."
    )
    code: str
    subject: FindingSubject
    detail: str = Field(
        description="Explanation of the specific condition that prevented application."
    )


class Raised(Value):
    """An unexpected execution failure retained as part of an attempt."""

    status: Literal["raised"] = Field(
        default="raised", description="Discriminator identifying an exception outcome."
    )
    error_type: str = Field(
        description="Exception or external failure type reported by the execution boundary."
    )
    error_message: str = Field(description="Human-readable failure explanation.")
    # Foreign failure payloads are retained as opaque text, never scientific inputs.
    details: tuple[str, ...] = Field(
        default=(), description="Opaque external failure details retained for diagnosis."
    )


type FailedOutcome = Rejected | Raised


class Attempt[ActionT: str, RequestT: Value, ResultT](Value):
    """One action's request and successful result share the same attempt owner."""

    action: ActionT
    request: RequestT | None = Field(
        description=(
            "Parsed arguments, or null for a historical attempt whose arguments were not retained"
        )
    )
    outcome: Annotated[Applied[ResultT] | Rejected | Raised, Field(discriminator="status")]


StagedEditQuestionAttempt = Attempt[Literal["edit_question"], EditQuestionRequest, None]
StagedEditAttempt = Attempt[Literal["edit_model"], EditModelRequest[GitOid], ModelEditResult]
StagedPrepareAttempt = Attempt[
    Literal["prepare_data"], PrepareDataRequest[GitOid, FileSourceRef], DataPreparationResult
]
StagedFitAttempt = Attempt[Literal["fit"], FitRequest[GitOid], None]
StagedSimulateAttempt = Attempt[Literal["simulate"], SimulateRequest[GitOid], None]
StagedDataDiffAttempt = Attempt[Literal["data_diff"], DataDiffRequest[GitOid], None]
StagedModelDiffAttempt = Attempt[Literal["model_diff"], ModelDiffRequest[GitOid], None]


type StagedActionAttempt = Annotated[
    StagedEditQuestionAttempt
    | StagedEditAttempt
    | StagedPrepareAttempt
    | StagedFitAttempt
    | StagedSimulateAttempt
    | StagedDataDiffAttempt
    | StagedModelDiffAttempt,
    Field(discriminator="action"),
]


EditQuestionAttempt = Attempt[Literal["edit_question"], EditQuestionRequest, GitOid]
EditAttempt = Attempt[Literal["edit_model"], EditModelRequest[GitOid], GitOid]
PrepareAttempt = Attempt[Literal["prepare_data"], PrepareDataRequest[GitOid, FileSourceRef], GitOid]
FitAttempt = Attempt[Literal["fit"], FitRequest[GitOid], GitOid]
SimulateAttempt = Attempt[Literal["simulate"], SimulateRequest[GitOid], GitOid]
DataDiffAttempt = Attempt[Literal["data_diff"], DataDiffRequest[GitOid], GitOid]
ModelDiffAttempt = Attempt[Literal["model_diff"], ModelDiffRequest[GitOid], GitOid]


type ActionAttempt = Annotated[
    EditQuestionAttempt
    | EditAttempt
    | PrepareAttempt
    | FitAttempt
    | SimulateAttempt
    | DataDiffAttempt
    | ModelDiffAttempt,
    Field(discriminator="action"),
]


class AttemptMetadata(Value):
    """Sequence, timing, and log references attached to a published attempt."""

    seq: int = Field(description="Attempt's position in the study journal.")
    attempt_id: UUID | None = Field(
        default=None, description="Execution identity used for progress tracking, when retained."
    )
    ts: str = Field(description="Recorded attempt timestamp.")
    messages: tuple[ActionMessage, ...] = Field(
        default=(), description="Structured action messages emitted during execution."
    )
    trace_ids: tuple[str, ...] = Field(
        default=(), description="References identifying the retained execution traces."
    )


class AttemptRecord(AttemptMetadata):
    """Stored inside the Git object, with no self-referential publication ID."""

    attempt: ActionAttempt

    def with_logs(self, *, messages: tuple[ActionMessage, ...], trace_ids: tuple[str, ...]) -> Self:
        """Return a record with replacement messages and trace references, preserving the attempt."""
        return self.revised(messages=messages, trace_ids=trace_ids)


class StudyRevision(Value):
    """Git publication wraps its already-owned record, without copying its fields."""

    commit_id: GitOid
    parent_ids: tuple[GitOid, ...]
    record: AttemptRecord


class RecordDependency(Value):
    """A journal dependency linking a call argument to the attempt that first published its input."""

    seq: int = Field(description="Sequence number of the consuming attempt.")
    source_seq: int = Field(
        description="Earlier sequence number that published the selected input."
    )
    argument: str = Field(description="Input reference field name with its `_ref` suffix removed.")


def failed_attempt(
    request: ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid],
    outcome: FailedOutcome,
) -> ActionAttempt:
    """Close the action/request relation for a rejection or execution failure."""
    match request:
        case EditQuestionRequest():
            return EditQuestionAttempt(action="edit_question", request=request, outcome=outcome)
        case EditModelRequest():
            return EditAttempt(action="edit_model", request=request, outcome=outcome)
        case PrepareDataRequest():
            return PrepareAttempt(action="prepare_data", request=request, outcome=outcome)
        case FitRequest():
            return FitAttempt(action="fit", request=request, outcome=outcome)
        case SimulateRequest():
            return SimulateAttempt(action="simulate", request=request, outcome=outcome)
        case DataDiffRequest():
            return DataDiffAttempt(action="data_diff", request=request, outcome=outcome)
        case ModelDiffRequest():
            return ModelDiffAttempt(action="model_diff", request=request, outcome=outcome)
    raise TypeError("Unknown action request")


def failed_staged_attempt(
    request: ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid],
    outcome: FailedOutcome,
) -> StagedActionAttempt:
    """Carry an expected execution refusal before result publication."""
    match request:
        case EditQuestionRequest():
            return StagedEditQuestionAttempt(
                action="edit_question", request=request, outcome=outcome
            )
        case EditModelRequest():
            return StagedEditAttempt(action="edit_model", request=request, outcome=outcome)
        case PrepareDataRequest():
            return StagedPrepareAttempt(action="prepare_data", request=request, outcome=outcome)
        case FitRequest():
            return StagedFitAttempt(action="fit", request=request, outcome=outcome)
        case SimulateRequest():
            return StagedSimulateAttempt(action="simulate", request=request, outcome=outcome)
        case DataDiffRequest():
            return StagedDataDiffAttempt(action="data_diff", request=request, outcome=outcome)
        case ModelDiffRequest():
            return StagedModelDiffAttempt(action="model_diff", request=request, outcome=outcome)
    raise TypeError("Unknown action request")


def applied_attempt[ResultT](
    request: ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid],
    applied: Applied[ResultT],
) -> StagedActionAttempt:
    """The execution transport is closed again before publication; mismatches are bugs."""
    result = applied.result
    match request, result:
        case EditQuestionRequest(), None:
            return StagedEditQuestionAttempt(
                action="edit_question",
                request=request,
                outcome=Applied(result=result, effects=applied.effects),
            )
        case EditModelRequest(), ModelEditResult():
            return StagedEditAttempt(
                action="edit_model",
                request=request,
                outcome=Applied(result=result, effects=applied.effects),
            )
        case PrepareDataRequest(), DataPreparationResult():
            return StagedPrepareAttempt(
                action="prepare_data",
                request=request,
                outcome=Applied(result=result, effects=applied.effects),
            )
        case FitRequest(), InferenceEvidence():
            return StagedFitAttempt(
                action="fit",
                request=request,
                outcome=Applied(result=None, effects=applied.effects),
            )
        case SimulateRequest(), ModelSimulationResult():
            return StagedSimulateAttempt(
                action="simulate",
                request=request,
                outcome=Applied(result=None, effects=applied.effects),
            )
        case DataDiffRequest(), None:
            return StagedDataDiffAttempt(
                action="data_diff",
                request=request,
                outcome=Applied(result=result, effects=applied.effects),
            )
        case ModelDiffRequest(), None:
            return StagedModelDiffAttempt(
                action="model_diff",
                request=request,
                outcome=Applied(result=result, effects=applied.effects),
            )
        case _:
            raise TypeError("Action result does not match its request")


def retained_attempt(
    attempt: StagedActionAttempt, result: GitOid, effects: ActionEffects
) -> ActionAttempt:
    """Close the saved action/result relationship after execution completes its body."""
    outcome = Applied(result=result, effects=effects)
    match attempt.action:
        case "edit_question":
            return EditQuestionAttempt(
                action=attempt.action, request=attempt.request, outcome=outcome
            )
        case "edit_model":
            return EditAttempt(action=attempt.action, request=attempt.request, outcome=outcome)
        case "prepare_data":
            return PrepareAttempt(action=attempt.action, request=attempt.request, outcome=outcome)
        case "fit":
            return FitAttempt(action=attempt.action, request=attempt.request, outcome=outcome)
        case "simulate":
            return SimulateAttempt(action=attempt.action, request=attempt.request, outcome=outcome)
        case "data_diff":
            return DataDiffAttempt(action=attempt.action, request=attempt.request, outcome=outcome)
        case "model_diff":
            return ModelDiffAttempt(action=attempt.action, request=attempt.request, outcome=outcome)


def argument_revisions(attempt: ActionAttempt) -> tuple[tuple[str, GitOid], ...]:
    """Action input fields ending in ``_ref`` name exact scalar or structured references.

    Their input schemas own the reference types: GitOid, DataRef, an optional
    reference, or a tuple of references. Inline scientific payloads are not scanned.
    """
    request = attempt.request
    if request is None:
        return ()
    references: list[tuple[str, GitOid]] = []
    selected = request.input.target if isinstance(request, FitRequest) else request.input
    for name in type(selected).model_fields:
        if not name.endswith("_ref"):
            continue
        value = cast(
            "GitOid | DataRef[GitOid, int | None] | tuple[GitOid | DataRef[GitOid, int | None], ...] | None",
            getattr(selected, name),
        )
        if value is not None:
            references.extend(
                (name.removesuffix("_ref"), ref if isinstance(ref, GitOid) else ref.revision)
                for ref in (value if isinstance(value, tuple) else (value,))
            )
    return tuple(references)


def record_dependencies(revisions: Sequence[StudyRevision]) -> list[RecordDependency]:
    """Join declared input references to their first successful publication."""
    producers: dict[GitOid, int] = {}
    for revision in revisions:
        record = revision.record
        outcome = record.attempt.outcome
        if not isinstance(outcome, Applied):
            continue
        producers[revision.commit_id] = record.seq
        for artifact in outcome.effects.produced:
            producers.setdefault(artifact.revision, record.seq)
    dependencies: list[RecordDependency] = []
    for revision in revisions:
        record = revision.record
        arguments: dict[tuple[int, str], None] = {}
        for name, identity in argument_revisions(record.attempt):
            source = producers.get(identity)
            if source is not None and source < record.seq:
                arguments[source, name] = None
        dependencies.extend(
            RecordDependency(seq=record.seq, source_seq=source, argument=name)
            for source, name in arguments
        )
    return dependencies


class ActionBase(Value):
    """Immutable execution base, captured before an action starts."""

    commit_id: GitOid
    state: StudyState
    saved: StudyRevision | None = None


def inference_record[T: StudyRevision](
    records: Iterable[T], dynamical_model_spec_revision: GitOid
) -> T | None:
    """Find the committed inference operation that produced this exact model value."""
    return next(
        (
            record
            for record in reversed(list(records))
            if record.record.attempt.action == "fit"
            and isinstance(record.record.attempt.outcome, Applied)
            and any(
                info.artifact_id == "model" and info.revision == dynamical_model_spec_revision
                for info in record.record.attempt.outcome.effects.produced
            )
        ),
        None,
    )
