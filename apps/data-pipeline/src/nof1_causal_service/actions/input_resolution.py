"""Resolve HTTP revision selectors to immutable scientific inputs in submission order."""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid, RevisionSelector
from nof1_causal_lab.artifacts.observations.preparation import FileSourceRef, SourceFolder
from nof1_causal_lab.artifacts.observations.reference import DataRef, DataSelection
from nof1_causal_service.actions.contracts.inputs import (
    ContinueFitTarget,
    DataDiffInput,
    EditModelInput,
    FitInput,
    ModelDiffInput,
    NewFitTarget,
    PrepareDataInput,
    SimulateInput,
)
from nof1_causal_service.actions.contracts.requests import (
    ActionInput,
    DataDiffRequest,
    EditModelRequest,
    EditQuestionRequest,
    FitRequest,
    ModelDiffRequest,
    PrepareDataRequest,
    ScientificActionRequest,
    SimulateRequest,
)
from nof1_causal_service.storage import backend as storage
from nof1_causal_service.storage import paths as data
from nof1_causal_service.study.errors import StudyLookupError
from nof1_causal_service.study.records import Applied
from nof1_causal_service.study.state import ArtifactRecord, StudyState, is_stale

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_service.study.persistence.history import StudyRepository
    from nof1_causal_service.study.persistence.store import ArtifactStore

type RevisionInputKind = Literal["question", "model", "panel", "simulation", "model_parent", "fit"]


def resolve_action_inputs(
    repository: StudyRepository, request: ActionInput
) -> tuple[
    ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid],
    Mapping[str, bytes],
]:
    """Pin selectors before call identity, keeping unpublished and failed outputs ineligible.

    Question, model and panel selectors use derived current selections, honoring retractions
    and input freshness. Simulation and fit selectors use submission order.
    Explicit hashes retain their meaning.
    """
    records = repository.records()
    state = repository.state(records[-1].commit_id) if records else StudyState()
    contents: Mapping[str, bytes] = dict[str, bytes]()

    @cache
    def latest(kind: RevisionInputKind) -> GitOid:
        if kind == "model_parent":
            return latest("model" if state.has("model") else "question")
        if kind == "simulation" or kind == "fit":
            for revision in reversed(records):
                attempt = revision.record.attempt
                if attempt.action == ("simulate" if kind == "simulation" else "fit") and isinstance(
                    attempt.outcome, Applied
                ):
                    return revision.commit_id
        else:
            artifact = state.get(kind)
            if artifact is not None and not is_stale(state, kind):
                return artifact.revision
        raise StudyLookupError(f"No valid {kind} revision is available for 'latest'")

    def resolve(selector: RevisionSelector, kind: RevisionInputKind) -> GitOid:
        return latest(kind) if selector == "latest" else selector

    def data_ref[IndexT: int | None](
        source: DataRef[RevisionSelector, IndexT],
    ) -> DataRef[GitOid, IndexT]:
        return DataRef[GitOid, IndexT](
            revision=resolve(source.revision, "panel"), replicate_index=source.replicate_index
        )

    def selection(value: DataSelection[RevisionSelector]) -> DataSelection[GitOid]:
        return tuple(data_ref(source) for source in value)

    match request:
        case EditQuestionRequest():
            resolved = request
        case EditModelRequest():
            resolved = EditModelRequest[GitOid](
                reasoning=request.reasoning,
                input=EditModelInput[GitOid](
                    parent_ref=resolve(request.input.parent_ref, "model_parent"),
                    dynamical_model_spec=request.input.dynamical_model_spec,
                ),
            )
        case PrepareDataRequest():
            dynamical_model_spec_ref = resolve(request.input.dynamical_model_spec_ref, "model")
            source, contents = _source_files(repository.workspace_id, request.input.source)
            resolved = PrepareDataRequest[GitOid, FileSourceRef](
                input=PrepareDataInput[GitOid, FileSourceRef](
                    dynamical_model_spec_ref=dynamical_model_spec_ref,
                    source=source,
                    extraction=request.input.extraction,
                    context=request.input.context,
                ),
                reasoning=request.reasoning,
            )
        case FitRequest():
            target = request.input.target
            resolved_target = (
                NewFitTarget[GitOid](
                    dynamical_model_spec_ref=resolve(target.dynamical_model_spec_ref, "model"),
                    data_ref=data_ref(target.data_ref),
                )
                if isinstance(target, NewFitTarget)
                else ContinueFitTarget[GitOid](fit_ref=resolve(target.fit_ref, "fit"))
            )
            resolved = FitRequest[GitOid](
                reasoning=request.reasoning,
                input=FitInput[GitOid](
                    target=resolved_target,
                    settings=request.input.settings,
                    repeat=request.input.repeat,
                ),
            )
        case SimulateRequest():
            resolved = SimulateRequest[GitOid](
                reasoning=request.reasoning,
                input=SimulateInput[GitOid](
                    simulation=request.input.simulation,
                    seed=request.input.seed,
                    num_replicates=request.input.num_replicates,
                    dynamical_model_spec_ref=resolve(
                        request.input.dynamical_model_spec_ref, "model"
                    ),
                ),
            )
        case DataDiffRequest():
            resolved = DataDiffRequest[GitOid](
                reasoning=request.reasoning,
                input=DataDiffInput[GitOid](
                    left_ref=selection(request.input.left_ref),
                    right_ref=selection(request.input.right_ref),
                ),
            )
        case ModelDiffRequest():
            resolved = ModelDiffRequest[GitOid](
                reasoning=request.reasoning,
                input=ModelDiffInput[GitOid](
                    before_ref=resolve(request.input.before_ref, "model_parent"),
                    after_ref=resolve(request.input.after_ref, "model_parent"),
                ),
            )
    return resolved, contents


def model_edit_state(store: ArtifactStore, parent_ref: GitOid) -> StudyState:
    """Resolve a question parent or a model parent with its pinned question."""
    parent = store.read_record(parent_ref)
    if parent.artifact_id == "question":
        return StudyState(current={"question": parent})
    if parent.artifact_id == "model":
        question = store.read_meta("question", parent.derived_from["question"])
        return StudyState(current={"question": question, "model": parent})
    raise StudyLookupError("Model edits require a question or model parent")


def _source_files(
    workspace_id: str, folder: SourceFolder
) -> tuple[FileSourceRef, Mapping[str, bytes]]:
    """Capture the folder's ordered file paths and bytes before naming the scientific call."""
    workspace = data.workspace_dir(workspace_id)
    directory = storage.join(workspace, folder)
    if not storage.exists(directory):
        raise StudyLookupError(f"Source folder does not exist: {folder}")
    files = sorted(storage.walk_files(directory))
    if not files:
        raise StudyLookupError(f"Source folder contains no files: {folder}")
    contents: dict[str, bytes] = {}
    for path in files:
        if not storage.is_remote() and not Path(path).resolve().is_relative_to(
            Path(workspace).resolve()
        ):
            raise StudyLookupError("Source files must remain within the workspace data directory")
        relative = path.removeprefix(workspace.rstrip("/") + "/")
        with storage.open_file(path, "rb") as handle:
            contents[relative] = handle.read()
    return FileSourceRef(
        files=tuple(contents),
        hashes={name: hashlib.sha256(body).hexdigest() for name, body in contents.items()},
    ), contents


def input_state(
    repository: StudyRepository,
    request: ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid],
) -> StudyState:
    """Select only the question and the revisions actually named by this call."""
    from nof1_causal_service.study.persistence.store import ArtifactStore

    store = ArtifactStore(repository.workspace_id, repository_path=repository.path)
    if isinstance(request, EditModelRequest):
        return model_edit_state(store, request.input.parent_ref)
    records = repository.attempts()
    question = next(
        (
            item
            for revision in records
            if revision.record.attempt.action == "edit_question"
            and isinstance(revision.record.attempt.outcome, Applied)
            for item in revision.record.attempt.outcome.effects.produced
            if item.artifact_id == "question"
        ),
        None,
    )
    current: dict[ArtifactId, ArtifactRecord] = (
        {"question": question} if question is not None else {}
    )
    from nof1_causal_service.actions.fit.selection import fit_source

    source = fit_source(store, request.input.target) if isinstance(request, FitRequest) else None
    dynamical_model_spec_revision = (
        source.dynamical_model_spec_ref
        if source is not None
        else request.input.dynamical_model_spec_ref
        if isinstance(request, (SimulateRequest, PrepareDataRequest))
        else None
    )
    panel: GitOid | None = None
    if source is not None:
        from nof1_causal_lab.artifacts.observations.preparation import PreparedDataMetadata
        from nof1_causal_service.study.queries.data import panel_revision, read_data_source

        if isinstance(read_data_source(store, source.data_ref.revision), PreparedDataMetadata):
            panel = panel_revision(store, source.data_ref.revision)
    if dynamical_model_spec_revision is not None:
        current["model"] = store.read_meta("model", dynamical_model_spec_revision)
        if source is not None:
            current["question"] = store.read_meta(
                "question", current["model"].derived_from["question"]
            )
    if panel is not None:
        current["panel"] = store.read_meta("panel", panel)
        effects = next(
            (
                outcome.effects
                for revision in records
                if isinstance(outcome := revision.record.attempt.outcome, Applied)
                and any(
                    item.artifact_id == "panel" and item.revision == panel
                    for item in outcome.effects.produced
                )
            ),
            None,
        )
        if effects is not None:
            current.update(
                {
                    item.artifact_id: item
                    for item in effects.produced
                    if item.artifact_id == "raw_data"
                }
            )
    data = (
        source.data_ref
        if source is not None
        else DataRef[GitOid, int](revision=panel, replicate_index=0)
        if panel is not None
        else None
    )
    if isinstance(request, PrepareDataRequest):
        from nof1_causal_service.actions.prepare_data.resolve import resolve_preparation

        resolve_preparation(store, request.input)
    return StudyState(current=current, data=data)
