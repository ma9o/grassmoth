"""Run fits and simulations against exact input revisions.

Each runner reads exactly its pinned input versions, writes new artifact versions with
the same pins in ``derived_from``, and returns effects for the workflow to publish.
Local studies can place numerical fitting on Modal without moving their history
or changing pinned versions. The enclosing action evaluates its checks before publication.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from grassmoth.public import FitInputs, FitOutput, InferenceHardware, StructuralSelection
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.actions.contracts.inputs import ContinueFitTarget, FitInput
from grassmoth_server.actions.contracts.requests import (
    DataDiffRequest,
    FitRequest,
    ModelDiffRequest,
    SimulateRequest,
    call_identity,
)
from grassmoth_server.actions.fit.selection import fit_source, prior_fit
from grassmoth_server.compute.remote_receipts import RemoteFitReceipt, load_remote_fit
from grassmoth_server.storage.paths import cache_dir
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import Applied

if TYPE_CHECKING:
    from grassmoth.public import GitOid
    from grassmoth_server.actions.contracts.identity import ArtifactId, CallId
    from grassmoth_server.actions.contracts.inputs import SimulateInput
    from grassmoth_server.study.state import StudyState


async def _run_fit(
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    request: FitInput[GitOid],
    call_id: CallId,
) -> Applied[GitOid]:
    """Keep continuation loading, compute and publication off the activity event loop."""
    return await asyncio.to_thread(_fit_selected, store, pins, request, call_id)


def _fit_selected(
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    request: FitInput[GitOid],
    call_id: CallId,
) -> Applied[GitOid]:
    from grassmoth_server.actions.fit.action import run_fit
    from grassmoth_server.actions.fit.execution import resolve_sampler_spec
    from grassmoth_server.config import get_config
    from grassmoth_server.study.queries.data import read_data_history

    source = fit_source(store, request.target)
    history = read_data_history(store, source.data_ref)
    previous = (
        prior_fit(store, request.target.fit_ref)
        if isinstance(request.target, ContinueFitTarget)
        else None
    )
    prior_settings = (
        previous.continuation
        if isinstance(previous, RemoteFitReceipt)
        else None
        if previous is None or previous.inference.evidence.checkpoint is None
        else previous.inference.evidence.checkpoint.settings
    )
    sampler = resolve_sampler_spec(request.settings, prior_settings)
    previous_cache = previous.cache if isinstance(previous, RemoteFitReceipt) else None
    if isinstance(previous, RemoteFitReceipt) and sampler.hardware == InferenceHardware.LOCAL:
        previous = load_remote_fit(
            Path(cache_dir(store.workspace_id)) / "remote-results", previous.cache
        )

    from grassmoth_server.study.persistence.store import model_values, read_model, read_question

    dynamical_model_spec = model_values(store, read_model(store, pins["model"]))
    question = read_question(store, pins["question"])
    inputs = FitInputs(
        selection=StructuralSelection.for_question(dynamical_model_spec, question),
        question=question,
        data_for_model=history.observations,
        sampler=sampler,
        compute_loo_diagnostics=get_config().inference.compute_loo_diagnostics,
        previous=previous if isinstance(previous, FitOutput) else None,
    )

    from grassmoth_server.compute.analysis_cache import AnalysisCall, fit_profile_directory

    analysis_call = AnalysisCall(workspace_id=store.workspace_id, call_id=call_id)
    if sampler.hardware != InferenceHardware.LOCAL:
        from grassmoth_server.actions.fit.remote import fit_on_modal

        result = fit_on_modal(
            inputs,
            analysis_call=analysis_call,
            profile=request.settings.profile,
            previous_cache=previous_cache,
        )
    else:
        result = run_fit(
            inputs,
            profile_dir=fit_profile_directory(Path(cache_dir(store.workspace_id)), analysis_call)
            if request.settings.profile
            else None,
        )

    from grassmoth_server.study.persistence.writes import write_model_revision

    saved = store.write_result(result)
    # A zero-draw continuation reuses the previous model without producing a revision.
    if isinstance(result, RemoteFitReceipt):
        produced = (
            (
                store.result_artifact(
                    store.write_artifact("model", derived_from=pins, produced_by="fit"), saved
                ),
            )
            if result.has_model and sampler.num_samples_per_chain > 0
            else ()
        )
    else:
        conditioned = result.dynamical_model_spec
        produced = (
            ()
            if conditioned is None or sampler.num_samples_per_chain == 0
            else (write_model_revision(store, conditioned, derived_from=pins, produced_by="fit"),)
        )
    return Applied(result=saved, effects=ActionEffects(produced=produced))


async def _run_simulate(
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    request: SimulateInput[GitOid],
) -> Applied[GitOid]:
    from grassmoth.public import ObservationPreflightFailure
    from grassmoth_server.actions.errors import ActionExecutionError
    from grassmoth_server.actions.simulate.action import run_simulation

    result = await asyncio.to_thread(run_simulation, store, request, pins["question"])
    if isinstance(result, ObservationPreflightFailure):
        raise ActionExecutionError(result.message)
    return Applied(result=store.write_result(result), effects=ActionEffects())


async def run_action_locally(
    workspace_id: str,
    request: FitRequest[GitOid]
    | SimulateRequest[GitOid]
    | DataDiffRequest[GitOid]
    | ModelDiffRequest[GitOid],
    pins: dict[ArtifactId, GitOid],
) -> Applied[GitOid]:
    """Run an action directly against its recorded inputs."""
    store = ArtifactStore(workspace_id)
    if isinstance(request, ModelDiffRequest):
        from grassmoth_server.actions.model_diff import model_diff

        report = await asyncio.to_thread(
            model_diff, workspace_id, request.input.before_ref, request.input.after_ref
        )
        return Applied(result=store.write_result(report), effects=ActionEffects())
    if isinstance(request, DataDiffRequest):
        from grassmoth_server.actions.data_diff import read_data_diff

        comparison = await asyncio.to_thread(read_data_diff, workspace_id, request)
        return Applied(
            result=store.write_result(comparison),
            effects=ActionEffects(),
        )
    if isinstance(request, FitRequest):
        return await _run_fit(
            store,
            pins,
            request.input,
            call_identity(request),
        )
    return await _run_simulate(store, pins, request.input)


def _pinned(store: ArtifactStore, selected: dict[ArtifactId, GitOid]) -> dict[ArtifactId, GitOid]:
    """Exact selected input revisions; every one must exist in the store."""
    for artifact_id, revision in selected.items():
        store.read_meta(artifact_id, revision)
    return dict(selected)


def _pin_action_inputs(
    workspace_id: str,
    request: FitRequest[GitOid]
    | SimulateRequest[GitOid]
    | DataDiffRequest[GitOid]
    | ModelDiffRequest[GitOid],
    state: StudyState,
) -> dict[ArtifactId, GitOid]:
    """Resolve selected inputs on the blocking execution boundary."""
    store = ArtifactStore(workspace_id)
    if isinstance(request, FitRequest):
        pins = _pinned(
            store,
            {
                "model": fit_source(store, request.input.target).dynamical_model_spec_ref,
                "question": state.current["question"].revision,
            },
        )
    elif isinstance(request, SimulateRequest):
        pins = _pinned(
            store,
            {
                "model": request.input.dynamical_model_spec_ref,
                "question": state.current["question"].revision,
            },
        )
    else:
        pins = {}
    return pins


async def run_action(
    workspace_id: str,
    request: FitRequest[GitOid]
    | SimulateRequest[GitOid]
    | DataDiffRequest[GitOid]
    | ModelDiffRequest[GitOid],
    state: StudyState,
) -> Applied[GitOid]:
    """Pin and compile inputs off the activity event loop, then execute the action."""
    pins = await asyncio.to_thread(_pin_action_inputs, workspace_id, request, state)
    return await run_action_locally(workspace_id, request, pins)
