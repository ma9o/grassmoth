"""Run fits and simulations against exact input revisions.

Each runner reads exactly its pinned input versions, writes new artifact versions with
the same pins in ``derived_from``, and returns effects for the workflow to publish.
Local studies can place numerical fitting on Modal without moving their history
or changing pinned versions. The enclosing action evaluates its checks before publication.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING

from nof1_causal_lab.public import InferenceHardware, StructuralSelection, TransferredTuning
from nof1_causal_service.actions.contracts.effects import ActionEffects
from nof1_causal_service.actions.contracts.inputs import ContinueFitTarget, FitInput
from nof1_causal_service.actions.contracts.requests import (
    DataDiffRequest,
    FitRequest,
    ModelDiffRequest,
    SimulateRequest,
)
from nof1_causal_service.actions.errors import ModelFitError
from nof1_causal_service.actions.fit.selection import fit_source, prior_fit
from nof1_causal_service.study.persistence.store import ArtifactStore
from nof1_causal_service.study.records import Applied

if TYPE_CHECKING:
    from nof1_causal_lab.public import GitOid
    from nof1_causal_service.actions.contracts.identity import ArtifactId
    from nof1_causal_service.actions.contracts.inputs import SimulateInput
    from nof1_causal_service.study.state import StudyState


async def _run_fit(
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    request: FitInput[GitOid],
) -> Applied[GitOid]:
    """Keep continuation loading, compute and publication off the activity event loop."""
    return await asyncio.to_thread(_fit_selected, store, pins, request)


def _fit_selected(
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    request: FitInput[GitOid],
) -> Applied[GitOid]:
    from nof1_causal_service.actions.fit.action import run_fit
    from nof1_causal_service.actions.fit.execution import resolve_sampler_spec
    from nof1_causal_service.config import get_config
    from nof1_causal_service.study.queries.data import read_data_history

    source = fit_source(store, request.target)
    history = read_data_history(store, source.data_ref)
    previous = (
        prior_fit(store, request.target.fit_ref)
        if isinstance(request.target, ContinueFitTarget)
        else None
    )
    continuation = None if previous is None else previous.inference.evidence.checkpoint

    from nof1_causal_service.study.persistence.store import model_values, read_model, read_question

    dynamical_model_spec = model_values(store, read_model(store, pins["model"]))
    question = read_question(store, pins["question"])
    selection = StructuralSelection.for_question(dynamical_model_spec, question)
    sampler = resolve_sampler_spec(
        request.settings, None if continuation is None else continuation.settings
    )
    if continuation is not None:
        if (
            sampler.num_chains != continuation.settings.num_chains
            or sampler.seed != continuation.settings.seed
        ):
            raise ModelFitError(
                "Continuation inherits its chain count and random stream",
                diagnostics={"code": "INCOMPATIBLE_CONTINUATION"},
            )
        if sampler.kernel_identity != continuation.settings.kernel_identity:
            if sampler.num_warmup == 0:
                raise ModelFitError(
                    "Switching algorithm or joint parameterization requires positive warmup",
                    diagnostics={"code": "CONTINUATION_REQUIRES_WARMUP"},
                )
            continuation = continuation.revised(tuning=TransferredTuning())

    config = get_config().inference
    if sampler.hardware != InferenceHardware.LOCAL:
        from nof1_causal_service.actions.fit.remote import fit_on_modal

        compute_fit = partial(fit_on_modal, sampler.hardware)
    else:
        compute_fit = run_fit
    result = compute_fit(
        selection=selection,
        data_for_model=history.observations,
        time_origin=history.time_origin,
        sampler=sampler,
        question=question,
        compute_loo_diagnostics=config.compute_loo_diagnostics,
        continuation=continuation,
        previous=previous,
    )

    from nof1_causal_service.study.persistence.writes import write_model_revision

    # A zero-draw continuation reuses the previous model without producing a revision.
    conditioned = result.dynamical_model_spec
    produced = (
        ()
        if conditioned is None or sampler.num_samples_per_chain == 0
        else (write_model_revision(store, conditioned, derived_from=pins, produced_by="fit"),)
    )
    return Applied(result=store.write_result(result), effects=ActionEffects(produced=produced))


async def _run_simulate(
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    request: SimulateInput[GitOid],
) -> Applied[GitOid]:
    from nof1_causal_lab.public import ObservationPreflightFailure
    from nof1_causal_service.actions.errors import ActionExecutionError
    from nof1_causal_service.actions.simulate.action import run_simulation

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
        from nof1_causal_service.actions.model_diff import model_diff

        report = await asyncio.to_thread(
            model_diff, workspace_id, request.input.before_ref, request.input.after_ref
        )
        return Applied(result=store.write_result(report), effects=ActionEffects())
    if isinstance(request, DataDiffRequest):
        from nof1_causal_service.actions.data_diff import read_data_diff

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
