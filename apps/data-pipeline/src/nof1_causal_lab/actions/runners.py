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

from nof1_causal_lab.actions.contracts.bodies import ContinueFitTarget, FitInput
from nof1_causal_lab.actions.contracts.effects import ActionEffects
from nof1_causal_lab.actions.contracts.requests import (
    DataDiffRequest,
    FitRequest,
    ModelDiffRequest,
    SimulateRequest,
)
from nof1_causal_lab.actions.errors import ModelFitError
from nof1_causal_lab.actions.fit.selection import fit_source, prior_fit
from nof1_causal_lab.artifacts.continuation import TransferredTuning, sampler_identity
from nof1_causal_lab.artifacts.sampling import InferenceHardware
from nof1_causal_lab.artifacts.simulation import ModelSimulationResult
from nof1_causal_lab.models.structure import StructuralSelection
from nof1_causal_lab.study.artifact_files import json_filename
from nof1_causal_lab.study.persistence.store import ArtifactStore
from nof1_causal_lab.study.records import Applied

if TYPE_CHECKING:
    from nof1_causal_lab.actions.contracts.bodies import SimulateInput
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
    from nof1_causal_lab.artifacts.reports.inference import InferenceEvidence
    from nof1_causal_lab.study.state import StudyState


async def _run_fit(
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    request: FitInput[GitOid],
) -> Applied[InferenceEvidence]:
    """Keep continuation loading, compute and publication off the activity event loop."""
    return await asyncio.to_thread(_fit_selected, store, pins, request)


def _fit_selected(
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    request: FitInput[GitOid],
) -> Applied[InferenceEvidence]:
    from nof1_causal_lab.actions.fit.action import fit, read_inference_report
    from nof1_causal_lab.actions.fit.execution import resolve_sampler_spec
    from nof1_causal_lab.config import get_config
    from nof1_causal_lab.study.queries.data import read_data_history

    source = fit_source(store, request.target)
    history = read_data_history(store, source.data_ref)
    previous = (
        prior_fit(store, request.target.fit_ref)
        if isinstance(request.target, ContinueFitTarget)
        else None
    )
    continuation = None if previous is None else previous.inference.evidence.checkpoint
    from functools import cache

    from nof1_causal_lab.study.persistence.store import read_model, read_question

    dynamical_model_spec = read_model(store, pins["model"])
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
        if sampler_identity(sampler) != sampler_identity(continuation.settings):
            if sampler.num_warmup == 0:
                raise ModelFitError(
                    "Switching algorithm or joint parameterization requires positive warmup",
                    diagnostics={"code": "CONTINUATION_REQUIRES_WARMUP"},
                )
            continuation = continuation.revised(tuning=TransferredTuning())
    from nof1_causal_lab.actions.checks import check_model_data
    from nof1_causal_lab.artifacts.reports.checks import Evaluated
    from nof1_causal_lab.models.ssm.compile.inputs import compile_ssm_inputs_from_model

    findings = check_model_data(
        compile_ssm_inputs_from_model(selection),
        history.observations,
        time_origin=history.time_origin,
        sampler=sampler,
    )
    if any(not isinstance(item, Evaluated) or item.outcome != "passed" for item in findings):
        raise ModelFitError(
            "Fit preflight rejected the inputs before compute allocation",
            diagnostics={"findings": [item.model_dump(mode="json") for item in findings]},
        )

    config = get_config().inference
    if sampler.hardware != InferenceHardware.LOCAL:
        from nof1_causal_lab.actions.fit.remote import fit_on_modal

        compute_fit = partial(fit_on_modal, sampler.hardware)
    else:
        compute_fit = fit
    result = compute_fit(
        selection=selection,
        data_for_model=history.observations,
        time_origin=history.time_origin,
        sampler=sampler,
        array_writer=store.write_array,
        array_loader=cache(store.read_array),
        compute_loo_diagnostics=config.compute_loo_diagnostics,
        continuation=continuation,
        previous=previous,
    )

    conditioned = result["_dynamical_model_spec"]
    evidence = result["evidence"]
    if conditioned is None:
        report = (
            read_inference_report(store, pins["model"], evidence, result["metadata"])
            if previous is None
            else previous.inference.revised(
                evidence=evidence,
                core=previous.inference.core.revised(inference_metadata=result["metadata"]),
            )
        )
        effects = ActionEffects(reports={"inference": store.write_report(report)})
    else:
        info = store.write_artifact(
            "model",
            derived_from=pins,
            produced_by="fit",
            json_files={
                json_filename("model", "model"): conditioned.model_dump(
                    mode="json", round_trip=True
                )
            },
        )
        report = read_inference_report(store, info.revision, evidence, result["metadata"])
        effects = ActionEffects(produced=(info,), reports={"inference": store.write_report(report)})
    from nof1_causal_lab.actions.model_checks import fit_check_report

    checks = fit_check_report(selection, question, history, preflight=findings)
    return Applied(
        result=evidence,
        effects=effects.revised(reports={**effects.reports, "checks": store.write_report(checks)}),
    )


async def _run_simulate(
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    request: SimulateInput[GitOid],
) -> Applied[ModelSimulationResult]:
    from nof1_causal_lab.actions.simulate.action import read_simulation_report, simulate
    from nof1_causal_lab.study.persistence.store import read_model, read_question

    dynamical_model_spec = read_model(store, pins["model"])
    selection = StructuralSelection.for_question(
        dynamical_model_spec, read_question(store, pins["question"])
    )
    report = await asyncio.to_thread(
        simulate,
        selection,
        request,
        store=store,
    )
    from nof1_causal_lab.actions.errors import ActionExecutionError
    from nof1_causal_lab.models.ssm.preparation.preflight import ObservationPreflightFailure

    if isinstance(report, ObservationPreflightFailure):
        raise ActionExecutionError(report.message)
    retained = ModelSimulationResult(evidence=report)
    findings = read_simulation_report(store, retained.evidence, request, pins["question"])
    return Applied(
        result=retained,
        effects=ActionEffects(reports={"simulation": store.write_report(findings)}),
    )


async def run_action_locally(
    workspace_id: str,
    request: FitRequest[GitOid]
    | SimulateRequest[GitOid]
    | DataDiffRequest[GitOid]
    | ModelDiffRequest[GitOid],
    pins: dict[ArtifactId, GitOid],
) -> Applied[InferenceEvidence] | Applied[ModelSimulationResult] | Applied[None]:
    """Run an action directly against its recorded inputs."""
    store = ArtifactStore(workspace_id)
    if isinstance(request, ModelDiffRequest):
        from nof1_causal_lab.actions.model_diff import model_diff

        report = await asyncio.to_thread(
            model_diff, workspace_id, request.input.before_ref, request.input.after_ref
        )
        return Applied(
            result=None, effects=ActionEffects(reports={"model-diff": store.write_result(report)})
        )
    if isinstance(request, DataDiffRequest):
        from nof1_causal_lab.actions.data_diff import read_data_diff

        comparison = await asyncio.to_thread(read_data_diff, workspace_id, request)
        return Applied(
            result=None,
            effects=ActionEffects(reports={"data-diff": store.write_report(comparison)}),
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
    """Resolve and compile selected inputs on the same blocking execution boundary."""
    store = ArtifactStore(workspace_id)
    if isinstance(request, FitRequest):
        pins = _pinned(
            store,
            {
                "model": fit_source(store, request.input.target).dynamical_model_spec_ref,
                "question": state.current["question"].revision,
            },
        )
        from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model
        from nof1_causal_lab.study.persistence.store import read_model, read_question

        compile_executable_model(
            StructuralSelection.for_question(
                read_model(store, pins["model"]), read_question(store, pins["question"])
            )
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
) -> Applied[InferenceEvidence] | Applied[ModelSimulationResult] | Applied[None]:
    """Pin and compile inputs off the activity event loop, then execute the action."""
    pins = await asyncio.to_thread(_pin_action_inputs, workspace_id, request, state)
    return await run_action_locally(workspace_id, request, pins)
