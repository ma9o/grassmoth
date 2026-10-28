"""Retain posterior atoms, native telemetry, and diagnostics during fit execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import wraps
from typing import TYPE_CHECKING, cast

import numpy as np

from grassmoth.actions.contracts.bodies import FitOutput
from grassmoth.actions.fit.conditioning import condition_model
from grassmoth.actions.fit.diagnostics import segment_diagnostics
from grassmoth.actions.fit.execution import ModelFitFailure
from grassmoth.actions.fit.inputs import ContinuationRefusal, resumed_checkpoint
from grassmoth.artifacts.arrays import ArrayVector, NumericalArray
from grassmoth.artifacts.identity import ConstructRef, ParameterRef
from grassmoth.artifacts.model.parameter import ParameterCoordinate
from grassmoth.artifacts.reports.inference import (
    ConvergenceCheckpointReport,
    ConvergenceStoppingReport,
    FitSegment,
    InferenceEvidence,
    InferenceMetadata,
    InferenceReport,
    InferenceReportCore,
    InferenceReportDetail,
)
from grassmoth.artifacts.reports.profiling import ProgramCompileReport
from grassmoth.artifacts.sampling import MarginalParticleGibbsSpec
from grassmoth.models.ssm.inference.diagnostics.convergence import chain_convergence
from grassmoth.models.ssm.inference.samplers.profiling import compilation_seconds, log_profile
from grassmoth.models.ssm.inference.samplers.progress import capture_phases
from grassmoth.models.ssm.inference.types import UnretainedRun

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FunctionType

    import jax
    from jax.stages import Compiled

    from grassmoth.actions.fit.inputs import FitInputs
    from grassmoth.artifacts.continuation import SamplerCheckpoint
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth.artifacts.reports.inference import FitRuntime
    from grassmoth.artifacts.reports.posterior_diagnostics import ChainDiagnostics
    from grassmoth.artifacts.reports.profiling import FitProfileReport
    from grassmoth.artifacts.reports.scorecard import MovementWindow
    from grassmoth.compute.budget import MemoryDemand
    from grassmoth.compute.context import ModelExecutionContext
    from grassmoth.compute.programs import PreparedProgram


@dataclass(frozen=True)
class FitResult:
    """The fitted model definition paired with its retained numerical inference evidence."""

    dynamical_model_spec: DynamicalModelSpec | None
    evidence: InferenceEvidence
    metadata: InferenceMetadata
    profile: FitProfileReport | None = None


def _fit(
    inputs: FitInputs,
    *,
    continuation: SamplerCheckpoint | None,
    runtime: FitRuntime,
    execution_context: ModelExecutionContext,
) -> FitResult | ModelFitFailure:
    """The exact producer's buffers cross the compute boundary without report reductions."""
    from grassmoth.actions.fit.execution import fit_model

    selection, sampler, previous = inputs.selection, inputs.sampler, inputs.previous
    with capture_phases(execution_context.clock) as phases:
        fitted = fit_model(inputs, continuation=continuation, execution_context=execution_context)
    if isinstance(fitted, ModelFitFailure):
        return fitted
    result = fitted.result
    profiling = result.profile if isinstance(result, UnretainedRun) else result.diagnostics.profile
    old_count = (
        0
        if previous is None
        else previous.inference.core.inference_metadata.num_samples_total // sampler.num_chains
    )

    def segment(diagnostics: ChainDiagnostics | None) -> FitSegment:
        retained_draws = (
            0 if isinstance(result, UnretainedRun) else result.diagnostics.mcmc.num_samples
        )
        checkpoint = (
            None if isinstance(result, UnretainedRun) else result.diagnostics.stopping_checkpoint
        )
        return FitSegment(
            diagnostics=diagnostics,
            convergence=chain_convergence(diagnostics),
            start=old_count,
            stop=old_count + retained_draws,
            settings=sampler,
            convergence_stopping=ConvergenceStoppingReport(
                retained_draws=retained_draws,
                checkpoint=None
                if checkpoint is None
                else ConvergenceCheckpointReport(draws=checkpoint.draws, report=checkpoint.report),
            )
            if sampler.stop_when_converged is not None
            else None,
            runtime=runtime,
            phases=tuple(phases),
            parameterization=result.parameterization
            if isinstance(result, UnretainedRun)
            else result.diagnostics.parameterization,
        )

    encoded_arrays: dict[int, tuple[object, NumericalArray]] = {}

    def encode(values: jax.Array | np.ndarray) -> NumericalArray:
        # Keep the source alive so its identity cannot be reused within this fit.
        identity = id(values)
        retained = encoded_arrays.get(identity)
        if retained is None:
            retained = (values, NumericalArray.from_numpy(np.asarray(values)))
            encoded_arrays[identity] = retained
        return retained[1]

    if isinstance(result, UnretainedRun):
        phase_fields = {
            phase: {name: encode(value) for name, value in fields.items()}
            for phase, fields in result.phase_extra_fields.items()
        }
        if previous is not None:
            prior = previous.inference
            return FitResult(
                profile=profiling,
                dynamical_model_spec=None,
                metadata=prior.core.inference_metadata.revised(
                    duration_seconds=fitted.duration_seconds,
                    segments=(*prior.core.inference_metadata.segments, segment(None)),
                ),
                evidence=(
                    prior.evidence.revised(phase_extra_fields=phase_fields)
                ).with_previous_phases(prior.evidence),
            )
        return FitResult(
            profile=profiling,
            dynamical_model_spec=None,
            metadata=InferenceMetadata(
                distribution=None,
                num_samples_total=0,
                num_chains=sampler.num_chains,
                duration_seconds=fitted.duration_seconds,
                engine=result.evidence,
                sampler_diagnostics=None,
                segments=(segment(None),),
            ),
            evidence=InferenceEvidence(phase_extra_fields=phase_fields),
        )
    native = result.diagnostics

    def retain(values: jax.Array | np.ndarray | None) -> NumericalArray | None:
        return encode(values) if values is not None else None

    conditioned, identity = condition_model(
        selection.dynamical_model_spec,
        fitted.panel.compiled_dynamical_model,
        result,
        times=fitted.panel.times,
        time_origin=inputs.data_for_model.time_origin,
    )
    layout = conditioned.law_layouts[identity]
    latent_indices = (
        tuple(
            i
            for identity in layout.constructs
            for i, (ref, _) in enumerate(native.latent_chains.states)
            if ref.id == identity
        )
        if native.latent_chains is not None
        else ()
    )
    evidence = InferenceEvidence(
        checkpoint=native.checkpoint,
        latent_free_mask=retain(native.latent_chains.free_mask[:, latent_indices])
        if native.latent_chains is not None
        else None,
        chain_extra_fields={
            name: encode(values)
            for name, values in native.mcmc.get_extra_fields(group_by_chain=True).items()
        },
        observation_log_probs=retain(native.observation_log_probs)
        if inputs.compute_loo_diagnostics
        else None,
        observed_rows=retain(np.any(np.isfinite(fitted.panel.observations), axis=1))
        if inputs.compute_loo_diagnostics
        or (
            isinstance(sampler.algorithm, MarginalParticleGibbsSpec)
            and "regional_movement" in sampler.algorithm.diagnostic_metrics
        )
        else None,
        exact_observation_rows=retain(native.exact_observation_rows),
        phase_extra_fields={
            phase: {name: encode(values) for name, values in buffers.items()}
            for phase, buffers in (native.phase_extra_fields or {}).items()
        },
        warmup_complete_log_posterior_history=retain(native.warmup_complete_log_posterior_history),
        all_complete_log_posterior_history=retain(native.all_complete_log_posterior_history),
        initial_latent_delta=retain(result.initial_latent_delta),
        final_latent_delta=retain(result.final_latent_delta),
    )
    metadata = InferenceMetadata(
        distribution=identity,
        segments=(
            *(previous.inference.core.inference_metadata.segments if previous is not None else ()),
            segment(
                segment_diagnostics(
                    conditioned, identity, evidence, sampler.num_chains, result.evidence
                )
            ),
        ),
        engine=result.evidence,
        duration_seconds=fitted.duration_seconds,
        num_chains=native.mcmc.num_chains,
        num_samples_total=native.mcmc.num_chains * (old_count + native.mcmc.num_samples),
        sampler_diagnostics=native.sampler_diagnostics,
    )
    if previous is not None:
        from grassmoth.artifacts.probability.serialization import (
            empirical_atoms,
            empirical_distribution,
        )

        assert previous.dynamical_model_spec is not None
        old = empirical_atoms(previous.dynamical_model_spec.distributions[identity])
        new = empirical_atoms(conditioned.distributions[identity])
        chains = sampler.num_chains
        combined = np.concatenate(
            (
                old.reshape(chains, old_count, -1),
                new.reshape(chains, native.mcmc.num_samples, -1),
            ),
            axis=1,
        )
        law = empirical_distribution(
            combined.reshape(-1, combined.shape[-1]),
        )
        conditioned = conditioned.with_entities(
            distributions={**conditioned.distributions, identity: law}
        )
        prior_evidence = previous.inference.evidence
        # A different kernel may expose different transition diagnostics. Only common
        # fields describe the full chain; each segment retains its native telemetry.
        evidence = evidence.revised(
            chain_extra_fields={
                name: NumericalArray.from_numpy(
                    np.concatenate(
                        (prior_evidence.chain_extra_fields[name].values, value.values), axis=1
                    )
                )
                for name, value in evidence.chain_extra_fields.items()
                if name in prior_evidence.chain_extra_fields
                and prior_evidence.chain_extra_fields[name].values.shape[2:]
                == value.values.shape[2:]
            },
            observation_log_probs=NumericalArray.from_numpy(
                np.concatenate(
                    (
                        prior_evidence.observation_log_probs.values,
                        evidence.observation_log_probs.values,
                    ),
                    axis=1,
                )
            )
            if prior_evidence.observation_log_probs is not None
            and evidence.observation_log_probs is not None
            else None,
        )
        evidence = evidence.with_previous_phases(prior_evidence)
    return FitResult(
        dynamical_model_spec=conditioned, evidence=evidence, metadata=metadata, profile=profiling
    )


def inference_report(
    dynamical_model_spec: DynamicalModelSpec,
    input_model: DynamicalModelSpec,
    evidence: InferenceEvidence,
    metadata: InferenceMetadata,
    windows: tuple[MovementWindow, ...],
) -> InferenceReport:
    """Compute exact-chain diagnostics from this fit's atoms and native telemetry."""
    if metadata.distribution is None:
        return InferenceReport(
            evidence=evidence,
            core=InferenceReportCore(
                inference_metadata=metadata,
                inference_diagnostics=None,
                convergence=chain_convergence(None),
                posterior_marginals=(),
                prior_densities={},
            ),
            detail=InferenceReportDetail(),
        )

    def render() -> InferenceReport:
        from grassmoth.artifacts.probability.serialization import empirical_atoms
        from grassmoth.models.ssm.inference.diagnostics.latent import LatentChainEvidence
        from grassmoth.models.ssm.inference.diagnostics.plots import (
            compute_posterior_marginals,
        )
        from grassmoth.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
        from grassmoth.models.ssm.inference.types import (
            JointPosteriorDraws,
            MCMCPosterior,
            ProductionDiagnostics,
        )

        assert metadata.distribution is not None
        layouts = {metadata.distribution: dynamical_model_spec.law_layouts[metadata.distribution]}
        atoms = {
            identity: empirical_atoms(dynamical_model_spec.distributions[identity])
            for identity in layouts
        }
        count = next(iter(atoms.values())).shape[0]
        samples: dict[str, np.ndarray] = {
            element: np.asarray(atoms[identity][:, layout.parameter_columns[element]])
            for identity, layout in layouts.items()
            for _, elements in layout.parameters
            for element in elements
        }
        references = {
            ParameterCoordinate(site_name=element, indices=()): (
                layout.labels[element],
                ParameterRef(parameter_id=parameter, element_id=element),
            )
            for layout in layouts.values()
            for parameter, elements in layout.parameters
            for element in elements
        }
        marginals = compute_posterior_marginals(samples, references)
        extra = {name: ref.values for name, ref in evidence.chain_extra_fields.items()}
        diagnostics = None
        traces, ranks, loo = (), (), None
        layout = layouts[metadata.distribution]
        if samples or layout.constructs:
            chains = metadata.num_chains
            if count % chains:
                raise ValueError("Retained atoms do not match their original chain count")
            posterior = MCMCPosterior.from_run(
                draws=JointPosteriorDraws(parameters=samples),
                diagnostics=ProductionDiagnostics(
                    latent_chains=LatentChainEvidence(
                        paths=np.stack(
                            tuple(
                                atoms[metadata.distribution][:, span]
                                for span in layout.trajectory_slices.values()
                            ),
                            axis=-1,
                        ).reshape(
                            chains, count // chains, len(layout.time_points), len(layout.constructs)
                        ),
                        free_mask=np.asarray(evidence.latent_free_mask.values),
                        states=tuple(
                            (
                                ConstructRef(id=identity),
                                dynamical_model_spec.get_construct(identity).name,
                            )
                            for identity in layout.constructs
                        ),
                    )
                    if evidence.latent_free_mask is not None
                    else None,
                    mcmc=TrajectoryMCMCResult(
                        chain_samples={
                            name: values.reshape(chains, -1) for name, values in samples.items()
                        },
                        chain_extra_fields=extra,
                        num_chains=chains,
                        num_samples=count // chains,
                        backend=metadata.engine.method,
                    ),
                    observation_log_probs=np.asarray(evidence.observation_log_probs.values)
                    if evidence.observation_log_probs is not None
                    else np.empty((chains, count // chains, 0)),
                    exact_observation_rows=np.asarray(evidence.exact_observation_rows.values)
                    if evidence.exact_observation_rows is not None
                    else None,
                ),
                evidence=metadata.engine,
            )
            diagnostics = (
                metadata.segments[0].diagnostics
                if len(metadata.segments) == 1
                else posterior.get_inference_diagnostics(references)
            )
            traces, ranks = posterior.get_chain_detail(references)
            if evidence.observation_log_probs is not None:
                assert evidence.observed_rows is not None
                loo = posterior.get_loo_diagnostics(
                    observed_rows=np.asarray(evidence.observed_rows.values)
                )

        from grassmoth.study.projections.priors import quantity_prior_densities

        fitted_parameters = frozenset(value.subject.parameter_id for value in marginals)
        priors = quantity_prior_densities(input_model, fitted_parameters)
        layout = layouts[metadata.distribution]
        atoms_value = NumericalArray.from_numpy(atoms[metadata.distribution])
        chain_size = count // metadata.num_chains
        from grassmoth.models.ssm.inference.diagnostics.scorecard import (
            frequency_refresh,
            regional_movement,
        )

        metrics = frozenset(
            metric
            for segment in metadata.segments[-1:]
            if isinstance(segment.settings.algorithm, MarginalParticleGibbsSpec)
            for metric in segment.settings.algorithm.diagnostic_metrics
        )
        labels = {
            identity: dynamical_model_spec.get_construct(identity).name
            for identity in layout.constructs
        }
        return InferenceReport(
            evidence=evidence,
            core=InferenceReportCore(
                inference_metadata=metadata,
                inference_diagnostics=diagnostics,
                convergence=chain_convergence(diagnostics),
                loo_diagnostics=loo[0] if loo is not None else None,
                posterior_marginals=marginals,
                prior_densities=priors,
                frequency_refresh=frequency_refresh(
                    atoms[metadata.distribution], layout, labels, metadata.num_chains
                )
                if "frequency_refresh" in metrics
                else (),
                regional_movement=regional_movement(
                    atoms[metadata.distribution],
                    layout,
                    labels,
                    metadata.num_chains,
                    free_mask=np.asarray(evidence.latent_free_mask.values),
                    final_delta=np.asarray(evidence.final_latent_delta.values)
                    if evidence.final_latent_delta is not None
                    else None,
                    windows=windows,
                    observed_rows=np.asarray(evidence.observed_rows.values),
                )
                if "regional_movement" in metrics
                and evidence.latent_free_mask is not None
                and evidence.observed_rows is not None
                else (),
            ),
            detail=InferenceReportDetail(
                trace_data=tuple(
                    trace.revised(
                        chains=tuple(
                            ArrayVector(
                                array=atoms_value,
                                indices=(None, layout.parameter_columns[trace.subject.element_id]),
                                start=chain * chain_size,
                                stop=(chain + 1) * chain_size,
                            )
                            for chain in range(metadata.num_chains)
                        )
                    )
                    for trace in traces
                ),
                rank_histograms=ranks,
                pareto_k=loo[1] if loo is not None else (),
            ),
        )

    return render()


def fit(
    inputs: FitInputs,
    *,
    execution_context: ModelExecutionContext,
    runtime: FitRuntime,
) -> FitOutput | ModelFitFailure | ContinuationRefusal:
    """Compute a complete fit from owned inputs and explicit execution dependencies."""
    from grassmoth.actions.checks import fit_check_report
    from grassmoth.artifacts.reports.scorecard import MovementWindow

    continuation = resumed_checkpoint(inputs)
    if isinstance(continuation, ContinuationRefusal):
        return continuation
    selection, question, previous = inputs.selection, inputs.question, inputs.previous
    time_origin = inputs.data_for_model.time_origin
    clock = execution_context.clock
    compilation_intervals: list[tuple[str, float, float]] = []
    program_counts: dict[str, int] = {}

    def named_program[Result](function: Callable[..., Result]) -> Callable[..., Result]:
        # Native call sites own descriptive names. This fit owns unique suffixes,
        # passed to the existing compiler dependency as ordinary function names.
        base = cast("FunctionType", function).__name__
        number = program_counts.get(base, 0) + 1
        program_counts[base] = number

        @wraps(function)
        def invoke(*args: object, **kwargs: object) -> Result:
            return function(*args, **kwargs)

        invoke.__name__ = base if number == 1 else f"{base}.{number}"
        return invoke

    def timed_compile[Result](
        function: Callable[..., Result], *args: object, **kwargs: object
    ) -> PreparedProgram[Result]:
        started = clock()
        named = named_program(function)
        program = execution_context.compiler(named, *args, **kwargs)
        compilation_intervals.append((cast("FunctionType", named).__name__, started, clock()))
        return program

    def timed_prepare[Result](
        function: Callable[..., Result], *args: object, **kwargs: object
    ) -> tuple[Callable[[], Compiled], tuple[jax.Array, ...], Result, MemoryDemand]:
        started = clock()
        named = named_program(function)
        compile_program, operands, shape, demand = execution_context.prepare_program(
            named, *args, **kwargs
        )

        def compile_timed() -> Compiled:
            compiled = compile_program()
            compilation_intervals.append((cast("FunctionType", named).__name__, started, clock()))
            return compiled

        return compile_timed, operands, shape, demand

    result = _fit(
        inputs,
        continuation=continuation,
        runtime=runtime,
        execution_context=replace(
            execution_context, compiler=timed_compile, prepare_program=timed_prepare
        )
        if execution_context.profile
        else execution_context,
    )
    if isinstance(result, ModelFitFailure):
        return result
    conditioned = result.dynamical_model_spec
    report = (
        previous.inference.revised(
            evidence=result.evidence,
            core=previous.inference.core.revised(inference_metadata=result.metadata),
        )
        if conditioned is None and previous is not None
        else inference_report(
            conditioned or selection.dynamical_model_spec,
            selection.dynamical_model_spec,
            result.evidence,
            result.metadata,
            tuple(
                MovementWindow(
                    name=name,
                    start_day=query.start_day(time_origin),
                    end_day=query.end_day(time_origin),
                )
                for name, query in question.queries.items()
            ),
        )
    )
    profiling = (
        None
        if result.profile is None
        else result.profile.revised(
            compile_seconds=compilation_seconds(
                tuple(
                    (start, stop)
                    for name, start, stop in compilation_intervals
                    if not name.startswith("profile_")
                )
            ),
            compilation=tuple(
                ProgramCompileReport(name=name, seconds=stop - start)
                for name, start, stop in sorted(compilation_intervals, key=lambda entry: entry[1])
                if not name.startswith("profile_")
            ),
            profiling_compilation=tuple(
                ProgramCompileReport(name=name, seconds=stop - start)
                for name, start, stop in sorted(compilation_intervals, key=lambda entry: entry[1])
                if name.startswith("profile_")
            ),
            device=runtime.devices,
        )
    )
    if profiling is not None:
        log_profile(profiling, execution_context.logger)
    return FitOutput(
        dynamical_model_spec=conditioned
        if conditioned is not None
        else previous.dynamical_model_spec
        if previous is not None
        else None,
        checks=fit_check_report(selection, question, inputs.data_for_model),
        inference=report,
        profile=profiling,
    )
