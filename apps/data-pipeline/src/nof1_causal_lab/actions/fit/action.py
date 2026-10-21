"""Retain posterior atoms, native telemetry, and diagnostics during fit execution."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from nof1_causal_lab.actions.errors import ModelFitError
from nof1_causal_lab.actions.fit.conditioning import condition_model
from nof1_causal_lab.actions.fit.diagnostics import segment_diagnostics
from nof1_causal_lab.actions.fit.runtime import fit_runtime
from nof1_causal_lab.artifacts.arrays import ArrayVector, NumericalArray
from nof1_causal_lab.artifacts.identity import ConstructRef, ParameterRef
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.model.parameter import ParameterCoordinate
from nof1_causal_lab.artifacts.reports.inference import (
    FitSegment,
    InferenceEvidence,
    InferenceMetadata,
    InferenceReport,
    InferenceReportCore,
    InferenceReportDetail,
)
from nof1_causal_lab.models.ssm.inference.diagnostics.convergence import chain_convergence
from nof1_causal_lab.models.ssm.inference.samplers.progress import capture_phases
from nof1_causal_lab.models.ssm.inference.types import UnretainedRun

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from nof1_causal_lab.actions.contracts.bodies import FitOutput
    from nof1_causal_lab.artifacts.continuation import SamplerCheckpoint
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.artifacts.observations.dataset import ObservationDataset
    from nof1_causal_lab.artifacts.probability.serialization import ArrayLoader
    from nof1_causal_lab.artifacts.reports.posterior_diagnostics import ChainDiagnostics
    from nof1_causal_lab.artifacts.sampling import SamplerSpec
    from nof1_causal_lab.models.structure import StructuralSelection
    from nof1_causal_lab.study.persistence.store import ArtifactStore


class FitResult(TypedDict):
    """The fitted model definition paired with its retained numerical inference evidence."""

    _dynamical_model_spec: DynamicalModelSpec | None
    evidence: InferenceEvidence
    metadata: InferenceMetadata


def append_phase_evidence(
    previous: InferenceEvidence, current: InferenceEvidence
) -> InferenceEvidence:
    """Retain earlier phase buffers under one additional continuation prefix."""
    return current.revised(
        phase_extra_fields={
            **{f"previous/{name}": values for name, values in previous.phase_extra_fields.items()},
            **current.phase_extra_fields,
        }
    )


def fit(
    *,
    selection: StructuralSelection,
    data_for_model: ObservationDataset,
    time_origin: datetime,
    sampler: SamplerSpec,
    array_writer: Callable[[np.ndarray], str],
    array_loader: ArrayLoader,
    compute_loo_diagnostics: bool,
    continuation: SamplerCheckpoint | None = None,
    previous: FitOutput | None = None,
) -> FitResult:
    """The exact producer's buffers cross the compute boundary without report reductions."""
    from nof1_causal_lab.actions.fit.execution import fit_model

    with capture_phases(time.monotonic) as phases:
        fitted = fit_model(
            selection,
            data_for_model,
            time_origin=time_origin,
            sampler=sampler,
            continuation=continuation,
        )
    if "error" in fitted:
        raise ModelFitError(
            fitted["error"], diagnostics={"duration_seconds": fitted["duration_seconds"]}
        )
    result = fitted["result"]
    runtime = fit_runtime(sampler)
    old_count = (
        0
        if previous is None
        else previous.inference.core.inference_metadata.num_samples_total // sampler.num_chains
    )

    def segment(diagnostics: ChainDiagnostics | None) -> FitSegment:
        return FitSegment(
            diagnostics=diagnostics,
            convergence=chain_convergence(diagnostics),
            start=old_count,
            stop=old_count + sampler.num_samples_per_chain,
            settings=sampler,
            runtime=runtime,
            phases=tuple(phases),
            parameterization=result.parameterization
            if isinstance(result, UnretainedRun)
            else result.diagnostics.parameterization,
        )

    if isinstance(result, UnretainedRun):
        phase_fields = {
            phase: {
                name: NumericalArray.from_numpy(np.asarray(value)) for name, value in fields.items()
            }
            for phase, fields in result.phase_extra_fields.items()
        }
        if previous is not None:
            prior = previous.inference
            return {
                "_dynamical_model_spec": None,
                "metadata": prior.core.inference_metadata.revised(
                    duration_seconds=fitted["duration_seconds"],
                    segments=(*prior.core.inference_metadata.segments, segment(None)),
                ),
                "evidence": append_phase_evidence(
                    prior.evidence, prior.evidence.revised(phase_extra_fields=phase_fields)
                ),
            }
        return {
            "_dynamical_model_spec": None,
            "metadata": InferenceMetadata(
                distribution=None,
                num_samples_total=0,
                num_chains=sampler.num_chains,
                duration_seconds=fitted["duration_seconds"],
                engine=result.evidence,
                sampler_diagnostics=None,
                segments=(segment(None),),
            ),
            "evidence": InferenceEvidence(phase_extra_fields=phase_fields),
        }
    native = result.diagnostics

    def retain(values: object | None) -> NumericalArray | None:
        return NumericalArray.from_numpy(np.asarray(values)) if values is not None else None

    conditioned, identity = condition_model(
        selection.dynamical_model_spec,
        fitted["panel"].compiled_dynamical_model,
        result,
        times=fitted["panel"].times,
        time_origin=time_origin,
        array_writer=array_writer,
        array_loader=array_loader,
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
            name: NumericalArray.from_numpy(np.asarray(values))
            for name, values in native.mcmc.get_extra_fields(group_by_chain=True).items()
        },
        observation_log_probs=retain(native.observation_log_probs)
        if compute_loo_diagnostics
        else None,
        observed_rows=retain(np.any(np.isfinite(fitted["panel"].observations), axis=1))
        if compute_loo_diagnostics
        else None,
        exact_observation_rows=retain(native.exact_observation_rows),
        phase_extra_fields={
            phase: {
                name: NumericalArray.from_numpy(np.asarray(values))
                for name, values in buffers.items()
            }
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
        duration_seconds=fitted["duration_seconds"],
        num_chains=native.mcmc.num_chains,
        num_samples_total=native.mcmc.num_chains * (old_count + native.mcmc.num_samples),
        sampler_diagnostics=native.sampler_diagnostics,
    )
    if previous is not None:
        from nof1_causal_lab.artifacts.probability.serialization import (
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
                new.reshape(chains, sampler.num_samples_per_chain, -1),
            ),
            axis=1,
        )
        law = empirical_distribution(
            combined.reshape(-1, combined.shape[-1]),
            array_writer=array_writer,
            array_loader=array_loader,
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
        evidence = append_phase_evidence(prior_evidence, evidence)
    return {"_dynamical_model_spec": conditioned, "evidence": evidence, "metadata": metadata}


def read_inference_report(
    store: ArtifactStore,
    revision: GitOid,
    evidence: InferenceEvidence,
    metadata: InferenceMetadata,
) -> InferenceReport:
    """Compute exact-chain diagnostics from this fit's atoms and native telemetry."""
    from nof1_causal_lab.study.persistence.store import read_model

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
        import jax.numpy as jnp

        from nof1_causal_lab.artifacts.probability.serialization import empirical_atoms
        from nof1_causal_lab.models.ssm.inference.diagnostics.latent import LatentChainEvidence
        from nof1_causal_lab.models.ssm.inference.diagnostics.plots import (
            compute_posterior_marginals,
        )
        from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
        from nof1_causal_lab.models.ssm.inference.types import (
            JointPosteriorDraws,
            MCMCPosterior,
            ProductionDiagnostics,
        )

        assert metadata.distribution is not None
        dynamical_model_spec = read_model(store, revision)
        layouts = {metadata.distribution: dynamical_model_spec.law_layouts[metadata.distribution]}
        atoms = {
            identity: empirical_atoms(dynamical_model_spec.distributions[identity])
            for identity in layouts
        }
        count = next(iter(atoms.values())).shape[0]
        samples: dict[str, jnp.ndarray] = {
            element: jnp.asarray(atoms[identity][:, layout.parameter_columns[element]])
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
        extra = {name: jnp.asarray(ref.values) for name, ref in evidence.chain_extra_fields.items()}
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
                    observation_log_probs=jnp.asarray(evidence.observation_log_probs.values)
                    if evidence.observation_log_probs is not None
                    else jnp.empty((chains, count // chains, 0)),
                    exact_observation_rows=jnp.asarray(evidence.exact_observation_rows.values)
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
                    observed_rows=jnp.asarray(evidence.observed_rows.values)
                )

        from nof1_causal_lab.study.projections.priors import quantity_prior_densities

        input_model = read_model(store, store.read_meta("model", revision).derived_from["model"])
        fitted_parameters = frozenset(value.subject.parameter_id for value in marginals)
        priors = quantity_prior_densities(input_model, fitted_parameters)
        layout = layouts[metadata.distribution]
        atoms_value = NumericalArray.from_numpy(atoms[metadata.distribution])
        chain_size = count // metadata.num_chains
        return InferenceReport(
            evidence=evidence,
            core=InferenceReportCore(
                inference_metadata=metadata,
                inference_diagnostics=diagnostics,
                convergence=chain_convergence(diagnostics),
                loo_diagnostics=loo[0] if loo is not None else None,
                posterior_marginals=marginals,
                prior_densities=priors,
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
