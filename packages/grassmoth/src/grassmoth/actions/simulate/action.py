"""Explicit nonlinear replication and measurements from current model uncertainty."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from grassmoth.actions.contracts.bodies import SimulateOutput
from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.simulation import (
    FitReliability,
    PairedArmSimulation,
    SimulationArm,
    SimulationEvidence,
    SimulationObservationLayout,
    SimulationReport,
    SingleArmSimulation,
)
from grassmoth.models.ssm import compiled_access as numeric
from grassmoth.models.ssm.predictive.simulation import (
    generate_simulation_batch,
    measure_simulation_batch,
)
from grassmoth.models.ssm.preparation.preflight import ObservationPreflightFailure

if TYPE_CHECKING:
    import dynestyx as dsx
    from jax import Array

    from grassmoth.artifacts.identity import GitRef
    from grassmoth.artifacts.reports.inference import InferenceReportCore
    from grassmoth.artifacts.reports.predictive_provenance import PredictiveLawProvenance
    from grassmoth.artifacts.simulation import SimulationSpec
    from grassmoth.compute.budget import ComputeBudget
    from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from grassmoth.models.structure import StructuralSelection


def _simulate(
    selection: StructuralSelection,
    design: SimulationSpec,
    *,
    seed: int,
    num_replicates: int,
    budget: ComputeBudget,
    fitted: tuple[GitRef, InferenceReportCore] | None,
) -> SimulationEvidence | ObservationPreflightFailure:
    """Generate current model histories; data_diff compares the saved observations separately."""
    from grassmoth.models.ssm.compile.inputs import compile_executable_model

    compiled_dynamical_model = compile_executable_model(selection)
    time_origin = next(
        (
            law.layout.time_origin
            for law in compiled_dynamical_model.laws
            if law.layout.constructs and not isinstance(law.layout.time_origin, str)
        ),
        design.start_instant,
    )
    start, end = design.start_day(time_origin), design.end_day(time_origin)
    assignments = design.assignments(time_origin)
    batch = generate_simulation_batch(
        compiled_dynamical_model,
        compute_budget=budget,
        start=start,
        end=end,
        assignments=assignments,
        time_origin=time_origin,
        seed=seed,
        draws=num_replicates,
    )
    if isinstance(batch, ObservationPreflightFailure):
        return batch
    support = batch.measurement_design.observation_support
    if support is None:
        raise ValueError("Simulation must retain its observation support")
    prediction = batch.prediction
    state_ids = tuple(numeric.state_ids(compiled_dynamical_model))
    variables = tuple(
        observation.observation for observation in compiled_dynamical_model.observations
    )
    action = SimulationArm(
        latent_paths=NumericalArray.from_numpy(np.asarray(prediction.trajectory.latents)),
        observations=NumericalArray.from_numpy(np.asarray(prediction.trajectory.observations)),
    )
    arms: SingleArmSimulation | PairedArmSimulation = SingleArmSimulation(action=action)
    if prediction.reference is not None:
        from grassmoth.actions.simulate.effects import summarize_causal_simulation

        reference = SimulationArm(
            latent_paths=NumericalArray.from_numpy(np.asarray(prediction.reference.latents)),
            observations=NumericalArray.from_numpy(np.asarray(prediction.reference.observations)),
        )
        arms = PairedArmSimulation(
            action=action,
            reference=reference,
            causal=summarize_causal_simulation(
                selection,
                design,
                fitted=fitted,
                action=action,
                reference=reference,
                state_ids=state_ids,
                indicator_ids=tuple(variable.id for variable in variables),
            ),
        )
    return SimulationEvidence(
        assignments=assignments,
        time_origin=time_origin,
        times=batch.times,
        draws=prediction.n_draws,
        seed=batch.measurement_design.seed,
        state_ids=state_ids,
        parameter_draws={
            name: NumericalArray.from_numpy(np.asarray(value))
            for name, value in prediction.parameters.items()
        },
        arms=arms,
        observation_layout=SimulationObservationLayout(
            variables=variables,
            support_start_times=NumericalArray.from_numpy(np.asarray(support.support_start_times)),
            support_end_times=NumericalArray.from_numpy(np.asarray(support.support_end_times)),
            mask=NumericalArray.from_numpy(np.asarray(prediction.trajectory.observations_mask)),
        ),
    )


def simulation_report(
    selection: StructuralSelection,
    evidence: SimulationEvidence,
    law: PredictiveLawProvenance,
    law_report: InferenceReportCore | None,
) -> SimulationReport:
    """Evaluate this simulation's findings from its already generated histories."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from grassmoth.actions.simulate.summaries import simulation_summary
    from grassmoth.artifacts.reports.predictive_provenance import (
        FittedLawProvenance,
        MixedLawProvenance,
    )
    from grassmoth.models.ssm.compile.inputs import compile_executable_model
    from grassmoth.models.ssm.diagnostics.simulation import DesignInfo
    from grassmoth.models.ssm.inference.diagnostics.convergence import convergence_failures
    from grassmoth.models.ssm.predictive.registry_runtime import _predictive_models
    from grassmoth.models.ssm.predictive.simulation import SimulationBatch
    from grassmoth.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory
    from grassmoth.models.ssm.preparation.observations import recorded_observation_support

    reliability: FitReliability = "unknown" if law.kind == "unknown" else "not_fitted"
    if isinstance(law, (FittedLawProvenance, MixedLawProvenance)):
        assert law_report is not None, "Fitted provenance supplies its retained report"
        reliability = (
            "unknown"
            if law_report.inference_diagnostics is None
            else "unconverged"
            if convergence_failures(law_report.convergence)
            else "converged"
        )
    latents = evidence.arms.action.latent_paths.values
    observations = evidence.arms.action.observations.values
    mask = evidence.observation_layout.mask.values
    reference = (
        evidence.arms.reference.latent_paths.values
        if isinstance(evidence.arms, PairedArmSimulation)
        else None
    )
    reference_observations = (
        evidence.arms.reference.observations.values
        if isinstance(evidence.arms, PairedArmSimulation)
        else None
    )
    for index, variable in enumerate(evidence.observation_layout.variables):
        levels = (
            ("0", "1")
            if variable.measurement_dtype == "binary"
            else variable.ordinal_levels or variable.categorical_levels
        )
        if levels is None:
            continue
        for buffer in (observations, reference_observations):
            if buffer is None:
                continue
            channel = buffer[:, :, index]
            codes = channel[mask[:, :, index] & np.isfinite(channel)]
            if not np.all((codes == np.floor(codes)) & (codes >= 0) & (codes < len(levels))):
                raise ValueError(
                    "Saved simulation emissions differ from their declared category codes"
                )
    summary = simulation_summary(
        evidence, latents, observations, mask, reference, reference_observations
    )
    compiled_dynamical_model = compile_executable_model(selection)
    times = jnp.asarray(evidence.times)
    parameters = {name: jnp.asarray(ref.values) for name, ref in evidence.parameter_draws.items()}
    support = recorded_observation_support(
        np.asarray(times),
        evidence.observation_layout.variables,
        evidence.observation_layout.support_start_times.values,
        evidence.observation_layout.support_end_times.values,
    )
    if isinstance(support, ObservationPreflightFailure):
        raise ValueError(f"Recorded simulation support is corrupt: {support.message}")
    from grassmoth.models.ssm.execution.observation_operator import (
        compile_observation_operator,
    )

    operator = compile_observation_operator(
        support,
        held_channels=tuple(
            index
            for index, observation in enumerate(compiled_dynamical_model.observations)
            if compiled_dynamical_model.states[observation.state_index].is_input
        ),
    )
    native = _predictive_models(compiled_dynamical_model, parameters, times)

    def responses(dynamical_model: dsx.DynamicalModel, path: Array) -> tuple[Array, Array]:
        observation = cast("HeterogeneousObservation", dynamical_model.observation_model)
        predictors = jax.vmap(observation.linear_predictor)(path)
        response = jax.vmap(lambda value: observation.at_predictor(value).response)(predictors)
        response, _ = operator.project_response_trajectory(response)
        return predictors, response

    predictors, means = eqx.filter_vmap(responses)(native, jnp.asarray(latents))
    trajectory = PredictiveTrajectory(
        jnp.asarray(latents),
        predictors,
        jnp.asarray(observations),
        jnp.asarray(mask),
        jnp.where(mask, means, jnp.nan),
    )
    prediction = PredictiveDraws(parameters, trajectory)
    variables = evidence.observation_layout.indicator_ids
    batch = SimulationBatch.from_draws(
        prediction,
        DesignInfo(
            t_grid=times,
            manifest_ids=variables,
            obs_index_by_indicator={
                identity: np.flatnonzero(np.asarray(mask[:, :, index]).any(axis=0))
                for index, identity in enumerate(variables)
            },
            values_by_indicator={identity: np.asarray([]) for identity in variables},
            n_draws=evidence.draws,
            seed=evidence.seed,
            observation_support=support,
        ),
        time_origin=evidence.time_origin,
    )
    findings, _ = measure_simulation_batch(compiled_dynamical_model, batch)
    return SimulationReport(
        evidence=evidence,
        summary=summary,
        law=law,
        findings=findings,
        fit_reliability=reliability,
    )


def simulate(
    selection: StructuralSelection,
    design: SimulationSpec,
    *,
    seed: int,
    num_replicates: int,
    budget: ComputeBudget,
    law: PredictiveLawProvenance,
    law_report: InferenceReportCore | None,
    fitted: tuple[GitRef, InferenceReportCore] | None,
) -> SimulateOutput | ObservationPreflightFailure:
    """Generate and evaluate complete histories using resolved scientific evidence."""
    from grassmoth.study.projections.visuals import simulation_observation_histories

    evidence = _simulate(
        selection, design, seed=seed, num_replicates=num_replicates, budget=budget, fitted=fitted
    )
    if isinstance(evidence, ObservationPreflightFailure):
        return evidence
    return SimulateOutput(
        report=simulation_report(selection, evidence, law, law_report),
        data=simulation_observation_histories(
            evidence,
            evidence.arms.action.observations.values,
            evidence.observation_layout.mask.values,
        ),
    )
