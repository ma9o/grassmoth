"""Explicit nonlinear replication and measurements from current model uncertainty."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

import numpy as np

from nof1_causal_lab.artifacts.arrays import NumericalArray
from nof1_causal_lab.artifacts.simulation import (
    FitReliability,
    PairedArmSimulation,
    SimulationArm,
    SimulationEvidence,
    SimulationObservationLayout,
    SimulationReport,
    SingleArmSimulation,
)
from nof1_causal_lab.compute.resources import read_compute_budget
from nof1_causal_lab.models.ssm import compiled_access as numeric
from nof1_causal_lab.models.ssm.predictive.simulation import (
    generate_simulation_batch,
    measure_simulation_batch,
)
from nof1_causal_lab.models.ssm.preparation.preflight import ObservationPreflightFailure

if TYPE_CHECKING:
    import dynestyx as dsx
    from jax import Array

    from nof1_causal_lab.actions.contracts.bodies import SimulateInput
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from nof1_causal_lab.models.structure import StructuralSelection
    from nof1_causal_lab.study.persistence.store import ArtifactStore


def simulate(
    selection: StructuralSelection,
    request: SimulateInput[GitOid],
    *,
    store: ArtifactStore,
) -> SimulationEvidence | ObservationPreflightFailure:
    """Generate current model histories; data_diff compares the saved observations separately."""
    from nof1_causal_lab.bootstrap import initialize_numerical_runtime

    initialize_numerical_runtime()
    from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model

    design = request.simulation
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
        compute_budget=read_compute_budget(),
        start=start,
        end=end,
        assignments=assignments,
        time_origin=time_origin,
        seed=request.seed,
        draws=request.num_replicates,
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
        from nof1_causal_lab.actions.simulate.effects import summarize_causal_simulation

        reference = SimulationArm(
            latent_paths=NumericalArray.from_numpy(np.asarray(prediction.reference.latents)),
            observations=NumericalArray.from_numpy(np.asarray(prediction.reference.observations)),
        )
        arms = PairedArmSimulation(
            action=action,
            reference=reference,
            causal=summarize_causal_simulation(
                selection,
                request,
                store=store,
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


def read_simulation_report(
    store: ArtifactStore,
    evidence: SimulationEvidence,
    request: SimulateInput[GitOid],
    question_revision: GitOid,
) -> SimulationReport:
    """Evaluate this simulation's findings from its already generated histories."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from nof1_causal_lab.actions.simulate.summaries import simulation_summary
    from nof1_causal_lab.artifacts.reports.predictive_provenance import (
        FittedLawProvenance,
        MixedLawProvenance,
    )
    from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model
    from nof1_causal_lab.models.ssm.diagnostics.simulation import DesignInfo
    from nof1_causal_lab.models.ssm.inference.diagnostics.convergence import convergence_failures
    from nof1_causal_lab.models.ssm.predictive.registry_runtime import _predictive_models
    from nof1_causal_lab.models.ssm.predictive.simulation import SimulationBatch
    from nof1_causal_lab.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory
    from nof1_causal_lab.models.ssm.preparation.observations import recorded_observation_support
    from nof1_causal_lab.models.structure import StructuralSelection
    from nof1_causal_lab.study.persistence.history import StudyRepository
    from nof1_causal_lab.study.persistence.store import read_model, read_question
    from nof1_causal_lab.study.queries.lineage import fitted_law_report, law_provenance

    def render() -> SimulationReport:
        dynamical_model_spec = read_model(store, request.dynamical_model_spec_ref)
        records = StudyRepository(store.workspace_id).attempts()
        law = law_provenance(
            store,
            store.read_meta("model", request.dynamical_model_spec_ref),
            dynamical_model_spec,
            None,
        )
        reliability: FitReliability = "unknown" if law.kind == "unknown" else "not_fitted"
        if isinstance(law, (FittedLawProvenance, MixedLawProvenance)):
            report = fitted_law_report(store, records, law.fitted_model_revision)
            reliability = (
                "unknown"
                if report.inference_diagnostics is None
                else "unconverged"
                if convergence_failures(report.convergence)
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
        selection = StructuralSelection.for_question(
            dynamical_model_spec, read_question(store, question_revision)
        )
        compiled_dynamical_model = compile_executable_model(selection)
        times = jnp.asarray(evidence.times)
        parameters = {
            name: jnp.asarray(ref.values) for name, ref in evidence.parameter_draws.items()
        }
        support = recorded_observation_support(
            np.asarray(times),
            evidence.observation_layout.variables,
            evidence.observation_layout.support_start_times.values,
            evidence.observation_layout.support_end_times.values,
        )
        if isinstance(support, ObservationPreflightFailure):
            raise ValueError(f"Recorded simulation support is corrupt: {support.message}")
        from nof1_causal_lab.models.ssm.execution.observation_operator import (
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
        findings, _ = measure_simulation_batch(
            compiled_dynamical_model, batch, clock=time.monotonic
        )
        return SimulationReport(
            evidence=evidence,
            summary=summary,
            law=law,
            findings=findings,
            fit_reliability=reliability,
        )

    return render()
