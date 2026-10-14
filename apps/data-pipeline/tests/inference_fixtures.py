"""Committed inference evidence for tests using small explicit numerical values."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.posterior import InferenceReportCore
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.autoreparam import Strategy, _minimal_reparam
from nof1_causal_lab.models.ssm.compile.observation_windows import compile_window_plan
from nof1_causal_lab.models.ssm.inference.shared import assemble_parameter_draws
from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure
from nof1_causal_lab.numpyro_json import empirical_atoms
from nof1_causal_lab.study.records import Applied
from nof1_causal_lab.study.state import ArtifactRecord
from tests.git_fixtures import git_oid

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpyro.primitives import Message

    from nof1_causal_lab.artifacts.dynamical_model_spec import DynamicalModelSpec
    from nof1_causal_lab.artifacts.identity import ArtifactId, ConstructId, GitOid
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledDynamicalModel
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws

_PRIOR_REVISION, _FITTED_REVISION = git_oid(1), git_oid(2)


def inference_log(
    model,
    *,
    revision=_FITTED_REVISION,
    prior_revision=_PRIOR_REVISION,
    seq=3,
    workspace_id="workspace",
):
    pins: dict[ArtifactId, GitOid] = {"model": prior_revision, "panel": git_oid(1)}
    from nof1_causal_lab.study.records import AttemptRecord, FitAttempt, StudyRevision

    record = AttemptRecord(
        seq=seq,
        ts="2026-07-03T00:00:00+00:00",
        attempt=FitAttempt(
            action="fit",
            request=None,
            outcome=Applied(
                result=git_oid(500 + seq),
                effects=ActionEffects(
                    produced=(
                        ArtifactRecord(
                            artifact_id="model",
                            revision=revision,
                            produced_by="fit",
                            derived_from=pins,
                        ),
                    ),
                ),
            ),
        ),
    )
    return StudyRevision(commit_id=git_oid(100 + seq), parent_ids=(), record=record)


def inference_metadata(model):
    """Metadata for an explicit test-owned joint law with one retained chain."""
    from nof1_causal_lab.artifacts.posterior import InferenceMetadata
    from nof1_causal_lab.artifacts.posterior_diagnostics import ParticleMCMCEvidence

    identity = next(iter(model.law_layouts))
    return InferenceMetadata(
        distribution=identity,
        engine=ParticleMCMCEvidence(),
        duration_seconds=0,
        num_samples_total=len(empirical_atoms(model.distributions[identity])),
        num_chains=1,
        sampler_diagnostics=None,
    )


def _report(model):
    from nof1_causal_lab.artifacts.identity import ConstructRef, DistributionId, ParameterRef
    from nof1_causal_lab.artifacts.posterior import (
        InferenceEvidence,
        InferenceMetadata,
        InferenceReport,
        InferenceReportDetail,
    )
    from nof1_causal_lab.artifacts.posterior_diagnostics import (
        ChainDiagnostics,
        LatentDiagnostics,
        ParameterDiagnostics,
        ParticleMCMCEvidence,
    )
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
    from nof1_causal_lab.models.ssm.inference.convergence import chain_convergence

    compiled = compile_model_fixture(model)
    binding = parameter_bindings(compiled)[0][0]
    element, label = next(iter(binding.elements.items()))
    diagnostics = ChainDiagnostics(
        num_chains=4,
        num_samples_per_chain=3,
        per_parameter=(
            ParameterDiagnostics(
                parameter=label,
                subject=ParameterRef(parameter_id=binding.parameter_id, element_id=element),
                r_hat=1.0,
                ess_bulk=800.0,
                ess_tail=600.0,
                mcse_mean=None,
            ),
        ),
        per_latent=tuple(
            LatentDiagnostics(
                subject=ConstructRef(id=s.id),
                label=s.name,
                coordinates=3,
                r_hat=1.0,
                ess_bulk=800.0,
                ess_tail=600.0,
                mcse_mean=None,
            )
            for s in compiled.states
            if not s.is_input
        ),
    )
    return InferenceReport(
        evidence=InferenceEvidence(),
        core=InferenceReportCore(
            inference_metadata=InferenceMetadata(
                distribution=DistributionId("distribution:fixture"),
                num_samples_total=12,
                num_chains=4,
                duration_seconds=0,
                engine=ParticleMCMCEvidence(),
                sampler_diagnostics=None,
            ),
            inference_diagnostics=diagnostics,
            convergence=chain_convergence(diagnostics),
            posterior_marginals=(),
            prior_densities={},
        ),
        detail=InferenceReportDetail(),
    )


def particle_posterior(draws):
    """Publish explicit single-chain test telemetry with fixed supplied draws."""
    from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
    from nof1_causal_lab.models.ssm.inference.types import (
        ParticleMCMCPosterior,
        ProductionDiagnostics,
    )

    chain_samples = {name: values[None, ...] for name, values in draws.parameters.items()}
    n_samples = draws.describe().n_draws
    return ParticleMCMCPosterior.from_run(
        draws=draws,
        diagnostics=ProductionDiagnostics(
            mcmc=TrajectoryMCMCResult(
                chain_samples=chain_samples,
                chain_extra_fields={},
                num_chains=1,
                num_samples=n_samples,
            ),
            observation_log_probs=jnp.zeros(
                (1, n_samples, draws.latent_paths.shape[1] if draws.latent_paths is not None else 0)
            ),
        ),
    )


class MinimalReparam(Strategy):
    """Test-owned minimal reparameterization strategy."""

    @override
    def configure(self, msg: Message):
        return _minimal_reparam(msg["fn"], is_observed=msg.get("is_observed", False))


def make_observation_support_runtime(
    *,
    anchor_times: np.ndarray,
    manifest_names: Sequence[str],
    support_kinds: Sequence[str | None],
    observation_windows: Sequence[str | None],
    support_start_times: np.ndarray,
    support_end_times: np.ndarray,
    summary_operators: Sequence[str | None] | None = None,
    anchor_policies: Sequence[str | None] | None = None,
) -> ObservationSupportRuntime:
    """Compile test-owned window bounds through the production quadrature owner."""
    if summary_operators is None:
        summary_operators = tuple(
            "mean" if kind == "interval" else "last" for kind in support_kinds
        )
    if anchor_policies is None:
        anchor_policies = tuple(
            "support_start" if operator == "first" else "support_end"
            for operator in summary_operators
        )
    plan = compile_window_plan(
        anchor_times, support_start_times, support_end_times, support_kinds, manifest_names
    )
    assert not isinstance(plan, ObservationPreflightFailure), plan
    return ObservationSupportRuntime.assembled(
        window_plan=plan,
        manifest_names=manifest_names,
        support_kinds=support_kinds,
        summary_operators=summary_operators,
        anchor_policies=anchor_policies,
        observation_windows=observation_windows,
        support_start_times=support_start_times,
        support_end_times=support_end_times,
    )


def parameter_draws(
    dynamical_model_spec: DynamicalModelSpec, n_draws: int
) -> dict[str, jnp.ndarray]:
    """Repeat the authored prior reference point without invoking inference."""
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.models.ssm.compile.inputs import compile_priors
    from nof1_causal_lab.prior_distributions import prior_reference_value

    priors, _, _ = compile_priors(
        compile_model_fixture(dynamical_model_spec), StructuralSelection(dynamical_model_spec, None)
    )
    return {
        name: jnp.broadcast_to(value, (n_draws, *value.shape))
        for name, law in priors.items()
        for value in [jnp.asarray(prior_reference_value(law))]
    }


def compile_fit_fixture(
    dynamical_model_spec: DynamicalModelSpec, outcome: ConstructId | None = None
):
    """Require real compilation in fixtures instead of forging fit evidence."""
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.models.ssm.compile.inputs import (
        CompiledFitInputs,
        compile_ssm_inputs_from_model,
    )

    inputs = compile_ssm_inputs_from_model(StructuralSelection(dynamical_model_spec, outcome))
    assert isinstance(inputs, CompiledFitInputs), inputs
    return inputs


def compile_model_fixture(
    dynamical_model_spec: DynamicalModelSpec, outcome: ConstructId | None = None
):
    """Compile native execution facts without imposing the fitting law restrictions."""
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model

    return compile_executable_model(StructuralSelection(dynamical_model_spec, outcome))


def bind_panel_fixture(
    compiled_dynamical_model: CompiledDynamicalModel, observations, times, *, support=None
):
    """Publish a complete numerical test panel, including its identity-bearing rows."""
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from nof1_causal_lab.artifacts.observation_data import ObservationDataset
    from nof1_causal_lab.models.ssm.observation_support import simulation_observation_support
    from nof1_causal_lab.models.ssm.runtime import BoundPanel, bind_panel

    observations, times = jnp.asarray(observations), jnp.asarray(times)
    origin = datetime(1970, 1, 1, tzinfo=UTC)
    support = (
        simulation_observation_support(
            compiled_dynamical_model, np.asarray(times), time_origin=origin
        )
        if support is None
        else support
    )
    rows = []
    for i, observation in enumerate(compiled_dynamical_model.observations):
        for t, at in enumerate(np.asarray(times)):
            start, end = support.support_start_times[t, i], support.support_end_times[t, i]
            rows.append(
                {
                    "indicator_id": str(observation.id),
                    "value": None if np.isnan(observations[t, i]) else float(observations[t, i]),
                    "anchor_time": origin + timedelta(days=float(at)),
                    "support_start": origin + timedelta(days=float(start))
                    if np.isfinite(start)
                    else None,
                    "support_end": origin + timedelta(days=float(end))
                    if np.isfinite(end)
                    else None,
                    "support_kind": support.support_kinds[i],
                    "summary_operator": support.summary_operators[i],
                    "anchor_policy": support.anchor_policies[i],
                    "observation_window": support.observation_windows[i],
                }
            )
    dataset = ObservationDataset.from_frame(
        pl.DataFrame(rows, schema_overrides={"value": pl.Float64}),
        tuple(item.observation for item in compiled_dynamical_model.observations),
        time_origin=origin,
    )
    panel = bind_panel(
        dataset, compiled_dynamical_model=compiled_dynamical_model, time_origin=origin
    )
    assert isinstance(panel, BoundPanel), panel
    return panel


def _scientific_draws(compiled_dynamical_model: CompiledDynamicalModel) -> JointPosteriorDraws:
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws

    if len(compiled_dynamical_model.laws) != 1:
        raise ValueError("Retained particle draws require all random quantities in one joint law")
    law = compiled_dynamical_model.laws[0]
    states = tuple(state.id for state in compiled_dynamical_model.states if not state.is_input)
    if set(law.layout.constructs) != set(states):
        raise ValueError("Retained particle draws require all random quantities in one joint law")
    parameters, paths = law.layout.unpack(jnp.asarray(empirical_atoms(law.distribution)))
    return JointPosteriorDraws(
        parameters=dict(parameters.items()),
        latent_paths=jnp.stack([paths[identity] for identity in law.layout.constructs], axis=-1),
        state_ids=law.layout.constructs,
    )


def model_draws(
    compiled_dynamical_model: CompiledDynamicalModel, *, input_values: jnp.ndarray | None = None
) -> JointPosteriorDraws:
    """Derive native tensors from the current model's aligned particle distribution."""
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws

    retained = _scientific_draws(compiled_dynamical_model)
    samples = assemble_parameter_draws(
        compiled_dynamical_model, retained.parameters, count=retained.describe().n_draws
    )
    paths = retained.latent_paths
    state_ids = [state.id for state in compiled_dynamical_model.states if not state.is_input]
    if paths is not None:
        if set(retained.state_ids) != set(state_ids):
            raise ValueError(
                "Stored trajectories do not match DynamicalModelSpec construct identities"
            )
        paths = paths[..., [retained.state_ids.index(identity) for identity in state_ids]]
        if numeric.input_mask(compiled_dynamical_model).any():
            if input_values is None:
                raise ValueError(
                    "Retained histories with exogenous inputs require the replayed panel path"
                )
            full_ids = numeric.state_ids(compiled_dynamical_model)
            full_paths = jnp.broadcast_to(input_values, (paths.shape[0], *input_values.shape))
            paths = full_paths.at[
                :, :, jnp.asarray([full_ids.index(identity) for identity in state_ids])
            ].set(paths)
            state_ids = full_ids
    return JointPosteriorDraws(parameters=samples, latent_paths=paths, state_ids=tuple(state_ids))
