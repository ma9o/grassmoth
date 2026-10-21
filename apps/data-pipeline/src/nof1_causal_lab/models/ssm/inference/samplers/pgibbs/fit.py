"""Production Particle Gibbs with an exact conditional parameter kernel and dSMC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.arrays import NumericalArray
from nof1_causal_lab.artifacts.continuation import (
    ParticleTuning,
    SamplerCheckpoint,
)
from nof1_causal_lab.artifacts.identity import ConstructRef
from nof1_causal_lab.artifacts.reports.posterior_diagnostics import (
    ConditionalMetropolisDiagnostics,
    ConditionalNutsDiagnostics,
    FixedParameterDiagnostics,
    MCMCEvidence,
    ParticleGibbsDiagnostics,
)
from nof1_causal_lab.artifacts.sampling import MetropolisHastingsSpec, ParticleGibbsSpec
from nof1_causal_lab.models.ssm.inference.continuation import (
    frozen_particle_tuning,
)
from nof1_causal_lab.models.ssm.inference.diagnostics.latent import LatentChainEvidence
from nof1_causal_lab.models.ssm.inference.extraction import (
    extract_grouped_public_samples,
    project_initialization,
)
from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
from nof1_causal_lab.models.ssm.inference.problem import build_particle_problem
from nof1_causal_lab.models.ssm.inference.samplers.particle.contract import PaidMixture
from nof1_causal_lab.models.ssm.inference.samplers.particle.weights import summarize_dsmc_weights
from nof1_causal_lab.models.ssm.inference.samplers.pgibbs.kernel import build_particle_gibbs_kernel
from nof1_causal_lab.models.ssm.inference.samplers.pgibbs.parameters import (
    ParameterKernel,
    fixed_parameter_kernel,
    metropolis_kernel,
    nuts_kernel,
)
from nof1_causal_lab.models.ssm.inference.samplers.pgibbs.runner import (
    ParticleGibbsRun,
    run_particle_gibbs,
)
from nof1_causal_lab.models.ssm.inference.samplers.progress import inference_phase
from nof1_causal_lab.models.ssm.inference.types import (
    JointPosteriorDraws,
    MCMCPosterior,
    ProductionDiagnostics,
    UnretainedRun,
)
from nof1_causal_lab.models.ssm.inference.warmup.initialization import (
    continued_particle_initialization,
    prepare_particle_initialization,
)
from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure
from nof1_causal_lab.models.ssm.transition_kinds import LATENT_TRANSITION_EULER_MARUYAMA

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable
    from datetime import datetime

    from nof1_causal_lab.artifacts.sampling import SamplerSpec
    from nof1_causal_lab.compute.budget import ComputeBudget
    from nof1_causal_lab.compute.programs import ProgramPreparer
    from nof1_causal_lab.models.ssm.inference.reparameterization import Strategy
    from nof1_causal_lab.models.ssm.parameterization import PriorRuntimeBundle
    from nof1_causal_lab.models.ssm.preparation.binding import BoundPanel


def fit_particle_gibbs(
    priors: PriorRuntimeBundle,
    panel: BoundPanel,
    *,
    prepare_program: ProgramPreparer,
    compute_budget: ComputeBudget,
    sampler: SamplerSpec,
    reparam: Strategy | None,
    clock: Callable[[], float],
    wall_clock: Callable[[], datetime],
    logger: logging.Logger,
    continuation: SamplerCheckpoint | None = None,
) -> MCMCPosterior | UnretainedRun | InitializationFailure:
    """Initialize paired starts, then retain only exact blocked PG updates."""
    options = sampler.algorithm
    assert isinstance(options, ParticleGibbsSpec)
    trace_key, laps_key, sample_key = jax.random.split(jax.random.key(sampler.seed), 3)
    with inference_phase(
        "pgibbs", "building_target", clock=clock, logger=logger, wall_clock=wall_clock
    ):
        problem = build_particle_problem(
            priors,
            panel,
            scheme=LATENT_TRANSITION_EULER_MARUYAMA,
            trace_key=trace_key,
            reparam=reparam,
        )
    target = problem.runtime
    if continuation is not None:
        initialization = continued_particle_initialization(continuation)
    else:
        with inference_phase(
            "pgibbs", "preparing_initialization", clock=clock, logger=logger, wall_clock=wall_clock
        ):
            initialization = prepare_particle_initialization(
                target,
                spec=options.initialization,
                key=laps_key,
                num_chains=sampler.num_chains,
                compute_budget=compute_budget,
            )
    if isinstance(initialization, InitializationFailure):
        return initialization
    paid_mix = (
        None
        if not options.dsmc.pilot_mixture
        else PaidMixture(
            initialization.pilot_mean,
            initialization.pilot_proposal_variance,
        )
    )

    def sample[State](
        parameters: ParameterKernel[State],
    ) -> ParticleGibbsRun | InitializationFailure:
        kernel = build_particle_gibbs_kernel(
            target,
            options.dsmc,
            parameters,
            num_particles=sampler.num_particles,
            num_warmup=sampler.num_warmup,
            paid_mix=paid_mix,
            diagnostic_metrics_all=options.diagnostic_metrics_all,
        )
        return run_particle_gibbs(
            target,
            kernel,
            sampler=sampler,
            initialization=initialization,
            compute_budget=compute_budget,
            key=sample_key,
            clock=clock,
            logger=logger,
            wall_clock=wall_clock,
            continuation=continuation,
            prepare_program=prepare_program,
        )

    started = clock()
    no_parameters = target.initial_position.size == 0
    if no_parameters:
        run = sample(fixed_parameter_kernel())
    elif isinstance(options.parameter_kernel, MetropolisHastingsSpec):
        run = sample(
            metropolis_kernel(
                target, options.parameter_kernel, initialization.parameter_chol, sampler.num_warmup
            )
        )
    else:
        run = sample(
            nuts_kernel(
                target, options.parameter_kernel, initialization.parameter_chol, sampler.num_warmup
            )
        )
    if isinstance(run, InitializationFailure):
        return run
    tuning = (
        continuation.tuning
        if continuation is not None and isinstance(continuation.tuning, ParticleTuning)
        else frozen_particle_tuning(initialization, None)
    )
    checkpoint = SamplerCheckpoint(
        settings=sampler,
        native_state=run.final_state,
        tuning=tuning,
        random_key=NumericalArray.from_numpy(np.asarray(jax.random.key_data(run.random_key))),
        parameters=NumericalArray.from_numpy(np.asarray(run.last_parameters)),
        paths=NumericalArray.from_numpy(np.asarray(run.last_paths)),
    )
    phase_fields = {
        name: project_initialization(
            values, bundle=problem, compiled_dynamical_model=panel.compiled_dynamical_model
        )
        if name in {"initialization", "continued_start"}
        else values
        for name, values in run.phase_extra_fields.items()
    }
    if sampler.num_samples_per_chain == 0:
        return UnretainedRun.completed(
            sampler, MCMCEvidence(method="pgibbs"), run.compiled_step, phase_fields
        )
    duration = clock() - started
    extra = run.phase_extra_fields["post_warmup"]
    parameter_diagnostics: (
        FixedParameterDiagnostics | ConditionalMetropolisDiagnostics | ConditionalNutsDiagnostics
    )
    if no_parameters:
        parameter_diagnostics = FixedParameterDiagnostics()
    elif isinstance(options.parameter_kernel, MetropolisHastingsSpec):
        parameter_diagnostics = ConditionalMetropolisDiagnostics(
            acceptance_rate=float(jnp.mean(extra["parameter_accept_prob"])),
            final_step_size=tuple(float(value) for value in run.final_param_step_size),
        )
    else:
        parameter_diagnostics = ConditionalNutsDiagnostics(
            acceptance_rate=float(jnp.mean(extra["parameter_accept_prob"])),
            final_step_size=tuple(float(value) for value in run.final_param_step_size),
            max_tree_depth_fraction=float(jnp.mean(extra["max_tree_depth_reached"])),
        )
    mcmc = TrajectoryMCMCResult(
        chain_samples=extract_grouped_public_samples(
            run.positions,
            bundle=problem,
            num_chains=sampler.num_chains,
            num_samples=sampler.num_samples_per_chain,
        ),
        chain_extra_fields=extra,
        num_chains=sampler.num_chains,
        num_samples=sampler.num_samples_per_chain,
        backend="pgibbs",
    )
    free_mask = target.state_density_mask
    if target.exact_constraints is not None:
        free_mask = free_mask & target.exact_constraints.free_mask
    return MCMCPosterior.from_run(
        draws=JointPosteriorDraws(
            parameters=mcmc.get_samples(),
            latent_paths=None
            if run.paths is None
            else run.paths.reshape((-1, *run.paths.shape[2:])),
            state_ids=tuple(state.id for state in panel.compiled_dynamical_model.states),
        ),
        diagnostics=ProductionDiagnostics(
            checkpoint=checkpoint,
            mcmc=mcmc,
            observation_log_probs=run.observation_log_probs,
            compiled_step=run.compiled_step,
            public_sites=tuple(sorted(problem.public_sites)),
            latent_chains=None
            if run.paths is None
            else LatentChainEvidence(
                paths=np.asarray(run.paths),
                free_mask=np.asarray(free_mask),
                states=tuple(
                    (ConstructRef(id=s.id), s.name) for s in panel.compiled_dynamical_model.states
                ),
            ),
            sampler_diagnostics=ParticleGibbsDiagnostics(
                settings=sampler,
                mcmc_phase_seconds=duration,
                initialization=initialization.diagnostics,
                parameters=parameter_diagnostics,
                latent_update_fraction=float(jnp.mean(extra["latent_accept_prob"])),
                latent_frozen_fraction=float(jnp.mean(extra["latent_frozen_frac"])),
                latent_move_rms_mean=float(jnp.mean(extra["latent_move_rms"])),
                dsmc_weights=summarize_dsmc_weights(
                    extra,
                    num_steps=len(target.times),
                    num_particles=sampler.num_particles,
                    phase="post_warmup",
                )
                if options.diagnostic_metrics_all
                else None,
            ),
            phase_extra_fields=phase_fields,
            warmup_complete_log_posterior_history=run.log_posterior[:, : sampler.num_warmup],
            all_complete_log_posterior_history=run.log_posterior,
            exact_observation_rows=None
            if target.exact_constraints is None
            else jnp.any(
                ~target.exact_constraints.free_mask & target.state_density_mask,
                axis=-1,
            ),
        ),
        initial_latent_delta=run.initial_latent_delta,
        final_latent_delta=run.final_latent_delta,
        evidence=MCMCEvidence(method="pgibbs"),
    )
