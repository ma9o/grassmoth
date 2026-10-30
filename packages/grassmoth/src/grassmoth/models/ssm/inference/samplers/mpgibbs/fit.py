"""Marginalized Particle Gibbs inference method."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.continuation import (
    ParticleTuning,
    SamplerCheckpoint,
)
from grassmoth.artifacts.identity import ConstructRef
from grassmoth.artifacts.reports.posterior_diagnostics import (
    MCMCEvidence,
    ParticleSamplerDiagnostics,
)
from grassmoth.artifacts.sampling import (
    DsmcSpec,
    GaussianReferenceTemperingSpec,
    MarginalParticleGibbsSpec,
)
from grassmoth.compute.programs import scalar_operand
from grassmoth.models.ssm.inference.continuation import (
    frozen_particle_tuning,
)
from grassmoth.models.ssm.inference.coordinate_structure import compile_translation_structure
from grassmoth.models.ssm.inference.diagnostics.latent import LatentChainEvidence
from grassmoth.models.ssm.inference.extraction import (
    build_pmcmc_mcmc_result,
    extract_grouped_public_samples,
    project_initialization,
)
from grassmoth.models.ssm.inference.problem import build_particle_problem
from grassmoth.models.ssm.inference.reparameterization import AutoReparam
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    MarginalParticleGibbsKernel,
    build_marginal_particle_gibbs_kernel,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import (
    run_marginal_particle_gibbs,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.tempering import (
    ReplicaExchangeKernel,
)
from grassmoth.models.ssm.inference.samplers.particle.contract import PaidMixture
from grassmoth.models.ssm.inference.samplers.particle.weights import summarize_particle_weights
from grassmoth.models.ssm.inference.samplers.progress import inference_phase
from grassmoth.models.ssm.inference.samplers.sweeps import retained_convergence_check
from grassmoth.models.ssm.inference.targets.coordinates import (
    PathChart,
    ScaleChart,
    TranslationChart,
)
from grassmoth.models.ssm.inference.types import (
    JointPosteriorDraws,
    MCMCPosterior,
    ProductionDiagnostics,
    UnretainedRun,
)
from grassmoth.models.ssm.inference.warmup.gaussian_reference import build_gaussian_reference
from grassmoth.models.ssm.inference.warmup.initialization import (
    continued_particle_initialization,
    prepare_particle_initialization,
)
from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure

if TYPE_CHECKING:
    from grassmoth.artifacts.reports.posterior_diagnostics import DiagnosticPhase
    from grassmoth.artifacts.sampling import SamplerSpec
    from grassmoth.compute.context import ModelExecutionContext
    from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
    from grassmoth.models.ssm.parameterization import PriorRuntimeBundle
    from grassmoth.models.ssm.preparation.binding import BoundPanel


def fit_marginal_particle_gibbs(
    priors: PriorRuntimeBundle,
    panel: BoundPanel,
    *,
    sampler: SamplerSpec,
    execution_context: ModelExecutionContext,
    continuation: SamplerCheckpoint | None = None,
) -> MCMCPosterior | UnretainedRun | InitializationFailure:
    """Run exact particle inference with resolved choices and owned warmup outputs."""
    compiler = execution_context.compiler
    clock = execution_context.clock
    logger = execution_context.logger
    observations = panel.observations
    options = sampler.algorithm
    assert isinstance(options, MarginalParticleGibbsSpec)
    overall_t0 = clock()
    logger.info(
        "marginal_particle_gibbs entry: chains=%d warmup=%d samples=%d T=%d "
        "n_manifest=%d num_particles=%d n_parameter_particles=%d",
        sampler.num_chains,
        sampler.num_warmup,
        sampler.num_samples_per_chain,
        int(observations.shape[0]),
        int(observations.shape[1]) if observations.ndim >= 2 else 0,
        options.path_update.num_particles,
        options.n_parameter_particles,
    )

    base_key = random.key(sampler.seed)
    trace_key, initialization_key, sampling_key = random.split(base_key, 3)

    phase_t0 = clock()
    logger.info("phase 1/4: building marginalized Particle Gibbs runtime bundle...")
    reparam = AutoReparam(centered=0.0)
    with inference_phase("mpgibbs", "building_target", execution_context=execution_context):
        bundle = build_particle_problem(
            priors,
            panel,
            trace_key=trace_key,
            # Fully non-centred; a strategy caches per-site choices, so each fit owns one.
            reparam=reparam,
            compiler=compiler,
        )
    logger.info(
        "phase 1/4: bundle ready in %.1fs (dim=%d, public_sites=%d)",
        (clock() - phase_t0),
        int(bundle.runtime.initial_position.shape[0]),
        len(bundle.public_sites),
    )

    phase_t0 = clock()
    logger.info("phase 2/4: Gaussian particle initialization")
    if continuation is not None:
        initialization = continued_particle_initialization(continuation)
    else:
        with inference_phase(
            "mpgibbs", "preparing_initialization", execution_context=execution_context
        ):
            initialization = prepare_particle_initialization(
                bundle.runtime,
                spec=options.initialization,
                key=initialization_key,
                num_chains=sampler.num_chains,
                execution_context=execution_context,
            )
    if isinstance(initialization, InitializationFailure):
        return initialization
    translation: TranslationChart | None = None
    translation_pairs = None
    if options.compiler_reparameterization or options.learned_translation:
        structure = compile_translation_structure(
            panel.compiled_dynamical_model,
            priors,
            bundle.runtime.parameters,
            site_info=bundle.site_info,
            reparam=reparam,
            compiler=compiler,
        )
        translation = TranslationChart(
            structure.initial_slopes
            if options.compiler_reparameterization
            else jnp.zeros_like(structure.initial_slopes),
            jnp.mean(initialization.positions, axis=0),
        )
        translation_pairs = structure.active_pairs
    chart: PathChart | None = translation
    if options.scale_chart:
        reference = jnp.mean(initialization.positions, axis=0)
        paths = jnp.asarray(initialization.paths)
        if translation is None:
            translation = TranslationChart(
                jnp.zeros((paths.shape[-1], reference.size), paths.dtype), reference
            )
        # Warmup windows replace this initial center with mean translated paths.
        center = jnp.mean(
            paths - jax.vmap(translation.offset)(initialization.positions)[:, None, :], axis=0
        )
        chart = ScaleChart(translation, bundle.runtime.diffusion_scale_exponents(reference), center)
    paid_mix = (
        PaidMixture(
            initialization.pilot_mean,
            initialization.pilot_proposal_variance,
            scalar_operand(value=options.path_update.pilot_mixture),
        )
        if isinstance(options.path_update, DsmcSpec)
        else None
    )

    phase_t0 = clock()
    logger.info("phase 3/4: building marginalized Particle Gibbs joint kernel...")

    def make_kernel(runtime: ParticleTarget) -> MarginalParticleGibbsKernel:
        return build_marginal_particle_gibbs_kernel(
            runtime,
            exact_constraints=bundle.exact_constraints,
            num_parameter_particles=options.n_parameter_particles,
            target_accept=options.param_target_accept,
            parameter_preconditioner_chol=initialization.parameter_chol,
            parameter_proposal=options.parameter_proposal,
            paid_mix=paid_mix,
            parameter_reference_path=jax.device_put(
                continuation.tuning.parameter_reference_path.values
            )
            if continuation is not None and isinstance(continuation.tuning, ParticleTuning)
            else initialization.paths[0],
            diagnostic_metrics=options.diagnostic_metrics,
            path_update=options.path_update,
            initial_chart=chart,
            learn_translation=options.learned_translation,
            translation_pairs=translation_pairs,
        )

    kernel = make_kernel(bundle.runtime)
    tempering = None
    reference = None
    if isinstance(options.tempering, GaussianReferenceTemperingSpec):
        ladder = options.tempering
        if (
            continuation is not None
            and isinstance(continuation.tuning, ParticleTuning)
            and continuation.tuning.gaussian_reference
        ):
            from grassmoth.models.ssm.inference.targets.gaussian_reference import (
                GaussianReference,
            )

            reference = GaussianReference(
                *(jnp.asarray(value.values) for value in continuation.tuning.gaussian_reference)
            )
        else:
            reference = build_gaussian_reference(
                bundle.runtime,
                positions=initialization.positions,
                paths=initialization.paths,
                parameter_chol=initialization.parameter_chol,
                scale=ladder.reference_scale,
            )
        tempering = ReplicaExchangeKernel(bundle.runtime, reference, make_kernel, ladder)
    run_result = run_marginal_particle_gibbs(
        bundle.runtime,
        kernel=kernel,
        sampler=sampler,
        key=sampling_key,
        initialization=initialization,
        execution_context=execution_context,
        convergence=retained_convergence_check(
            sampler.stop_when_converged,
            bundle=bundle,
            compiled_dynamical_model=panel.compiled_dynamical_model,
            free_mask=bundle.runtime.state_density_mask
            if bundle.exact_constraints is None
            else bundle.runtime.state_density_mask & bundle.exact_constraints.free_mask,
            compiler=compiler,
        ),
        adaptation_scheme=options.adaptation_scheme,
        tempering=tempering,
        continuation=continuation,
    )
    final_chart = run_result.final_chart
    if final_chart is not None:
        translation = (
            final_chart if isinstance(final_chart, TranslationChart) else final_chart.translation
        )
        logger.info(
            "frozen translation chart (sampler coordinates): slopes=%s reference=%s",
            np.asarray(translation.slopes).tolist(),
            np.asarray(translation.reference).tolist(),
        )
        if isinstance(final_chart, ScaleChart):
            center = np.asarray(final_chart.center)
            logger.info(
                "frozen scale chart (sampler coordinates): exponents=%s center min=%s mean=%s max=%s",
                np.asarray(final_chart.exponents).tolist(),
                center.min(axis=0).tolist(),
                center.mean(axis=0).tolist(),
                center.max(axis=0).tolist(),
            )
    tuning = (
        continuation.tuning
        if continuation is not None and isinstance(continuation.tuning, ParticleTuning)
        else frozen_particle_tuning(initialization, reference)
    ).revised(parameter_chol=NumericalArray.from_numpy(np.asarray(run_result.final_parameter_chol)))
    checkpoint = SamplerCheckpoint(
        settings=sampler,
        native_state=run_result.final_state,
        tuning=tuning,
        random_key=NumericalArray.from_numpy(
            np.asarray(jax.random.key_data(run_result.random_key))
        ),
        parameters=NumericalArray.from_numpy(np.asarray(run_result.last_parameters)),
        paths=NumericalArray.from_numpy(np.asarray(run_result.last_paths)),
    )
    starting_values = project_initialization(
        run_result.starting_values,
        bundle=bundle,
        compiled_dynamical_model=panel.compiled_dynamical_model,
        compiler=compiler,
    )
    if sampler.num_samples_per_chain == 0:
        return UnretainedRun.completed(
            sampler,
            run_result.compiled_step,
            {
                "initialization" if continuation is None else "continued_start": starting_values,
                "warmup": run_result.warmup_chain_extra_fields,
                "post_warmup": run_result.chain_extra_fields,
            },
            profile=run_result.profile,
        )
    mcmc_phase_seconds = clock() - phase_t0
    logger.info("phase 3/4: MCMC complete in %.1fs", mcmc_phase_seconds)

    phase_t0 = clock()
    logger.info("phase 4/4: extracting public posterior samples...")
    grouped_public_samples = extract_grouped_public_samples(
        run_result.grouped_positions,
        bundle=bundle,
        num_chains=sampler.num_chains,
        num_samples=run_result.retained_draws,
        compiler=compiler,
    )
    mcmc = build_pmcmc_mcmc_result(
        chain_samples=grouped_public_samples,
        chain_extra_fields=run_result.chain_extra_fields,
        num_chains=sampler.num_chains,
        num_samples=run_result.retained_draws,
        backend="marginal_particle_gibbs",
    )
    chain_extra_fields = run_result.chain_extra_fields
    summary_extra_fields = (
        chain_extra_fields
        if sampler.num_samples_per_chain > 0
        else run_result.warmup_chain_extra_fields
    )
    diagnostic_summary_phase: DiagnosticPhase = (
        "post_warmup" if sampler.num_samples_per_chain > 0 else "warmup"
    )
    kernel_diagnostics = ParticleSamplerDiagnostics(
        settings=sampler,
        parameter_kernel="m_pgibbs_random_walk"
        if options.parameter_proposal == "random_walk"
        else "m_pgibbs_pseudo_langevin",
        mcmc_phase_seconds=float(mcmc_phase_seconds),
        drift=kernel.path_update.drift
        if isinstance(kernel.path_update, DsmcSpec)
        else ("observation" if kernel.path_update.kappa else "none"),
        diagnostic_metrics=kernel.diagnostic_metrics,
        param_target_accept=float(kernel.target_accept[0]),
        parameter_preconditioned=bool(kernel.preconditioned),
        diagnostic_summary_phase=diagnostic_summary_phase,
        parameter_accept_rate=float(jnp.mean(summary_extra_fields["parameter_accept_prob"])),
        latent_update_fraction=float(jnp.mean(summary_extra_fields["latent_accept_prob"])),
        latent_frozen_fraction=float(jnp.mean(summary_extra_fields["latent_frozen_frac"])),
        latent_block_coords=kernel.path_update.latent_block_coords
        if isinstance(kernel.path_update, DsmcSpec)
        else "all",
        initial_param_step_size=jax.device_get(run_result.initial_param_step_size).tolist(),
        final_param_step_size=jax.device_get(run_result.final_param_step_size).tolist(),
        chain_post_warmup_complete_log_posterior_mean=jax.device_get(
            run_result.post_warmup_complete_log_posterior_mean
        ).tolist(),
        initialization=initialization.diagnostics,
        path_weights=summarize_particle_weights(
            summary_extra_fields,
            num_steps=len(bundle.runtime.times),
            num_particles=options.path_update.num_particles,
            phase=diagnostic_summary_phase,
            sequential=not isinstance(options.path_update, DsmcSpec),
        )
        if "particle_weights" in kernel.diagnostic_metrics
        else None,
        tempering=run_result.tempering,
        latent_move_rms_mean=float(jnp.mean(summary_extra_fields["latent_move_rms"]))
        if "latent_move_rms" in summary_extra_fields
        else None,
        parameter_jump_rms_mean=float(jnp.mean(summary_extra_fields["parameter_jump_rms"]))
        if "parameter_jump_rms" in summary_extra_fields
        else None,
        reference_path_hit_rate_mean=float(
            jnp.mean(summary_extra_fields["reference_path_hit_rate"])
        )
        if "reference_path_hit_rate" in summary_extra_fields
        else None,
        selected_particle_unique_count_mean=float(
            jnp.mean(summary_extra_fields["selected_particle_unique_count"])
        )
        if "selected_particle_unique_count" in summary_extra_fields
        else None,
        amala_grad_norm_mean=float(jnp.mean(summary_extra_fields["amala_grad_norm_mean"])),
        amala_grad_norm_max=float(jnp.max(summary_extra_fields["amala_grad_norm_max"])),
    )
    diagnostics = ProductionDiagnostics(
        profile=run_result.profile,
        stopping_checkpoint=run_result.stopping_checkpoint,
        checkpoint=checkpoint,
        compiled_step=run_result.compiled_step,
        mcmc=mcmc,
        public_sites=tuple(sorted(bundle.public_sites)),
        observation_log_probs=run_result.observation_log_probs,
        latent_chains=LatentChainEvidence(
            paths=np.asarray(run_result.latent_paths),
            free_mask=np.asarray(
                bundle.runtime.state_density_mask
                if bundle.exact_constraints is None
                else bundle.runtime.state_density_mask & bundle.exact_constraints.free_mask
            ),
            states=tuple(
                (ConstructRef(id=s.id), s.name) for s in panel.compiled_dynamical_model.states
            ),
        ),
        sampler_diagnostics=kernel_diagnostics,
        phase_extra_fields={
            "initialization" if continuation is None else "continued_start": starting_values,
            "warmup": run_result.warmup_chain_extra_fields,
            "post_warmup": run_result.chain_extra_fields,
            "all": run_result.all_chain_extra_fields,
        },
        warmup_complete_log_posterior_history=run_result.warmup_complete_log_posterior_history,
        all_complete_log_posterior_history=run_result.all_complete_log_posterior_history,
        exact_observation_rows=jnp.any(
            ~bundle.exact_constraints.free_mask & bundle.runtime.state_density_mask, axis=-1
        )
        if bundle.exact_constraints is not None
        else None,
    )
    logger.info(
        "phase 4/4: posterior extraction complete in %.1fs. marginal_particle_gibbs total: %.1fs",
        (clock() - phase_t0),
        (clock() - overall_t0),
    )
    latent_paths = run_result.latent_paths
    return MCMCPosterior.from_run(
        draws=JointPosteriorDraws(
            parameters=mcmc.get_samples(),
            latent_paths=latent_paths.reshape((-1, *latent_paths.shape[2:])),
        ),
        diagnostics=diagnostics,
        initial_latent_delta=run_result.initial_latent_delta,
        final_latent_delta=run_result.final_latent_delta,
        evidence=MCMCEvidence(),
    )
