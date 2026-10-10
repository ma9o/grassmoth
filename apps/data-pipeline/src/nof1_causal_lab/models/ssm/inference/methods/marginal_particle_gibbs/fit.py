"""Marginalized Particle Gibbs inference method."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.compute.budget import ComputeBudget
    from nof1_causal_lab.models.ssm.autoreparam import Strategy
    from nof1_causal_lab.models.ssm.parameterization import PriorRuntimeBundle
    from nof1_causal_lab.models.ssm.runtime import BoundPanel
    from nof1_causal_lab.sampler_config import SamplerSpec

import logging
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.artifacts.posterior_diagnostics import (
    ParticleSamplerDiagnostics,
)
from nof1_causal_lab.models.ssm.inference.methods._pmcmc_shared import (
    build_pmcmc_mcmc_result,
    extract_grouped_public_samples,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.runner import (
    run_marginal_particle_gibbs,
)
from nof1_causal_lab.models.ssm.inference.problem import (
    build_particle_problem,
)
from nof1_causal_lab.models.ssm.inference.types import (
    JointPosteriorDraws,
    ParticleMCMCEvidence,
    ParticleMCMCPosterior,
    ProductionDiagnostics,
)
from nof1_causal_lab.models.ssm.inference.warmup.parameter_warmup import prepare_parameter_warmup
from nof1_causal_lab.models.ssm.transition_kinds import (
    LATENT_TRANSITION_EULER_MARUYAMA,
)

logger = logging.getLogger(__name__)


def fit_marginal_particle_gibbs(
    priors: PriorRuntimeBundle,
    panel: BoundPanel,
    *,
    compute_budget: ComputeBudget,
    sampler: SamplerSpec,
    reparam: Strategy | None = None,
    clock: Callable[[], float],
) -> ParticleMCMCPosterior:
    """Run exact particle inference with resolved choices and owned warmup outputs."""
    observations = panel.observations
    options = sampler.marginal_particle_gibbs
    initial_latent_trajectories = None
    overall_t0 = clock()
    logger.info(
        "marginal_particle_gibbs entry: chains=%d warmup=%d samples=%d T=%d "
        "n_manifest=%d num_particles=%d n_parameter_particles=%d init_method=%s",
        sampler.num_chains,
        sampler.num_warmup,
        sampler.num_samples_per_chain,
        int(observations.shape[0]),
        int(observations.shape[1]) if observations.ndim >= 2 else 0,
        sampler.num_particles,
        options.n_parameter_particles,
        options.init_method,
    )

    base_key = random.PRNGKey(sampler.seed)
    trace_key, pathfinder_key, pf_sample_key = random.split(base_key, 3)

    phase_t0 = clock()
    logger.info("phase 1/4: building marginalized Particle Gibbs runtime bundle...")
    # The model is a continuous-time nonlinear SDE; the particle smoother always
    # discretizes it with the nonlinearity-preserving Euler-Maruyama scheme.
    # Linearised discretisation is confined to the warmup/init backend.
    scheme = LATENT_TRANSITION_EULER_MARUYAMA
    bundle = build_particle_problem(
        priors,
        panel,
        scheme=scheme,
        trace_key=trace_key,
        reparam=reparam,
    )
    logger.info(
        "phase 1/4: bundle ready in %.1fs (dim=%d, public_sites=%d)",
        (clock() - phase_t0),
        int(bundle.runtime.initial_position.shape[0]),
        len(bundle.public_sites),
    )

    phase_t0 = clock()
    warmup_result = prepare_parameter_warmup(
        priors,
        panel,
        compute_budget=compute_budget,
        bundle=bundle.runtime,
        method_label="marginal_particle_gibbs",
        phase_label="phase 2/4",
        trace_key=trace_key,
        pathfinder_key=pathfinder_key,
        sample_key=pf_sample_key,
        reparam=reparam,
        seed=sampler.seed,
        n_ieks_iters=options.n_ieks_iters,
        num_chains=sampler.num_chains,
        init_method=options.init_method,
        initial_positions_override=None,
        parameter_preconditioner_chol=None,
        auto_preconditioner_method=options.auto_preconditioner_method,
        auto_preconditioner_maxiter=options.auto_preconditioner_maxiter,
        pathfinder_num_elbo_samples=options.pathfinder_num_elbo_samples,
        pathfinder_maxiter=options.pathfinder_maxiter,
        n_pathfinder_starts=options.n_pathfinder_starts,
        pathfinder_parallel_workers=options.pathfinder_parallel_workers,
        pathfinder_init_scale=options.pathfinder_init_scale,
        clock=clock,
    )
    init_positions = warmup_result.init_positions
    parameter_preconditioner_chol = warmup_result.preconditioner_chol
    logger.info("phase 2/4: parameter warmup ready in %.1fs", (clock() - phase_t0))

    pilot_means = pilot_vars = pilot_wide_vars = None
    if options.dsmc_leaf_proposal == "paid_mix":
        # The paid mixture leaf needs FIXED per-time pilot moments. The IEKS smoothed
        # paths at the warmup init positions provide them: means from the cross-chain
        # average, a per-coordinate scale from the paths' temporal spread plus the
        # cross-chain disagreement. These are proposal-side quantities of a fixed
        # (chain-independent) mixture component, so approximation cannot bias the
        # sampler — only the exactly-computed mixture density enters the weights —
        # and the init-only linearization policy is satisfied by construction.
        from nof1_causal_lab.models.ssm.inference.warmup.latent_init import (
            compute_ieks_latent_paths,
        )

        pilot_positions = (
            init_positions
            if init_positions is not None
            # Random init leaves positions to the runner; anchor the pilot at the
            # prior center instead (any fixed position yields a valid fixed proposal).
            else jnp.broadcast_to(
                bundle.runtime.initial_position,
                (sampler.num_chains, int(bundle.runtime.initial_position.shape[0])),
            )
        )
        ieks_paths = compute_ieks_latent_paths(
            priors,
            panel,
            positions=pilot_positions,
            trace_key=trace_key,
            reparam=reparam,
            n_ieks_iters=options.n_ieks_iters,
        )
        if bundle.exact_constraints is not None:
            ieks_paths = bundle.exact_constraints.project(ieks_paths)
        initial_latent_trajectories = ieks_paths
        pilot_means = jnp.mean(ieks_paths, axis=0)
        temporal_var = jnp.var(pilot_means, axis=0)
        cross_chain_var = jnp.var(ieks_paths, axis=0)
        var_floor = 1e-6 * (1.0 + temporal_var)
        core_var = temporal_var[None, :] + cross_chain_var + var_floor[None, :]
        pilot_vars = options.paid_mix_pilot_var_scale * core_var
        pilot_wide_vars = options.paid_mix_wide_mult * core_var

    phase_t0 = clock()
    logger.info("phase 3/4: building marginalized Particle Gibbs joint kernel...")
    kernel = build_marginal_particle_gibbs_kernel(
        bundle.runtime,
        exact_constraints=bundle.exact_constraints,
        num_particles=sampler.num_particles,
        num_parameter_particles=options.n_parameter_particles,
        param_step_size=options.param_step_size,
        target_accept=options.param_target_accept,
        min_scale=options.param_step_size_min,
        max_scale=options.param_step_size_max,
        parameter_preconditioner_chol=parameter_preconditioner_chol,
        parameter_proposal=options.parameter_proposal,
        latent_delta=options.latent_delta,
        amala_delta_init=options.amala_delta_init,
        amala_delta_min=options.amala_delta_min,
        amala_delta_max=options.amala_delta_max,
        amala_target_accept=options.amala_target_accept,
        amala_adaptation_window=options.amala_adaptation_window,
        amala_adaptation_tolerance=options.amala_adaptation_tolerance,
        amala_adaptation_rho=options.amala_adaptation_rho,
        amala_adaptation_rho_min=options.amala_adaptation_rho_min,
        amala_adaptation_gamma=options.amala_adaptation_gamma,
        amala_kappa=options.amala_kappa,
        amala_grad_clip=options.amala_grad_clip,
        dsmc_leaf_proposal=options.dsmc_leaf_proposal,
        latent_block_coords=options.latent_block_coords,
        paid_mix_z_weight=options.paid_mix_z_weight,
        paid_mix_pilot_weight=options.paid_mix_pilot_weight,
        pilot_means=pilot_means,
        pilot_vars=pilot_vars,
        pilot_wide_vars=pilot_wide_vars,
        # A provided initialization also supplies a fixed proposal oracle. It
        # stays fixed across chains and never follows the sampled trajectory.
        parameter_reference_path=(
            None
            if initial_latent_trajectories is None
            else jnp.asarray(initial_latent_trajectories)[0]
        ),
        diagnostic_metrics_all=options.diagnostic_metrics_all,
        diagnostic_metrics=options.diagnostic_metrics,
    )
    run_result = run_marginal_particle_gibbs(
        bundle.runtime,
        kernel=kernel,
        num_warmup=sampler.num_warmup,
        num_samples=sampler.num_samples_per_chain,
        num_chains=sampler.num_chains,
        seed=sampler.seed,
        init_scale=options.init_scale,
        adaptation_rate=options.adaptation_rate,
        retain_latent_paths=sampler.retain_latent_paths,
        init_positions=init_positions,
        initial_latent_trajectories=initial_latent_trajectories,
        compute_latent_posterior_summary=options.compute_latent_posterior_summary,
        adaptation_scheme=options.adaptation_scheme,
        clock=clock,
    )
    mcmc_phase_seconds = clock() - phase_t0
    logger.info("phase 3/4: MCMC complete in %.1fs", mcmc_phase_seconds)

    phase_t0 = clock()
    logger.info("phase 4/4: extracting public posterior samples...")
    grouped_public_samples = extract_grouped_public_samples(
        run_result.grouped_positions,
        bundle=bundle,
        num_chains=sampler.num_chains,
        num_samples=sampler.num_samples_per_chain,
    )
    mcmc = build_pmcmc_mcmc_result(
        chain_samples=grouped_public_samples,
        chain_extra_fields=run_result.chain_extra_fields,
        num_chains=sampler.num_chains,
        num_samples=sampler.num_samples_per_chain,
        backend="marginal_particle_gibbs",
    )
    chain_extra_fields = run_result.chain_extra_fields
    summary_extra_fields = (
        chain_extra_fields
        if sampler.num_samples_per_chain > 0
        else run_result.warmup_chain_extra_fields
    )
    diagnostic_summary_phase = "post_warmup" if sampler.num_samples_per_chain > 0 else "warmup"
    kernel_diagnostics = ParticleSamplerDiagnostics(
        settings=sampler,
        parameter_kernel="m_pgibbs_random_walk"
        if options.parameter_proposal == "random_walk"
        else "m_pgibbs_pseudo_langevin",
        mcmc_phase_seconds=float(mcmc_phase_seconds),
        dsmc_leaf_proposal=kernel.dsmc_leaf_proposal,
        diagnostic_metrics=tuple(sorted(kernel.diagnostic_metrics)),
        param_target_accept=float(kernel.target_accept),
        parameter_preconditioned=bool(kernel.preconditioned),
        diagnostic_summary_phase=diagnostic_summary_phase,
        parameter_accept_rate=float(jnp.mean(summary_extra_fields["parameter_accept_prob"])),
        latent_update_fraction=float(jnp.mean(summary_extra_fields["latent_accept_prob"])),
        latent_frozen_fraction=float(jnp.mean(summary_extra_fields["latent_frozen_frac"])),
        latent_block_coords=kernel.latent_block_coords,
        initial_param_step_size=jax.device_get(run_result.initial_param_step_size).tolist(),
        final_param_step_size=jax.device_get(run_result.final_param_step_size).tolist(),
        chain_post_warmup_complete_log_posterior_mean=jax.device_get(
            run_result.post_warmup_complete_log_posterior_mean
        ).tolist(),
        parameter_warmup=warmup_result.warmup_diagnostics,
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
        compiled_step=run_result.compiled_step,
        mcmc=mcmc,
        public_sites=tuple(sorted(bundle.public_sites)),
        observation_log_probs=run_result.observation_log_probs,
        marginal_particle_gibbs=kernel_diagnostics,
        marginal_particle_gibbs_phase_extra_fields={
            "warmup": run_result.warmup_chain_extra_fields,
            "post_warmup": run_result.chain_extra_fields,
            "all": run_result.all_chain_extra_fields,
        },
        warmup_complete_log_posterior_history=run_result.warmup_complete_log_posterior_history,
        all_complete_log_posterior_history=run_result.all_complete_log_posterior_history,
        latent_posterior_summary=run_result.latent_posterior_summary,
        exact_observation_rows=jnp.any(~bundle.exact_constraints.free_mask, axis=-1)
        if bundle.exact_constraints is not None
        else None,
        warmup_latent_paths=run_result.warmup_latent_paths,
        all_latent_paths=run_result.all_latent_paths,
    )
    logger.info(
        "phase 4/4: posterior extraction complete in %.1fs. marginal_particle_gibbs total: %.1fs",
        (clock() - phase_t0),
        (clock() - overall_t0),
    )
    latent_paths = run_result.latent_paths
    return ParticleMCMCPosterior.from_run(
        draws=JointPosteriorDraws(
            parameters=mcmc.get_samples(),
            latent_paths=latent_paths.reshape((-1, *latent_paths.shape[2:]))
            if latent_paths is not None
            else None,
        ),
        diagnostics=diagnostics,
        initial_latent_delta=run_result.initial_latent_delta,
        final_latent_delta=run_result.final_latent_delta,
        evidence=ParticleMCMCEvidence(),
    )
