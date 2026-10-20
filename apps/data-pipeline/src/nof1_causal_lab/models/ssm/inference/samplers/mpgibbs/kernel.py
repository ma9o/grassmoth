"""Marginalized Particle Gibbs joint parameter/trajectory kernel.

Implements the M=2-by-default collapsed Particle Gibbs construction from
Corenflos (2025), "Particle Gibbs without the Gibbs bit", for directly
evaluable SSM potentials. The parameter proposal is formed in unconstrained
space via the auxiliary decomposition

    u | theta  ~ N(theta + 2 delta Sigma g(theta), 2 delta Sigma)
    theta' | u ~ N(u, 2 delta Sigma)

where ``g`` is a fixed, parameter-only gradient oracle. The default computes
the complete-data gradient on a fixed pilot path chosen at construction; it
must not use the changing reference trajectory during the chain. ``parameter_proposal``
selects the drift: ``random_walk`` sets ``g = 0`` (the symmetric special case),
while the default ``pseudo_langevin`` (Corenflos 2025, §3.1) drifts the
theta->u half by ``g`` so the two halves reproduce preconditioned MALA. The
resulting asymmetry is corrected exactly in the Barker label-selection weights
(identically zero in the random-walk case). The latent trajectory is updated by
conditional SMC against the posterior mixture over the parameter ensemble.

Step-size convention. ``delta`` here (``param_step_size``) is the variance
coefficient of each half, not the conventional MALA step. The combined kernel
has variance ``4 delta Sigma`` (and drift ``2 delta Sigma g`` for pseudo-Langevin),
so ``param_step_size`` maps to MALA's ``h`` and to the RW variance coefficient
via ``h = 4 * param_step_size`` in both branches. Tune accordingly: a familiar
MALA step of ``h = 0.1`` corresponds to ``param_step_size = 0.025`` here.
"""

# References:
#   https://arxiv.org/abs/2505.04611 — Corenflos (2025), "Particle Gibbs
#     without the Gibbs bit" (arXiv:2505.04611): the collapsed M-ensemble parameter
#     proposal (§3.1 pseudo-Langevin) and Barker label-selection weights live here.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import random

from nof1_causal_lab.artifacts.sampling import DsmcSpec, MarginalParticleGibbsSpec
from nof1_causal_lab.models.ssm.compile.state_space import AdditiveRecurrences, ParallelRecurrences
from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCState, _clip_scale
from nof1_causal_lab.models.ssm.inference.samplers.particle.context import (
    build_smoother_context,
)
from nof1_causal_lab.models.ssm.inference.samplers.particle.diagnostics import (
    build_particle_diagnostic_flags,
    resolve_particle_diagnostic_metrics,
)
from nof1_causal_lab.models.ssm.inference.samplers.particle.dsmc import step as dsmc_step
from nof1_causal_lab.models.ssm.inference.samplers.particle.dsmc import (
    trace_step as dsmc_trace_step,
)
from nof1_causal_lab.models.ssm.inference.samplers.particle.math import (
    _masked_mean,
    _select_pytree,
)
from nof1_causal_lab.models.ssm.inference.samplers.particle.setup import build_smoother_setup
from nof1_causal_lab.models.ssm.inference.tracing import (
    CategoricalTraceRecord,
    MarginalParticleTraceRecord,
    ParameterProposalTraceRecord,
    TracedResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.models.ssm.inference.conditioning import ExactStateConstraints
    from nof1_causal_lab.models.ssm.inference.samplers.particle.contract import (
        PaidMixture,
    )
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget


_DEFAULT_OPTIONS = MarginalParticleGibbsSpec()
type MarginalParticleStepResult = tuple[TrajectoryMCMCState, dict[str, jnp.ndarray]]


class MarginalParticleStep(Protocol):
    """A joint transition with a static dSMC concurrency limit."""

    def __call__(
        self, state: TrajectoryMCMCState, key: jnp.ndarray, *, batch_size: int | None = None
    ) -> MarginalParticleStepResult:
        """Advance once with all nodes parallel unless a batch limit is supplied."""
        ...


class MarginalParticleGibbsKernel(NamedTuple):
    """Callable joint kernel and static metadata."""

    step_fn: MarginalParticleStep
    parameter_gradient_fn: Callable[[jnp.ndarray], jnp.ndarray]
    trace_step_fn: Callable[  # noqa: F841, RUF100 - Notebook consumers are outside Vulture's source-only pass.
        [TrajectoryMCMCState, jnp.ndarray],
        TracedResult[MarginalParticleStepResult, MarginalParticleTraceRecord],
    ]
    num_particles: int
    num_parameter_particles: int
    initial_param_step_size: float
    target_accept: float
    min_scale: float
    max_scale: float
    preconditioned: bool
    dsmc: DsmcSpec
    diagnostic_metrics: frozenset[str]
    exact_constraints: ExactStateConstraints | None


def build_marginal_particle_gibbs_kernel(
    target: ParticleTarget,
    *,
    num_particles: int,
    num_parameter_particles: int,
    param_step_size: float,
    target_accept: float | None = None,
    min_scale: float = _DEFAULT_OPTIONS.param_step_size_min,
    max_scale: float = _DEFAULT_OPTIONS.param_step_size_max,
    parameter_preconditioner_chol: jnp.ndarray | None = None,
    parameter_proposal: Literal[
        "random_walk", "pseudo_langevin"
    ] = _DEFAULT_OPTIONS.parameter_proposal,
    dsmc: DsmcSpec = _DEFAULT_OPTIONS.dsmc,
    paid_mix: PaidMixture | None = None,
    parameter_reference_path: jnp.ndarray | None = None,
    exact_constraints: ExactStateConstraints | None = None,
    diagnostic_metrics_all: bool = _DEFAULT_OPTIONS.diagnostic_metrics_all,
    diagnostic_metrics: tuple[str, ...] | list[str] | None = _DEFAULT_OPTIONS.diagnostic_metrics,
) -> MarginalParticleGibbsKernel:
    """Build a marginalized Particle Gibbs joint state update."""
    resolved_diagnostic_metrics = resolve_particle_diagnostic_metrics(
        diagnostic_metrics_all=diagnostic_metrics_all,
        diagnostic_metrics=diagnostic_metrics,
    )
    diagnostic_flags = build_particle_diagnostic_flags(
        diagnostic_metrics=resolved_diagnostic_metrics,
    )
    if num_particles < 2:
        raise ValueError("marginal_particle_gibbs requires num_particles >= 2.")
    if num_parameter_particles < 2:
        raise ValueError("marginal_particle_gibbs requires num_parameter_particles >= 2.")
    if min_scale > max_scale:
        raise ValueError(
            "marginal_particle_gibbs min_scale must be <= max_scale; "
            f"got {min_scale} > {max_scale}."
        )
    use_gradient_drift = parameter_proposal == "pseudo_langevin"
    if target_accept is None:
        target_accept = _DEFAULT_OPTIONS.param_target_accept

    assert isinstance(target.state_space.recurrence, (AdditiveRecurrences, ParallelRecurrences)), (
        "The fitting preflight must establish an associative deterministic recurrence"
    )
    local_target = target.tree_target()

    runtime_observations = target.observations
    runtime_times = target.times

    fixed_path = (
        target.initial_path(target.context(target.initial_position, runtime_times))
        if parameter_reference_path is None
        else jnp.asarray(parameter_reference_path)
    )
    if exact_constraints is not None:
        fixed_path = exact_constraints.project(fixed_path)
    latent_free_mask = (
        target.independent_mask
        if exact_constraints is None
        else exact_constraints.free_mask & target.independent_mask
    )

    def _theta_logpost_grad(z: jnp.ndarray) -> jnp.ndarray:
        # q(u | theta) uses a theta-only oracle, fixed for every kernel call.
        # Conditioning this oracle on the current path invalidates the label correction.
        if not use_gradient_drift:
            return jnp.zeros_like(z)
        evaluate: Callable[[jnp.ndarray], jnp.ndarray] = jax.grad(
            lambda zz: target.log_posterior(zz, fixed_path, runtime_observations, runtime_times)
        )
        return evaluate(z)

    preconditioner = (
        None
        if parameter_preconditioner_chol is None
        else jnp.asarray(
            parameter_preconditioner_chol,
            dtype=target.initial_position.dtype,
        )
    )

    def _propose_parameter_ensemble(
        current_position: jnp.ndarray,
        current_gradient: jnp.ndarray,
        key: jnp.ndarray,
        step_size: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dim = int(current_position.shape[0])
        step = jnp.asarray(step_size, dtype=current_position.dtype)
        proposal_scale = jnp.sqrt(jnp.asarray(2.0, dtype=current_position.dtype) * step)
        if preconditioner is None:
            chol = jnp.eye(dim, dtype=current_position.dtype)
        else:
            chol = jnp.asarray(preconditioner, dtype=current_position.dtype)
        aux_key, proposal_key = random.split(key)

        # Pseudo-Langevin (Corenflos 2025, §3.1): drift the θ→u half by 2·step·Σ·g(θ),
        # with Σ = chol·cholᵀ and g the fixed parameter-only gradient oracle. The θ'←u
        # half stays a plain Gaussian, so only u carries the drift. Combined, the two
        # halves reproduce preconditioned MALA; random-walk is the zero-drift case.
        if use_gradient_drift:
            with jax.named_scope("propose_grad_drift"):
                grad_ref = current_gradient
                drift = (jnp.asarray(2.0, dtype=step.dtype) * step) * (chol @ (chol.T @ grad_ref))
        else:
            grad_ref = jnp.zeros_like(current_position)
            drift = jnp.zeros_like(current_position)

        u = (
            current_position
            + drift
            + proposal_scale
            * (
                random.normal(aux_key, current_position.shape, dtype=current_position.dtype)
                @ chol.T
            )
        )
        proposal_eps = random.normal(
            proposal_key,
            (num_parameter_particles - 1, dim),
            dtype=current_position.dtype,
        )
        proposed = u[None, :] + proposal_scale * (proposal_eps @ chol.T)
        ensemble = jnp.concatenate([current_position[None, :], proposed], axis=0)

        # Barker label-prior correction Δ_l = log q(u|θˡ) − log q(θˡ|u). With the drift
        # only in q(u|·), Δ_l = g(θˡ)·(u−θˡ) − step·‖cholᵀ g(θˡ)‖². It is identically 0
        # in the random-walk (symmetric) case, recovering eq. (14). Added to the label
        # prior so the marginalized-PGibbs selection stays exact under the asymmetry.
        if use_gradient_drift:
            with jax.named_scope("propose_grad_barker"):
                grads = jnp.concatenate(
                    [
                        grad_ref[None, :],
                        jax.vmap(lambda th: _theta_logpost_grad(th))(proposed),
                    ],
                    axis=0,
                )
                whitened_sq = jnp.sum((grads @ chol) ** 2, axis=1)
                drift_dot = jnp.sum(grads * (u[None, :] - ensemble), axis=1)
                label_correction = drift_dot - step * whitened_sq
        else:
            grads = jnp.zeros_like(ensemble)
            label_correction = jnp.zeros((num_parameter_particles,), dtype=current_position.dtype)

        return ensemble, label_correction, u, grads

    static = build_smoother_setup(
        target,
        dsmc,
        num_particles=num_particles,
        num_parameter_particles=num_parameter_particles,
        free_mask=latent_free_mask,
        paid_mix=paid_mix,
        diagnostic_metrics=resolved_diagnostic_metrics,
    )

    def _step(
        state: TrajectoryMCMCState,
        key: jnp.ndarray,
        *,
        record_trace: bool,
        batch_size: int | None = None,
    ) -> (
        MarginalParticleStepResult
        | TracedResult[MarginalParticleStepResult, MarginalParticleTraceRecord]
    ):
        param_key, block_key, label_key = random.split(key, 3)
        x_ref = state.latent_trajectory
        traj_dtype = state.trajectory_log_prob.dtype
        complete_dtype = state.complete_log_posterior.dtype

        with jax.named_scope("propose_parameters"):
            parameter_particles, label_correction, parameter_auxiliary, parameter_gradients = (
                _propose_parameter_ensemble(
                    state.position,
                    state.parameter_gradient,
                    param_key,
                    _clip_scale(
                        state.param_step_size,
                        min_scale=min_scale,
                        max_scale=max_scale,
                    ),
                )
            )

        with jax.named_scope("build_context"):
            ctx = build_smoother_context(
                static,
                parameter_particles,
                label_correction,
                reference=state.latent_trajectory,
                delta=state.latent_delta,
                trajectory_dtype=state.trajectory_log_prob.dtype,
            )
        contexts = ctx.contexts

        def root_correction(path: jax.Array) -> jax.Array:
            return jnp.asarray(
                jax.vmap(lambda context: target.replay_correction(local_target, context, path))(
                    contexts
                )
            )

        with jax.named_scope("dsmc_smoother_full"):
            smoother = (dsmc_trace_step if record_trace else dsmc_step)(
                ctx,
                block_key,
                x_ref,
                root_correction if target.state_space.requires_replay else None,
                batch_size=batch_size,
            )
            smoother_result = smoother.result if isinstance(smoother, TracedResult) else smoother

        with jax.named_scope("postprocess"):
            latent_path = smoother_result.latent_path
            scores = smoother_result.label_scores
            final_label_log_probs = scores.log_probs
            origin_path = smoother_result.origin_path
            zero_scalar = jnp.asarray(0.0, dtype=traj_dtype)
            selected_label = random.categorical(label_key, final_label_log_probs).astype(jnp.int32)
            next_position = parameter_particles[selected_label]
            next_context = _select_pytree(contexts, selected_label)
            latent_path = target.replay_path(next_context, latent_path)
            next_traj_lp = scores.trajectory_log_probs[selected_label].astype(traj_dtype)
            next_complete = scores.log_posteriors[selected_label].astype(complete_dtype)
            latent_move = latent_path - x_ref
            latent_move_rms_per_t = jnp.sqrt(
                _masked_mean(latent_move * latent_move, latent_free_mask, axis=-1)
            )
            latent_move_rms = jnp.sqrt(_masked_mean(latent_move * latent_move, latent_free_mask))
            latent_move_max_abs = jnp.max(jnp.abs(latent_move))
            parameter_accepted = jnp.any(next_position != state.position).astype(
                state.position.dtype
            )
            latent_updated = jnp.any(
                (latent_path != x_ref) & smoother_result.proposal_mask, axis=-1
            ).astype(state.position.dtype)
            # Per-(t, d) exact-equality freeze indicator. A completely stuck chain has
            # zero autocovariance at every lag and therefore reports PERFECT ESS under
            # initial-positive-sequence estimators — this counter is the direct gauge
            # standard diagnostics structurally cannot provide. With coordinate-block
            # proposals it also resolves per-coordinate freezing that the per-t
            # `latent_accepted` trace cannot see.
            latent_frozen = (latent_path == x_ref).astype(state.position.dtype)
            latent_frozen_frac = _masked_mean(latent_frozen, latent_free_mask)
            latent_frozen_frac_by_d = _masked_mean(latent_frozen, latent_free_mask, axis=0)

            step_info: dict[str, jax.Array] = {
                **smoother_result.diagnostics,
                "parameter_accepted": parameter_accepted,
                "latent_accepted": latent_updated,
                "latent_proposed": jnp.any(smoother_result.proposal_mask, axis=-1),
                "latent_frozen_frac": latent_frozen_frac,
                "latent_frozen_frac_by_d": latent_frozen_frac_by_d,
                "selected_label": selected_label.astype(jnp.float32),
                "final_particle": origin_path[-1].astype(jnp.float32),
                "latent_move_rms": latent_move_rms,
                "latent_move_max_abs": latent_move_max_abs,
                "latent_move_rms_per_t": latent_move_rms_per_t,
                "final_label_log_probs": final_label_log_probs.astype(jnp.float32),
            }
            if diagnostic_flags.parameter_movement:
                parameter_jump = next_position - state.position
                step_info["parameter_jump_rms"] = (
                    jnp.sqrt(jnp.mean(parameter_jump * parameter_jump))
                    if parameter_jump.size
                    else zero_scalar
                )
            if diagnostic_flags.particle_identity:
                reference_path_hit_rate = jnp.mean((origin_path == 0).astype(state.position.dtype))
                particle_ids = jnp.arange(num_particles, dtype=origin_path.dtype)
                selected_particle_unique_count = jnp.sum(
                    jnp.any(origin_path[:, None] == particle_ids[None, :], axis=0)
                ).astype(traj_dtype)
                step_info.update(
                    {
                        "selected_particle_per_t": origin_path.astype(jnp.float32),
                        "reference_path_hit_rate": reference_path_hit_rate.astype(jnp.float32),
                        "selected_particle_unique_count": (
                            selected_particle_unique_count.astype(jnp.float32)
                        ),
                    }
                )

        result = (
            state._replace(
                position=next_position,
                parameter_gradient=parameter_gradients[selected_label],
                latent_context=next_context,
                latent_trajectory=latent_path,
                trajectory_log_prob=next_traj_lp,
                complete_log_posterior=next_complete,
            ),
            step_info,
        )
        if isinstance(smoother, TracedResult):
            return TracedResult(
                result,
                MarginalParticleTraceRecord(
                    parameters=ParameterProposalTraceRecord(
                        candidates=parameter_particles,
                        auxiliary=parameter_auxiliary,
                        label_log_prior=ctx.initial_label_log_probs,
                    ),
                    smoother=smoother.trace,
                    label_selection=CategoricalTraceRecord(
                        log_weights=final_label_log_probs,
                        selected=selected_label,
                    ),
                ),
            )
        return result

    def _step_fn(
        state: TrajectoryMCMCState, key: jnp.ndarray, *, batch_size: int | None = None
    ) -> MarginalParticleStepResult:
        result = _step(state, key, record_trace=False, batch_size=batch_size)
        assert not isinstance(result, TracedResult)
        return result

    def _trace_step_fn(
        state: TrajectoryMCMCState, key: jnp.ndarray
    ) -> TracedResult[MarginalParticleStepResult, MarginalParticleTraceRecord]:
        result = _step(state, key, record_trace=True)
        assert isinstance(result, TracedResult)
        return result

    return MarginalParticleGibbsKernel(
        step_fn=_step_fn,
        parameter_gradient_fn=_theta_logpost_grad,
        trace_step_fn=_trace_step_fn,
        num_particles=num_particles,
        num_parameter_particles=num_parameter_particles,
        initial_param_step_size=param_step_size,
        target_accept=target_accept,
        min_scale=min_scale,
        max_scale=max_scale,
        preconditioned=parameter_preconditioner_chol is not None,
        dsmc=dsmc,
        diagnostic_metrics=resolved_diagnostic_metrics,
        exact_constraints=exact_constraints,
    )
