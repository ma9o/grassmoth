"""Marginalized Particle Gibbs joint parameter/trajectory kernel.

Implements the M=2-by-default collapsed Particle Gibbs construction from
Corenflos (2025), "Particle Gibbs without the Gibbs bit", for directly
evaluable SSM potentials. The parameter proposal is formed in unconstrained
space via the auxiliary decomposition

    u | theta  ~ N(theta + a L b(theta), 2 delta Sigma)
    theta' | u ~ N(u, 2 delta Sigma)

where ``Sigma = LLᵀ``, ``a = sqrt(2 delta)`` and
``b(theta) = v / max(1, ||v||)``, with ``v = a Lᵀg(theta)``.
Here ``g`` is a fixed, parameter-only gradient oracle. The default computes
the complete-data gradient on a fixed pilot path chosen at construction; it
must not use the changing reference trajectory during the chain. ``parameter_proposal``
selects the drift: the default ``random_walk`` sets ``g = 0`` (the symmetric special
case), while ``pseudo_langevin`` (Corenflos 2025, §3.1) bounds the
theta->u drift to one half-proposal standard deviation. Below that bound the
two halves reproduce preconditioned MALA. A stale pilot, poor preconditioner or
separated modes can still reduce mixing. The
resulting asymmetry is corrected exactly in the Barker label-selection weights
(identically zero in the random-walk case). The latent trajectory is updated by
conditional SMC against the posterior mixture over the parameter ensemble.

Step-size convention. ``delta`` (``param_step_size``) is the variance coefficient
of each half, not the conventional MALA step. The combined kernel has variance
``4 delta Sigma`` (and drift ``a L b`` for pseudo-Langevin), so an unbounded MALA
step ``h`` corresponds to ``delta = h / 4``. Fits start at
``INITIAL_PARAM_STEP_SIZE``; warmup adapts ``delta`` within the step bounds and
rescales it whenever ``Sigma`` is re-estimated, keeping the average variance.
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
from jax.scipy.linalg import solve_triangular

from grassmoth.artifacts.sampling import DsmcSpec, MarginalParticleGibbsSpec, PathUpdateSpec
from grassmoth.compute.programs import scalar_operand
from grassmoth.models.ssm.compile.state_space import AdditiveRecurrences, ParallelRecurrences
from grassmoth.models.ssm.inference.mcmc_state import TrajectoryMCMCState, _clip_scale
from grassmoth.models.ssm.inference.samplers.particle.context import (
    build_smoother_context,
)
from grassmoth.models.ssm.inference.samplers.particle.diagnostics import (
    build_particle_diagnostic_flags,
)
from grassmoth.models.ssm.inference.samplers.particle.dsmc import step as dsmc_step
from grassmoth.models.ssm.inference.samplers.particle.dsmc import (
    trace_step as dsmc_trace_step,
)
from grassmoth.models.ssm.inference.samplers.particle.guided import guided_step, trace_guided_step
from grassmoth.models.ssm.inference.samplers.particle.math import (
    _masked_mean,
    _select_pytree,
)
from grassmoth.models.ssm.inference.samplers.particle.setup import build_smoother_setup
from grassmoth.models.ssm.inference.tracing import (
    CategoricalTraceRecord,
    MarginalParticleTraceRecord,
    ParameterProposalTraceRecord,
    TracedResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from grassmoth.artifacts.sampling import ParticleDiagnosticMetric
    from grassmoth.models.ssm.inference.conditioning import ExactStateConstraints
    from grassmoth.models.ssm.inference.samplers.particle.contract import (
        PaidMixture,
    )
    from grassmoth.models.ssm.inference.targets.particle import ParticleTarget


_DEFAULT_OPTIONS = MarginalParticleGibbsSpec()
# Warmup adapts the step from here; fits always run warmup, so these are not settings.
INITIAL_PARAM_STEP_SIZE = 0.02
PARAM_STEP_SIZE_MIN = 1e-6
PARAM_STEP_SIZE_MAX = 1e3
type MarginalParticleStepResult = tuple[TrajectoryMCMCState, dict[str, jnp.ndarray]]


def _bounded_parameter_drift(gradient: jax.Array, chol: jax.Array, scale: jax.Array) -> jax.Array:
    """Unit-ball drift in whitened half-proposal coordinates, without norm overflow.

    An undefined pilot score defines zero drift, not zero target density. This is
    a total parameter-only proposal oracle, used identically in both Gaussian laws.
    """
    whitened = scale * (gradient @ chol)
    whitened = jnp.where(jnp.all(jnp.isfinite(whitened), axis=-1, keepdims=True), whitened, 0.0)
    magnitude = jnp.max(jnp.abs(whitened), axis=-1, keepdims=True, initial=1.0)
    unit = whitened / magnitude
    return unit / jnp.maximum(1.0 / magnitude, jnp.linalg.norm(unit, axis=-1, keepdims=True))


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
    num_parameter_particles: int
    initial_param_step_size: jax.Array
    target_accept: jax.Array
    min_scale: jax.Array
    max_scale: jax.Array
    preconditioned: bool
    initial_parameter_chol: jax.Array
    path_update: PathUpdateSpec
    diagnostic_metrics: tuple[ParticleDiagnosticMetric, ...]
    exact_constraints: ExactStateConstraints | None


def build_marginal_particle_gibbs_kernel(
    target: ParticleTarget,
    *,
    num_parameter_particles: int,
    param_step_size: float = INITIAL_PARAM_STEP_SIZE,
    target_accept: float | None = None,
    min_scale: float = PARAM_STEP_SIZE_MIN,
    max_scale: float = PARAM_STEP_SIZE_MAX,
    parameter_preconditioner_chol: jnp.ndarray | None = None,
    parameter_proposal: Literal[
        "random_walk", "pseudo_langevin"
    ] = _DEFAULT_OPTIONS.parameter_proposal,
    path_update: PathUpdateSpec = _DEFAULT_OPTIONS.path_update,
    paid_mix: PaidMixture | None = None,
    parameter_reference_path: jnp.ndarray | None = None,
    exact_constraints: ExactStateConstraints | None = None,
    diagnostic_metrics: tuple[ParticleDiagnosticMetric, ...] = _DEFAULT_OPTIONS.diagnostic_metrics,
) -> MarginalParticleGibbsKernel:
    """Build a marginalized Particle Gibbs joint state update."""
    diagnostic_flags = build_particle_diagnostic_flags(diagnostic_metrics=diagnostic_metrics)
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

    runtime_min_scale = scalar_operand(value=min_scale)
    runtime_max_scale = scalar_operand(value=max_scale)

    assert all(
        isinstance(stage, (AdditiveRecurrences, ParallelRecurrences))
        for stage in target.state_space.recurrence
    ), "The fitting preflight must establish an associative deterministic recurrence"

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
        gradient = evaluate(z)
        return jnp.where(jnp.all(jnp.isfinite(gradient)), gradient, 0.0)

    def _propose_parameter_ensemble(
        current_position: jnp.ndarray,
        current_gradient: jnp.ndarray,
        key: jnp.ndarray,
        step_size: jnp.ndarray,
        chol: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dim = int(current_position.shape[0])
        step = jnp.asarray(step_size, dtype=current_position.dtype)
        proposal_scale = jnp.sqrt(jnp.asarray(2.0, dtype=current_position.dtype) * step)
        aux_key, proposal_key = random.split(key)

        # Only θ→u is drifted. Bound in the covariance metric so a poor fixed
        # pilot cannot move u thousands of Gaussian standard deviations away.
        if use_gradient_drift:
            with jax.named_scope("propose_grad_drift"):
                grad_ref = current_gradient
                drift = proposal_scale * (
                    chol @ _bounded_parameter_drift(grad_ref, chol, proposal_scale)
                )
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

        # Exact Barker correction log q(u|θ) − log q(θ|u), using the bounded
        # drift at EVERY label: b·L⁻¹(u−θ)/a − ||b||²/2. Bounding only the
        # forward drift while correcting with the raw score would change the target.
        if use_gradient_drift:
            with jax.named_scope("propose_grad_barker"):
                grads = jnp.concatenate(
                    [
                        grad_ref[None, :],
                        jax.vmap(lambda th: _theta_logpost_grad(th))(proposed),
                    ],
                    axis=0,
                )
                bounded = _bounded_parameter_drift(grads, chol, proposal_scale)
                displacement = (
                    solve_triangular(chol, (u[None, :] - ensemble).T, lower=True).T / proposal_scale
                )
                label_correction = jnp.sum(bounded * displacement - 0.5 * bounded**2, axis=1)
        else:
            grads = jnp.zeros_like(ensemble)
            label_correction = jnp.zeros((num_parameter_particles,), dtype=current_position.dtype)

        return ensemble, label_correction, u, grads

    static = build_smoother_setup(
        target,
        path_update,
        num_parameter_particles=num_parameter_particles,
        free_mask=latent_free_mask,
        paid_mix=paid_mix,
        diagnostic_metrics=diagnostic_metrics,
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
                        min_scale=runtime_min_scale[0],
                        max_scale=runtime_max_scale[0],
                    ),
                    state.parameter_chol,
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

        def root_correction(path: jax.Array, local_prior: jax.Array) -> jax.Array:
            return jnp.asarray(
                jax.vmap(lambda context, prior: target.replay_correction(context, path, prior))(
                    contexts, local_prior
                )
            )

        with jax.named_scope("particle_smoother_full"):
            if isinstance(path_update, DsmcSpec):
                smoother = (dsmc_trace_step if record_trace else dsmc_step)(
                    ctx,
                    block_key,
                    x_ref,
                    root_correction if target.state_space.requires_replay else None,
                    batch_size=batch_size,
                )
            else:
                guide = target.context(ctx.guide_position, runtime_times)
                smoother = (trace_guided_step if record_trace else guided_step)(
                    target, ctx, guide, path_update, block_key, x_ref
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
            parameter_jump = next_position - state.position
            step_info["parameter_jump_rms"] = jax.lax.cond(
                diagnostic_flags.parameter_movement[0],
                lambda: (
                    jnp.sqrt(jnp.mean(parameter_jump * parameter_jump))
                    if parameter_jump.size
                    else zero_scalar
                ),
                lambda: zero_scalar,
            )

            def particle_identity() -> dict[str, jax.Array]:
                particle_ids = jnp.arange(path_update.num_particles, dtype=origin_path.dtype)
                return {
                    "selected_particle_per_t": origin_path.astype(jnp.float32),
                    "reference_path_hit_rate": jnp.mean((origin_path == 0).astype(jnp.float32)),
                    "selected_particle_unique_count": jnp.sum(
                        jnp.any(origin_path[:, None] == particle_ids[None, :], axis=0)
                    ).astype(jnp.float32),
                }

            step_info.update(
                jax.lax.cond(
                    diagnostic_flags.particle_identity[0],
                    particle_identity,
                    lambda: {
                        "selected_particle_per_t": jnp.zeros(origin_path.shape, jnp.float32),
                        "reference_path_hit_rate": jnp.asarray(0.0, jnp.float32),
                        "selected_particle_unique_count": jnp.asarray(0.0, jnp.float32),
                    },
                )
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
        num_parameter_particles=num_parameter_particles,
        initial_param_step_size=scalar_operand(value=param_step_size),
        target_accept=scalar_operand(value=target_accept),
        min_scale=runtime_min_scale,
        max_scale=runtime_max_scale,
        preconditioned=parameter_preconditioner_chol is not None,
        initial_parameter_chol=jnp.eye(
            target.initial_position.size, dtype=target.initial_position.dtype
        )
        if parameter_preconditioner_chol is None
        else jnp.asarray(parameter_preconditioner_chol, dtype=target.initial_position.dtype),
        path_update=path_update,
        diagnostic_metrics=diagnostic_metrics,
        exact_constraints=exact_constraints,
    )
