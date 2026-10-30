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

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import solve_triangular

from grassmoth.artifacts.sampling import (
    DsmcSpec,
    MarginalParticleGibbsSpec,
    ParameterProposal,
    ParticleDiagnosticMetric,
    PathUpdateSpec,
)
from grassmoth.compute.programs import ProgramCompiler, scalar_operand
from grassmoth.models.ssm.compile.state_space import AdditiveRecurrences, ParallelRecurrences
from grassmoth.models.ssm.inference.conditioning import ExactStateConstraints
from grassmoth.models.ssm.inference.mcmc_state import (
    TrajectoryMCMCState,
    _clip_scale,
    trajectory_state,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.augmentation import (
    ConditionalPathScores,
    MarginalLabelScores,
    PathUpdateBoundary,
    PrecisionPathUpdate,
    begin_path_update,
    finish_path_update,
)
from grassmoth.models.ssm.inference.samplers.particle.contract import (
    PaidMixture,
    ParticleSmootherResult,
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
from grassmoth.models.ssm.inference.targets.coordinates import (
    PathChart,
    ScaleChart,
    TranslationChart,
)
from grassmoth.models.ssm.inference.targets.particle import ParticleContext, ParticleTarget
from grassmoth.models.ssm.inference.tracing import (
    CategoricalTraceRecord,
    MarginalParticleTraceRecord,
    ParameterProposalTraceRecord,
    SequentialSmootherTraceRecord,
    SmootherTraceRecord,
    TracedResult,
)
from grassmoth.models.ssm.shapes import Array, Float, PRNGKeyArray

_DEFAULT_OPTIONS = MarginalParticleGibbsSpec()
# Warmup adapts the step from here; fits always run warmup, so these are not settings.
INITIAL_PARAM_STEP_SIZE = 0.02
PARAM_STEP_SIZE_MIN = 1e-6
PARAM_STEP_SIZE_MAX = 1e3
type MarginalParticleStepResult = tuple[TrajectoryMCMCState, dict[str, jnp.ndarray]]


def _bounded_parameter_drift(
    gradient: Float[Array, "*batch U"], chol: Float[Array, "U U"], scale: Float[Array, ""]
) -> Float[Array, "*batch U"]:
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


class ParameterMoveInputs(NamedTuple):
    """Native proposal and boundary shared by production and component compilation."""

    particles: jax.Array
    correction: jax.Array
    auxiliary: jax.Array
    gradients: jax.Array
    boundary: PathUpdateBoundary
    label_key: jax.Array
    target: ParticleTarget
    reference: jax.Array


class SelectedParameter(NamedTuple):
    """The native marginal label choice, before deterministic path replay."""

    label: jax.Array
    position: jax.Array
    context: ParticleContext
    scores: MarginalLabelScores


class ParticleProfilePrograms(NamedTuple):
    """Bound compiled native parts, with every numerical output synchronized by the caller."""

    parameter_move: Callable[[], tuple[jax.Array, ...]]
    path_smoother: Callable[[], tuple[jax.Array, ...]]


@partial(jax.tree_util.register_dataclass, data_fields=("operands",), meta_fields=("function",))
@dataclass(frozen=True, eq=False)
class SmootherProfileProgram[Operand]:
    """One symbolic trace owns its function and operands, including abstract JAX leaves."""

    operands: tuple[Operand, ...]
    function: Callable[[list[Operand], jax.Array], ParticleSmootherResult]

    def run(self, path: jax.Array) -> ParticleSmootherResult:
        """Invoke the native smoother with the operands from the same symbolic trace."""
        return self.function(list(self.operands), path)


class MarginalParticleGibbsKernel(NamedTuple):
    """Callable joint kernel and static metadata."""

    profile_parts: Callable[
        [TrajectoryMCMCState, jax.Array, int, ProgramCompiler], ParticleProfilePrograms
    ]
    step_fn: MarginalParticleStep
    parameter_gradient_fn: Callable[[Float[Array, " U"]], Float[Array, " U"]]
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
    initial_chart: PathChart | None
    learn_translation: bool
    translation_pairs: jax.Array | None


def _restrict_chart(chart: PathChart, movable: jax.Array) -> PathChart:
    """Leave coordinates that some label cannot move in their physical values."""
    translation = chart if isinstance(chart, TranslationChart) else chart.translation
    translation = TranslationChart(
        jnp.where(movable[:, None], translation.slopes, 0.0), translation.reference
    )
    if isinstance(chart, TranslationChart):
        return translation
    return ScaleChart(translation, jnp.where(movable[:, None], chart.exponents, 0.0), chart.center)


def build_marginal_particle_gibbs_kernel(
    target: ParticleTarget,
    *,
    num_parameter_particles: int,
    param_step_size: float = INITIAL_PARAM_STEP_SIZE,
    target_accept: float | None = None,
    parameter_preconditioner_chol: Float[Array, "U U"] | None = None,
    parameter_proposal: ParameterProposal = _DEFAULT_OPTIONS.parameter_proposal,
    path_update: PathUpdateSpec = _DEFAULT_OPTIONS.path_update,
    paid_mix: PaidMixture | None = None,
    parameter_reference_path: Float[Array, "T D"] | None = None,
    exact_constraints: ExactStateConstraints | None = None,
    diagnostic_metrics: tuple[ParticleDiagnosticMetric, ...] = _DEFAULT_OPTIONS.diagnostic_metrics,
    initial_chart: PathChart | None = None,
    learn_translation: bool = False,
    translation_pairs: jax.Array | None = None,
) -> MarginalParticleGibbsKernel:
    """Build a marginalized Particle Gibbs joint state update."""
    diagnostic_flags = build_particle_diagnostic_flags(diagnostic_metrics=diagnostic_metrics)
    if num_parameter_particles < 2:
        raise ValueError("marginal_particle_gibbs requires num_parameter_particles >= 2.")
    use_gradient_drift = parameter_proposal == "pseudo_langevin"
    if target_accept is None:
        target_accept = _DEFAULT_OPTIONS.param_target_accept

    runtime_min_scale = scalar_operand(value=PARAM_STEP_SIZE_MIN)
    runtime_max_scale = scalar_operand(value=PARAM_STEP_SIZE_MAX)

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
    if initial_chart is not None:
        # A shared chart must preserve every label's constrained path support.
        movable = jnp.all(latent_free_mask, axis=0)
        initial_chart = _restrict_chart(initial_chart, movable)
        if translation_pairs is not None:
            translation_pairs = translation_pairs & movable[:, None]
    if learn_translation:
        assert initial_chart is not None
        assert translation_pairs is not None

    def _theta_logpost_grad(z: Float[Array, " U"]) -> Float[Array, " U"]:
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
        current_position: Float[Array, " U"],
        current_gradient: Float[Array, " U"],
        key: PRNGKeyArray,
        step_size: Float[Array, ""],
        chol: Float[Array, "U U"],
    ) -> tuple[Float[Array, "K U"], Float[Array, " K"], Float[Array, " U"], Float[Array, "K U"]]:
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

    def _prepare(state: TrajectoryMCMCState, key: jax.Array) -> ParameterMoveInputs:
        param_key, block_key, label_key = random.split(key, 3)
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
            chart_target = target if state.chart is None else target.in_coordinates(state.chart)
            chart_state = (
                state
                if state.chart is None
                else state._replace(
                    latent_context=chart_target.context(state.position, runtime_times),
                    latent_trajectory=state.chart.encode(state.position, state.latent_trajectory),
                )
            )
            chart_setup = (
                static
                if state.chart is None
                else build_smoother_setup(
                    chart_target,
                    path_update,
                    num_parameter_particles=num_parameter_particles,
                    free_mask=latent_free_mask,
                    paid_mix=paid_mix,
                    diagnostic_metrics=diagnostic_metrics,
                )
            )
            boundary = begin_path_update(
                chart_target,
                chart_setup,
                path_update,
                chart_state,
                parameter_particles,
                label_correction,
                block_key,
            )
        return ParameterMoveInputs(
            parameter_particles,
            label_correction,
            parameter_auxiliary,
            parameter_gradients,
            boundary,
            label_key,
            chart_target,
            chart_state.latent_trajectory,
        )

    def _smooth(
        prepared: ParameterMoveInputs,
        x_ref: jax.Array,
        *,
        record_trace: bool,
        batch_size: int | None,
    ) -> (
        ParticleSmootherResult
        | TracedResult[ParticleSmootherResult, SmootherTraceRecord]
        | TracedResult[ParticleSmootherResult, SequentialSmootherTraceRecord]
    ):
        boundary = prepared.boundary
        chart_target = prepared.target
        ctx = boundary.context
        block_key = boundary.key
        contexts = ctx.contexts

        def root_correction(path: jax.Array, local_prior: jax.Array) -> jax.Array:
            return jnp.asarray(
                jax.vmap(
                    lambda context, prior: chart_target.replay_correction(context, path, prior)
                )(contexts, local_prior)
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
                guide = chart_target.context(ctx.guide_position, runtime_times)
                smoother = (trace_guided_step if record_trace else guided_step)(
                    chart_target,
                    ctx,
                    guide,
                    path_update,
                    block_key,
                    x_ref,
                    log_precisions=boundary.precisions.log_values
                    if isinstance(boundary, PrecisionPathUpdate)
                    else None,
                )
        return smoother

    def _select(
        prepared: ParameterMoveInputs, smoother: ParticleSmootherResult
    ) -> SelectedParameter:
        with jax.named_scope("label_selection"):
            scores = finish_path_update(
                prepared,
                ConditionalPathScores(smoother.label_scores),
                prepared.target,
                smoother.latent_path,
            )
            label = random.categorical(prepared.label_key, scores.log_probs).astype(jnp.int32)
            return SelectedParameter(
                label,
                prepared.particles[label],
                _select_pytree(prepared.boundary.context.contexts, label),
                scores,
            )

    def _physical_path(
        state: TrajectoryMCMCState,
        selected: SelectedParameter,
        latent_path: jax.Array,
        chart_reference: jax.Array,
    ) -> tuple[ParticleContext, jax.Array, jax.Array, jax.Array]:
        """Restore the physical path and its caches in production and profiling."""
        label, position, context, scores = selected
        trajectory_dtype = state.trajectory_log_prob.dtype
        complete_dtype = state.complete_log_posterior.dtype
        if state.chart is None:
            return (
                context,
                target.replay_path(context, latent_path),
                scores.trajectory_log_probs[label].astype(trajectory_dtype),
                scores.log_posteriors[label].astype(complete_dtype),
            )
        # Retaining a chart coordinate at the same parameter retains its exact
        # stored physical value. A float32 round trip must not manufacture movement.
        retained = (
            jnp.all(position == state.position)
            & (latent_path == chart_reference)
            & target.independent_mask
        )
        latent_path = jnp.where(
            retained, state.latent_trajectory, state.chart.decode(position, latent_path)
        )
        context = target.context(position, runtime_times)
        latent_path = target.replay_path(context, latent_path)
        # Score the returned physical values without replaying a second time.
        trajectory_density = jnp.asarray(
            target.factor_log_prob(context, latent_path, runtime_observations),
            dtype=trajectory_dtype,
        )
        complete = jnp.asarray(
            target.log_prior(position) + trajectory_density, dtype=complete_dtype
        )
        return context, latent_path, trajectory_density, complete

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
        prepared = _prepare(state, key)
        (
            parameter_particles,
            _label_correction,
            parameter_auxiliary,
            parameter_gradients,
            boundary,
            _,
            _chart_target,
            chart_reference,
        ) = prepared
        ctx = boundary.context
        x_ref = state.latent_trajectory
        traj_dtype = state.trajectory_log_prob.dtype
        smoother = _smooth(
            prepared, chart_reference, record_trace=record_trace, batch_size=batch_size
        )
        smoother_result = smoother.result if isinstance(smoother, TracedResult) else smoother
        with jax.named_scope("postprocess"):
            latent_path = smoother_result.latent_path
            selected = _select(prepared, smoother_result)
            selected_label, next_position, _, scores = selected
            final_label_log_probs = scores.log_probs
            origin_path = smoother_result.origin_path
            zero_scalar = jnp.asarray(0.0, dtype=traj_dtype)
            with jax.named_scope("particle_smoother_full"):
                next_context, latent_path, next_traj_lp, next_complete = _physical_path(
                    state, selected, latent_path, chart_reference
                )
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
                # A parameter compensation can move x while leaving z fixed.
                # Tune the conditional path step using movement in its own chart;
                # the recorded movement and freeze diagnostics remain physical.
                "latent_controller_accepted": jnp.any(
                    (smoother_result.latent_path != chart_reference)
                    & smoother_result.proposal_mask,
                    axis=-1,
                ).astype(state.position.dtype)
                if state.chart is not None
                else latent_updated,
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
            trajectory_state(
                position=next_position,
                parameter_gradient=parameter_gradients[selected_label],
                latent_context=next_context,
                latent_trajectory=latent_path,
                trajectory_log_prob=next_traj_lp,
                complete_log_posterior=next_complete,
                latent_delta=state.latent_delta,
                param_step_size=state.param_step_size,
                parameter_chol=state.parameter_chol,
                param_da=state.param_da,
                chart=state.chart,
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

    def _profile_parts(
        states: TrajectoryMCMCState, keys: jax.Array, batch_size: int, compiler: ProgramCompiler
    ) -> ParticleProfilePrograms:
        # Closure conversion exposes exactly the numerical operands consumed by
        # the native smoother, including arrays captured by its density helpers.
        # It happens while tracing, never during a timed device execution.
        def smoother_program(
            prepared: ParameterMoveInputs, path: jax.Array
        ) -> SmootherProfileProgram[jax.Array]:
            def smooth(path: jax.Array) -> ParticleSmootherResult:
                result = _smooth(prepared, path, record_trace=False, batch_size=batch_size)
                assert not isinstance(result, TracedResult)
                return result

            values, converted = jax.jit(smooth).trace(path).closure_convert()
            return SmootherProfileProgram(
                tuple(jnp.asarray(value) for value in values),
                cast("Callable[[list[jax.Array], jax.Array], ParticleSmootherResult]", converted),
            )

        state = jax.tree.map(lambda value: value[0], states)
        prepared = _prepare(state, keys[0])
        draw_shape = jax.eval_shape(
            lambda path: _smooth(prepared, path, record_trace=False, batch_size=batch_size),
            prepared.reference,
        )
        empty_draws = jax.tree.map(
            lambda shape: jnp.zeros((states.position.shape[0], *shape.shape), shape.dtype),
            draw_shape,
        )

        def profile_parameter_move(
            state: TrajectoryMCMCState, key: jax.Array, draw: ParticleSmootherResult
        ) -> tuple[SmootherProfileProgram[jax.Array], SelectedParameter]:
            prepared = _prepare(state, key)
            # This compilation owns both the operand order and the converted
            # smoother. Retracing concrete inputs could deduplicate constants
            # differently, so retain the function from this exact symbolic trace.
            return smoother_program(prepared, prepared.reference), _select(prepared, draw)

        parameter = compiler(jax.vmap(profile_parameter_move), states, keys, empty_draws)
        smoother, _ = parameter.call(states, keys, empty_draws)

        def profile_smoother_setup(
            smoother: SmootherProfileProgram[jax.Array], path: jax.Array
        ) -> ParticleSmootherResult:
            return smoother.run(path)

        smooth_batch = jax.vmap(profile_smoother_setup)
        chart_paths = (
            states.latent_trajectory
            if states.chart is None
            else jax.vmap(lambda chart, position, path: chart.encode(position, path))(
                states.chart, states.position, states.latent_trajectory
            )
        )
        draws = compiler(smooth_batch, smoother, chart_paths).call(smoother, chart_paths)
        smoother, selected = jax.block_until_ready(parameter.call(states, keys, draws))

        def profile_path_smoother(
            smoother: SmootherProfileProgram[jax.Array],
            state: TrajectoryMCMCState,
            selected: SelectedParameter,
            chart_reference: jax.Array,
        ) -> tuple[ParticleSmootherResult, tuple[ParticleContext, jax.Array, jax.Array, jax.Array]]:
            result = smoother.run(chart_reference)
            return result, _physical_path(state, selected, result.latent_path, chart_reference)

        path = compiler(jax.vmap(profile_path_smoother), smoother, states, selected, chart_paths)

        def run_parameter() -> tuple[jax.Array, ...]:
            return cast(
                "tuple[jax.Array, ...]", tuple(jax.tree.leaves(parameter.call(states, keys, draws)))
            )

        def run_path() -> tuple[jax.Array, ...]:
            return cast(
                "tuple[jax.Array, ...]",
                tuple(jax.tree.leaves(path.call(smoother, states, selected, chart_paths))),
            )

        return ParticleProfilePrograms(run_parameter, run_path)

    return MarginalParticleGibbsKernel(
        profile_parts=_profile_parts,
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
        initial_chart=initial_chart,
        learn_translation=learn_translation,
        translation_pairs=translation_pairs,
    )
