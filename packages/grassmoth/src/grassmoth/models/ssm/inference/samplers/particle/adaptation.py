"""Warmup-only path scales, with observation-curvature limits for guided drift."""

from collections.abc import Callable
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from grassmoth.artifacts.sampling import FixedLatentStepSpec, PathUpdateSpec
from grassmoth.compute.programs import scalar_operand
from grassmoth.models.ssm.inference.mcmc_state import _clip_scale
from grassmoth.models.ssm.inference.targets.particle import ParticleContext, ParticleTarget
from grassmoth.models.ssm.inference.targets.sequential import sequential_observations
from grassmoth.models.ssm.shapes import Array, Bool, Float, Int

INITIAL_DELTA = 0.01
ADAPTATION_WINDOW = 100
ADAPTATION_TOLERANCE = 0.05
ADAPTATION_RHO = 0.5
ADAPTATION_RHO_MIN = 1e-3
ADAPTATION_GAMMA = -0.5
DELTA_MIN = 1e-5
DELTA_MAX = 10.0


def _curvature_sizes(
    observe: Callable[[Float[Array, " D"], Int[Array, ""]], Float[Array, ""]],
    path: Float[Array, "T D"],
    free_mask: Bool[Array, "T D"],
) -> Float[Array, "T D"]:
    """Absolute Hessian row sums on free coordinates, without a step-size floor."""

    def at_time(
        state: Float[Array, " D"], free: Bool[Array, " D"], index: Int[Array, ""]
    ) -> Float[Array, " D"]:
        hessian = jax.hessian(observe)(state, index)
        hessian = (hessian + hessian.T) * 0.5 * free[:, None] * free[None, :]
        return jnp.sum(jnp.abs(hessian), axis=-1)

    return jax.vmap(at_time)(path, free_mask, jnp.arange(path.shape[0]))


def _curvature_limits(
    observe: Callable[[Float[Array, " D"], Int[Array, ""]], Float[Array, ""]],
    path: Float[Array, "T D"],
    free_mask: Bool[Array, "T D"],
) -> Float[Array, "T D"]:
    """Initial diagonal limits from the auxiliary's local information.

    For a scalar affine score with Hessian H, the auxiliary variance A = delta/2
    adds K = (1 + A*H)**2 / A of information about the current path. A = 1/|H|
    cancels it for H < 0, where delta is 2R for a Gaussian reading of variance
    R, and minimizes it to 4*H for H > 0. Where H vanishes, K = 1/A has no
    finite minimizer; the cap and the movement multiplier are heuristics.
    Absolute row sums extend the scalar rule conservatively to coupled
    coordinates, and one tight channel cannot throttle another coordinate.
    The initializer uses this pointwise rule. Subsequent geometry comes from
    completed warmup windows, not the current state. The local rule bounds
    neither nonlinear nor label-mixture curvature.
    """
    return 2.0 / jnp.maximum(_curvature_sizes(observe, path, free_mask), 2.0 / DELTA_MAX)


def _guided_observation(
    target: ParticleTarget,
    context: ParticleContext,
    path: Float[Array, "T D"],
) -> Callable[[Float[Array, " D"], Int[Array, ""]], Float[Array, ""]]:
    """Bind one exact observation-factor projection for curvature measurements."""
    memory = sequential_observations(
        target, jax.tree.map(lambda value: value[None], context), context
    )
    reference = memory.path(path)
    return lambda state, index: memory.log_probs(
        memory.replace_current(reference[index], state, index), index
    )[0]


def guided_curvature(
    target: ParticleTarget,
    context: ParticleContext,
    path: Float[Array, "T D"],
    free_mask: Bool[Array, "T D"],
) -> Float[Array, "T D"]:
    """Measure exact observation curvature without selecting the next transition's metric."""
    return _curvature_sizes(_guided_observation(target, context, path), path, free_mask)


def guided_delta_limit(
    target: ParticleTarget,
    context: ParticleContext,
    path: Float[Array, "T D"],
    free_mask: Bool[Array, "T D"],
) -> Float[Array, "T D"]:
    """Select the initial pointwise metric before learning from warmup history."""
    return _curvature_limits(_guided_observation(target, context, path), path, free_mask)


class CurvatureMoments(NamedTuple):
    """Per-coordinate RMS observation curvature in the current warmup window."""

    sample_count: Int[Array, ""]
    rms: Float[Array, "*batch T D"]


def initialize_curvature_moments(geometry: Float[Array, "*batch T D"]) -> CurvatureMoments:
    """Start an empty window independently for each replica, time and free coordinate."""
    return CurvatureMoments(jnp.asarray(0), jnp.zeros_like(geometry))


def adapt_curvature_geometry(
    moments: CurvatureMoments,
    curvature: Float[Array, "*batch T D"],
    geometry: Float[Array, "*batch T D"],
    iteration: Int[Array, ""],
    windows: Int[Array, "W 2"],
) -> tuple[CurvatureMoments, Float[Array, "*batch T D"]]:
    """Publish only completed-window geometry, retaining it throughout every window.

    For scalar Hessian H and fixed auxiliary variance A, expected local
    information is E[(1+A*H)**2/A] = 1/A + 2 E[H] + A E[H**2]. Its minimizing
    A is 1/sqrt(E[H**2]); delta is twice A. Absolute Hessian row sums retain
    the existing conservative extension to coupled coordinates. The step cap
    handles zero curvature. Hypot accumulates RMS without squaring large
    finite float32 curvature. Repeated states count as chain history.

    Expanding windows discard earlier transient moments. Their initial and
    terminal buffers do not learn geometry, and retention never changes it.
    Window boundaries are still adaptive warmup, not invariant transitions.
    A multimodal or poorly explored window can give inefficient geometry;
    exact particle weights remain unchanged.
    """
    active = jnp.any((iteration >= windows[:, 0]) & (iteration <= windows[:, 1]))
    boundary = jnp.any(iteration == windows[:, 1])

    def accumulate() -> tuple[CurvatureMoments, jax.Array]:
        count = moments.sample_count + 1
        weight = moments.sample_count / count
        rms = jnp.hypot(jnp.sqrt(weight) * moments.rms, curvature / jnp.sqrt(count))
        updated = CurvatureMoments(count, rms)

        def finish() -> tuple[CurvatureMoments, jax.Array]:
            learned = 2.0 / jnp.maximum(rms, 2.0 / DELTA_MAX)
            return initialize_curvature_moments(geometry), learned

        return cast(
            "tuple[CurvatureMoments, jax.Array]",
            jax.lax.cond(boundary, finish, lambda: (updated, geometry)),
        )

    return cast(
        "tuple[CurvatureMoments, jax.Array]",
        jax.lax.cond(active, accumulate, lambda: (moments, geometry)),
    )


def initial_delta(spec: PathUpdateSpec) -> float:
    """Resolve the assigned fixed scale or the adaptation starting value."""
    return (
        spec.latent_step.delta
        if isinstance(spec.latent_step, FixedLatentStepSpec)
        else INITIAL_DELTA
    )


class LatentAdaptation(NamedTuple):
    """Runtime scale-adaptation controls, independent of the proposal structure."""

    enabled: Bool[Array, "1"]
    target: Float[Array, "1"]


def latent_adaptation(spec: PathUpdateSpec) -> LatentAdaptation:
    """Resolve the authored scale policy before tracing any transitions."""
    return LatentAdaptation(
        enabled=scalar_operand(value=not isinstance(spec.latent_step, FixedLatentStepSpec)),
        target=scalar_operand(
            value=0.5
            if isinstance(spec.latent_step, FixedLatentStepSpec)
            else spec.latent_step.target
        ),
    )


class DsmcAdaptation(NamedTuple):
    """Per-time scales and the recent accepted/proposed movement counts."""

    delta: Float[Array, "*batch T"]
    accepted: Float[Array, "*batch 100 T"]
    proposed: Float[Array, "*batch 100 T"]
    iteration: Int[Array, ""]
    accepted_total: Float[Array, "*batch T"]
    proposed_total: Float[Array, "*batch T"]


def initialize_adaptation(delta: Float[Array, "*batch T"]) -> DsmcAdaptation:
    """Initialize one scale vector or a batch of scale vectors."""
    shape = (*delta.shape[:-1], ADAPTATION_WINDOW, delta.shape[-1])
    window = jnp.zeros(shape, dtype=delta.dtype)
    return DsmcAdaptation(
        delta, window, window, jnp.asarray(0), jnp.zeros_like(delta), jnp.zeros_like(delta)
    )


@jax.jit
def adapt_scales(
    state: DsmcAdaptation,
    settings: LatentAdaptation,
    accepted: Float[Array, "*batch T"],
    proposed: Bool[Array, "*batch T"] | Float[Array, "*batch T"],
    num_warmup: int | Int[Array, ""],
    maximum: Float[Array, "*batch T"],
) -> DsmcAdaptation:
    """Adapt only proposed coordinates; freeze scales and windows after warmup."""
    target = settings.target[0]

    def update() -> DsmcAdaptation:
        slot = state.iteration % ADAPTATION_WINDOW
        accept_window = state.accepted.at[..., slot, :].set(accepted)
        proposal_window = state.proposed.at[..., slot, :].set(proposed)
        accepted_total = state.accepted_total - state.accepted[..., slot, :] + accepted
        count = state.proposed_total - state.proposed[..., slot, :] + proposed
        rate = accepted_total / jnp.maximum(count, 1)
        iteration = state.iteration + 1
        learning_rate = jnp.maximum(
            iteration.astype(state.delta.dtype) ** ADAPTATION_GAMMA * ADAPTATION_RHO,
            ADAPTATION_RHO_MIN,
        )
        change = learning_rate * state.delta * (rate - target) / target
        active = (
            (jnp.abs(rate - target) >= ADAPTATION_TOLERANCE)
            & (iteration > ADAPTATION_WINDOW)
            & (count > 0)
        )
        delta = _clip_scale(
            jnp.where(active, state.delta + change, state.delta),
            min_scale=jnp.minimum(DELTA_MIN, maximum),
            max_scale=maximum,
        )
        return DsmcAdaptation(
            delta, accept_window, proposal_window, iteration, accepted_total, count
        )

    adapted: DsmcAdaptation = jax.lax.cond(
        settings.enabled[0] & (state.iteration < num_warmup), update, lambda: state
    )
    return adapted
