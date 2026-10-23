"""Exact Metropolis companion moves composed with each marginalized particle sweep.

The centred move proposes every parameter with the stochastic path coordinates
held fixed; deterministic stages are replayed under the proposal. The scale move
proposes only the coordinates behind the diffusion-scale sites and holds the
exact path innovations fixed, so the path is rebuilt under the new scales; its
ratio includes that change of variables through the non-centred chart. Both are
symmetric random walks shaped by the sweep's parameter preconditioner, with a
step scalar of their own, so each leaves the exact joint posterior invariant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
from blackjax.adaptation.step_size import dual_averaging_adaptation

from grassmoth.artifacts.model.parameter import SiteKind
from grassmoth.models.ssm.inference.joint_coordinates import build_joint_coordinates

if TYPE_CHECKING:
    from collections.abc import Callable

    from blackjax.adaptation.step_size import DualAveragingAdaptationState

    from grassmoth.models.ssm.inference.mcmc_state import TrajectoryMCMCState
    from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
    from grassmoth.models.ssm.parameterization import PriorRuntimeBundle

type CompanionMoveName = Literal["centred_mh", "scale_noncentred_mh"]
type CompanionProposal = Callable[
    [TrajectoryMCMCState, jax.Array], tuple[TrajectoryMCMCState, jax.Array]
]


class CompanionMove(NamedTuple):
    """One exact proposal map over the masked unconstrained parameter coordinates."""

    name: CompanionMoveName
    mask: jax.Array
    propose: CompanionProposal


class CompanionKernel(NamedTuple):
    """Ordered moves applied to the latest state with separate randomness."""

    moves: tuple[CompanionMove, ...]

    def step(
        self, state: TrajectoryMCMCState, step_sizes: jax.Array, key: jax.Array
    ) -> tuple[TrajectoryMCMCState, jax.Array, jax.Array]:
        """Apply every move once; return acceptance indicators and parameter jump RMS per move."""
        accepted: list[jax.Array] = []
        jumps: list[jax.Array] = []
        for index, move in enumerate(self.moves):
            state, accept, jump = _metropolis(
                move, state, step_sizes[index], jax.random.fold_in(key, index)
            )
            accepted.append(accept)
            jumps.append(jump)
        return state, jnp.stack(accepted), jnp.stack(jumps)


def _metropolis(
    move: CompanionMove, state: TrajectoryMCMCState, step_size: jax.Array, key: jax.Array
) -> tuple[TrajectoryMCMCState, jax.Array, jax.Array]:
    proposal_key, accept_key = jax.random.split(key)
    noise = jax.random.normal(proposal_key, state.position.shape, state.position.dtype)
    # step_size uses the sweep's convention: proposal variance 4 * step * Sigma.
    displacement = jnp.sqrt(4 * step_size) * (state.parameter_chol @ noise)
    proposal, log_ratio = move.propose(
        state, state.position + jnp.where(move.mask, displacement, 0)
    )
    # A NaN ratio compares false, so it rejects.
    accept = jnp.log(jax.random.uniform(accept_key, dtype=log_ratio.dtype)) < log_ratio
    updated = cast("TrajectoryMCMCState", jax.lax.cond(accept, lambda: proposal, lambda: state))
    jump = updated.position - state.position
    return (
        updated,
        accept.astype(state.position.dtype),
        jnp.sqrt(jnp.mean(jump * jump)) if jump.size else jnp.zeros((), state.position.dtype),
    )


class CompanionTuning(NamedTuple):
    """Per-chain step scalars of each move, adapted during warmup and frozen after it."""

    step_size: jax.Array
    dual_averaging: DualAveragingAdaptationState


def initial_companion_tuning(step_size: jax.Array, moves: int) -> CompanionTuning:
    """Start every move of each chain from that chain's initial sweep step."""
    steps = jnp.repeat(step_size[:, None], moves, axis=1)
    return CompanionTuning(steps, restarted_dual_averaging(steps))


def restarted_dual_averaging(step_size: jax.Array) -> DualAveragingAdaptationState:
    """Start dual averaging at each step scalar, whatever the batch shape."""
    initialize = dual_averaging_adaptation(target=0.0)[0]
    flat = jax.vmap(initialize)(step_size.reshape(-1))
    return jax.tree.map(lambda value: value.reshape(step_size.shape), flat)


def diffusion_scale_mask(target: ParticleTarget, priors: PriorRuntimeBundle) -> jax.Array:
    """Select the unconstrained coordinates the diffusion-scale sites depend on."""
    sites = tuple(
        site.name for site in priors.registry if site.site_kind == SiteKind.DIFFUSION_DIAG
    )
    if not sites:
        return jnp.zeros(target.initial_position.shape, dtype=bool)

    def scales(position: jax.Array) -> jax.Array:
        values = target.parameters.constrain(position)
        return jnp.concatenate(tuple(jnp.ravel(values[name]) for name in sites))

    # NumPyro owns each site's bijection, including any prior reparameterization.
    return jnp.any(jax.jacfwd(scales)(target.initial_position) != 0, axis=0)


def build_companion_proposal(
    target: ParticleTarget,
    name: CompanionMoveName,
    parameter_gradient: Callable[[jax.Array], jax.Array],
) -> CompanionProposal:
    """Map a state and proposed parameters to the proposed state and exact log ratio."""
    observations, times = target.observations, target.times
    coordinates = (
        build_joint_coordinates(target, "non_centred") if name == "scale_noncentred_mh" else None
    )

    def propose(
        state: TrajectoryMCMCState, position: jax.Array
    ) -> tuple[TrajectoryMCMCState, jax.Array]:
        context = target.context(position, times)
        if coordinates is None:
            path = target.replay_path(context, state.latent_trajectory)
            current, _ = target.log_posterior_from_context(
                state.position,
                target.context(state.position, times),
                state.latent_trajectory,
                observations,
            )
            complete, trajectory = target.log_posterior_from_context(
                position, context, path, observations
            )
            log_ratio = complete - current
        else:
            point = coordinates.encode(state.position, state.latent_trajectory)
            proposed = point.at[: coordinates.parameter_dimension].set(
                position - coordinates.parameter_center
            )
            _, path = coordinates.decode(proposed)
            log_ratio = coordinates.log_density(proposed) - coordinates.log_density(point)
            complete, trajectory = target.log_posterior_from_context(
                position, context, path, observations
            )
        return state._replace(
            position=position,
            parameter_gradient=parameter_gradient(position),
            latent_context=context,
            latent_trajectory=path,
            complete_log_posterior=complete.astype(state.complete_log_posterior.dtype),
            trajectory_log_prob=trajectory.astype(state.trajectory_log_prob.dtype),
        ), log_ratio

    return propose


def build_companion_kernel(
    target: ParticleTarget,
    companion: Literal["centred_mh", "centred_and_scale_noncentred_mh"],
    *,
    diffusion_mask: jax.Array,
    parameter_gradient: Callable[[jax.Array], jax.Array],
) -> CompanionKernel:
    """Order the centred move first; a scale move without diffusion coordinates is omitted."""
    moves = (
        CompanionMove(
            "centred_mh",
            jnp.ones(target.initial_position.shape, dtype=bool),
            build_companion_proposal(target, "centred_mh", parameter_gradient),
        ),
    )
    if companion == "centred_and_scale_noncentred_mh" and bool(jnp.any(diffusion_mask)):
        moves = (
            *moves,
            CompanionMove(
                "scale_noncentred_mh",
                diffusion_mask,
                build_companion_proposal(target, "scale_noncentred_mh", parameter_gradient),
            ),
        )
    return CompanionKernel(moves)
