"""Exact companion moves composed with each marginalized particle sweep.

The centred move proposes every parameter with the stochastic path coordinates
held fixed; deterministic stages are replayed under the proposal. The scale move
proposes only the coordinates behind the diffusion-scale sites and holds the
exact path innovations fixed, so the path is rebuilt under the new scales; its
ratio includes that change of variables through the non-centred chart. Both are
symmetric random walks shaped by the sweep's parameter preconditioner, with a
step scalar of their own, so each leaves the exact joint posterior invariant.
The centred NUTS move targets the centred move's conditional with one NUTS
transition, which adapts its own step and diagonal mass in warmup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
from blackjax.adaptation.step_size import dual_averaging_adaptation
from numpyro.infer.hmc import HMCState, hmc

from grassmoth.artifacts.model.parameter import SiteKind
from grassmoth.models.ssm.inference.joint_coordinates import build_joint_coordinates

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

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


class ConditionalNuts(NamedTuple):
    """NUTS on every parameter at the current stochastic path, adapting itself in warmup."""

    initialize: Callable[[TrajectoryMCMCState, jax.Array, int], HMCState]
    step: Callable[
        [TrajectoryMCMCState, HMCState, jax.Array],
        tuple[TrajectoryMCMCState, HMCState, Mapping[str, jax.Array]],
    ]


class CompanionKernel(NamedTuple):
    """Ordered Metropolis moves, then the optional NUTS move, each with separate randomness."""

    moves: tuple[CompanionMove, ...]
    nuts: ConditionalNuts | None

    def initial_tuning(
        self, states: TrajectoryMCMCState, key: jax.Array, num_warmup: int
    ) -> CompanionTuning:
        """Start each Metropolis move from its chain's sweep step, and NUTS at its chain's state."""
        steps = jnp.repeat(states.param_step_size[:, None], len(self.moves), axis=1)
        nuts = self.nuts
        return CompanionTuning(
            steps,
            restarted_dual_averaging(steps),
            None
            if nuts is None
            else jax.vmap(lambda state, chain_key: nuts.initialize(state, chain_key, num_warmup))(
                states, jax.random.split(key, steps.shape[0])
            ),
        )

    def step(
        self, state: TrajectoryMCMCState, tuning: CompanionTuning, key: jax.Array
    ) -> tuple[TrajectoryMCMCState, CompanionTuning, jax.Array, dict[str, jax.Array]]:
        """Apply every move once; return Metropolis acceptance indicators and per-move records."""
        accepted: list[jax.Array] = []
        records: dict[str, jax.Array] = {}
        for index, move in enumerate(self.moves):
            state, accept, jump = _metropolis(
                move, state, tuning.step_size[index], jax.random.fold_in(key, index)
            )
            accepted.append(accept)
            records |= {
                f"{move.name}_accept_prob": accept,
                f"{move.name}_jump_rms": jump,
                f"{move.name}_step_size": tuning.step_size[index],
            }
        if self.nuts is not None and tuning.nuts is not None:
            state, moved, nuts_records = self.nuts.step(
                state, tuning.nuts, jax.random.fold_in(key, len(self.moves))
            )
            tuning = tuning._replace(nuts=moved)
            records |= nuts_records
        acceptance = (
            jnp.stack(accepted) if accepted else jnp.zeros((0,), dtype=state.position.dtype)
        )
        return state, tuning, acceptance, records


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
    """Per-chain Metropolis step scalars and NUTS state, adapted during warmup and frozen after."""

    step_size: jax.Array
    dual_averaging: DualAveragingAdaptationState
    nuts: HMCState | None


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


def build_conditional_nuts(target: ParticleTarget, rebuild: CompanionProposal) -> ConditionalNuts:
    """Refresh NUTS at the sweep's parameters and path, then rebuild the state where it lands."""

    def potential(path: jax.Array) -> Callable[[jax.Array], jax.Array]:
        # The path density replays deterministic coordinates under each position.
        return lambda position: (
            -target.log_posterior(position, path, target.observations, target.times)
        )

    initialize_hmc, sample_hmc = hmc(potential_fn_gen=potential, algo="NUTS")

    def initialize(state: TrajectoryMCMCState, key: jax.Array, num_warmup: int) -> HMCState:
        return initialize_hmc(
            state.position,
            num_warmup=num_warmup,
            inverse_mass_matrix=jnp.sum(state.parameter_chol * state.parameter_chol, axis=1),
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False,
            model_args=(state.latent_trajectory,),
            rng_key=key,
        )

    def step(
        state: TrajectoryMCMCState, nuts: HMCState, key: jax.Array
    ) -> tuple[TrajectoryMCMCState, HMCState, Mapping[str, jax.Array]]:
        path = state.latent_trajectory
        # The sweep moved the parameters and the path since the last transition.
        energy, gradient = jax.value_and_grad(potential(path))(state.position)
        current = nuts._replace(
            z=state.position, potential_energy=energy, z_grad=gradient, rng_key=key
        )
        moved = sample_hmc(current, model_args=(path,))
        updated, _ = rebuild(state, moved.z)
        jump = moved.z - state.position
        return (
            updated,
            # The sweep's key drives the transition. The state keeps its own key,
            # which a continued run's typed sweep keys must not replace.
            moved._replace(rng_key=nuts.rng_key),
            {
                "centred_nuts_accept_prob": jnp.asarray(moved.accept_prob),
                "centred_nuts_jump_rms": jnp.sqrt(jnp.mean(jump * jump)),
                "centred_nuts_step_size": jnp.asarray(current.adapt_state.step_size),
                # The chain diagnostics count divergences and tree depths under these names.
                "diverging": jnp.asarray(moved.diverging),
                "tree_depth": jnp.ceil(jnp.log2(moved.num_steps + 1)).astype(jnp.int32),
            },
        )

    return ConditionalNuts(initialize, step)


def build_companion_kernel(
    target: ParticleTarget,
    companion: Literal["centred_mh", "centred_and_scale_noncentred_mh", "centred_nuts"],
    *,
    diffusion_mask: jax.Array,
    parameter_gradient: Callable[[jax.Array], jax.Array],
) -> CompanionKernel:
    """Order the centred move first; a scale move without diffusion coordinates is omitted."""
    centred = build_companion_proposal(target, "centred_mh", parameter_gradient)
    if companion == "centred_nuts":
        return CompanionKernel((), build_conditional_nuts(target, centred))
    moves = (
        CompanionMove("centred_mh", jnp.ones(target.initial_position.shape, dtype=bool), centred),
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
    return CompanionKernel(moves, None)
