"""Data-conditioned path centers used only to initialize particle chains.

Each Gauss--Newton iteration smooths an affine approximation of the dynamics
and a positive-semidefinite quadratic approximation of the exact reading
densities, interval readings included through the shared curvature's window
memory. Neither approximation enters a sampling target or a diagnostic.
Strong nonlinearity, separated modes and clipped curvature can leave a poor
start.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.inference.warmup.path_curvature import linearize_path

if TYPE_CHECKING:
    from grassmoth.models.ssm.inference.targets.particle import ParticleContext, ParticleTarget


def ieks_path(
    target: ParticleTarget, context: ParticleContext, initial_path: jax.Array
) -> jax.Array:
    """Run at most twenty damped smoothing iterations at fixed parameters.

    Exact replay enforces recurrences, static values and observed constraints
    after every trial. Backtracking accepts only improvement of the exact path
    density, including every interval reading. A failed numerical solve
    propagates a non-finite result to the initializer's typed failure boundary;
    it never selects a replacement flat path. The limits of twenty iterations and
    eight step halvings are heuristics; no paper or recorded benchmark sets them.
    """

    def score(path: jax.Array) -> jax.Array:
        return jnp.asarray(target.factor_log_prob(context, path, target.observations))

    def iteration(path: jax.Array) -> jax.Array:
        curvature = linearize_path(target, context, path)
        proposed = curvature.mean(target, context, path)
        direction = proposed - path
        current_score = score(path)

        def trial(
            carry: tuple[jax.Array, jax.Array, jax.Array], _: None
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
            accepted, result, damping = carry
            candidate = target.replay_path(context, path + damping * direction)
            candidate_score = score(candidate)
            better = jnp.isfinite(candidate_score) & (candidate_score >= current_score)
            take = ~accepted & better
            return (
                accepted | better,
                jnp.where(take, candidate, result),
                damping * 0.5,
            ), None

        (_, result, _), _ = jax.lax.scan(
            trial, (jnp.asarray(False), path, jnp.asarray(1.0, dtype=path.dtype)), None, length=8
        )
        return jnp.where(jnp.all(jnp.isfinite(proposed)), result, jnp.full_like(result, jnp.nan))

    return jnp.asarray(jax.lax.fori_loop(0, 20, lambda _, path: iteration(path), initial_path))
