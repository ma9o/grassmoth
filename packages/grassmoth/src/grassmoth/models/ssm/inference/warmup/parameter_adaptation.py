"""Windowed dense proposal geometry learned from exact MPGibbs warmup states.

Corenflos, https://arxiv.org/abs/2505.04611, section 3.4 permits classical
adaptation of the parameter covariance and scalar step. NumPyro owns the
expanding-window schedule; this module owns within-chain covariance estimation.
"""

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from numpyro.infer.hmc_util import build_adaptation_schedule


class ParameterCovariance(NamedTuple):
    """Within-chain Welford moments in the initializer's whitened coordinates."""

    reference_chol: jax.Array
    sample_count: jax.Array
    mean: jax.Array
    m2: jax.Array


def covariance_windows(num_warmup: int) -> jax.Array:
    """Exclude the initial and terminal step-size-only buffers from covariance learning."""
    windows = tuple(build_adaptation_schedule(num_warmup))[1:-1]
    return jnp.asarray(
        tuple((window.start, window.end) for window in windows), dtype=jnp.int32
    ).reshape((-1, 2))


def initialize_parameter_covariance(positions: jax.Array, chol: jax.Array) -> ParameterCovariance:
    """Start one moment accumulator per cold chain, with a shared reference geometry."""
    chains, dimension = positions.shape
    return ParameterCovariance(
        reference_chol=chol,
        sample_count=jnp.asarray(0),
        mean=jnp.zeros_like(positions),
        m2=jnp.zeros((chains, dimension, dimension), dtype=positions.dtype),
    )


def adapt_parameter_covariance(
    state: ParameterCovariance,
    positions: jax.Array,
    chol: jax.Array,
    iteration: jax.Array,
    windows: jax.Array,
) -> tuple[ParameterCovariance, jax.Array, jax.Array]:
    """Count every selected state, replace geometry at window ends, and reset moments.

    Averaging within-chain covariance excludes between-chain mean dispersion.
    Five reference observations regularize the estimate towards the initial
    positive-definite covariance. Their weight decreases with the window length.
    The returned matrix is fixed for the next complete transition.
    """
    active = jnp.any((iteration >= windows[:, 0]) & (iteration <= windows[:, 1]))
    boundary = jnp.any(iteration == windows[:, 1])

    def accumulate() -> tuple[ParameterCovariance, jax.Array, jax.Array]:
        values = solve_triangular(state.reference_chol, positions.T, lower=True).T
        count = state.sample_count + 1
        delta = values - state.mean
        mean = state.mean + delta / count
        m2 = state.m2 + jnp.einsum("ci,cj->cij", delta, values - mean)
        updated = state._replace(sample_count=count, mean=mean, m2=m2)

        def finish() -> tuple[ParameterCovariance, jax.Array, jax.Array]:
            empirical = jnp.mean(m2, axis=0) / (count - 1)
            weight = count / (count + 5.0)
            covariance = weight * (empirical + empirical.T) / 2 + (1 - weight) * jnp.eye(
                chol.shape[0], dtype=chol.dtype
            )
            learned = jnp.asarray(state.reference_chol @ jnp.linalg.cholesky(covariance))
            return (
                updated._replace(
                    sample_count=jnp.zeros_like(count),
                    mean=jnp.zeros_like(mean),
                    m2=jnp.zeros_like(m2),
                ),
                learned,
                boundary,
            )

        return cast(
            "tuple[ParameterCovariance, jax.Array, jax.Array]",
            jax.lax.cond(boundary, finish, lambda: (updated, chol, boundary)),
        )

    return cast(
        "tuple[ParameterCovariance, jax.Array, jax.Array]",
        jax.lax.cond(active, accumulate, lambda: (state, chol, boundary)),
    )
