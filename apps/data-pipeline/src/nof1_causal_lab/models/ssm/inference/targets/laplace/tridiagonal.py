"""Time-parallel block-tridiagonal algebra for initialization-only Laplace solves.

Gaussian endpoint factors compose by eliminating their shared endpoint. A prefix
scan yields every leading Schur complement without inverting off-diagonal blocks.
Affine scans then solve the triangular systems and recover the inverse subset.
Each operation has logarithmic sequential depth and linear storage in time.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla

from nof1_causal_lab.models.ssm.covariance_utils import symmetrize, symmetrize_with_jitter


def _transpose(matrix: jax.Array) -> jax.Array:
    return jnp.swapaxes(matrix, -1, -2)


def _compose_precision(
    left: tuple[jax.Array, jax.Array, jax.Array],
    right: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Integrate the shared endpoint of two consecutive Gaussian factors."""
    a_left, cross_left, c_left = left
    a_right, cross_right, c_right = right
    middle = symmetrize(c_left + a_right)
    solved = jla.solve(
        middle, jnp.concatenate((_transpose(cross_left), cross_right), axis=-1), assume_a="pos"
    )
    dim = middle.shape[-1]
    return (
        symmetrize(a_left - cross_left @ solved[..., :dim]),
        -cross_left @ solved[..., dim:],
        symmetrize(c_right - _transpose(cross_right) @ solved[..., dim:]),
    )


def _compose_affine(
    left: tuple[jax.Array, jax.Array], right: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    a_left, b_left = left
    a_right, b_right = right
    return a_right @ a_left, a_right @ b_left + b_right


def _compose_covariance(
    left: tuple[jax.Array, jax.Array], right: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    a_left, b_left = left
    a_right, b_right = right
    return a_right @ a_left, a_right @ b_left @ _transpose(a_right) + b_right


def factor_tridiagonal(
    diagonal: jax.Array, upper: jax.Array, *, jitter: float
) -> tuple[jax.Array, jax.Array]:
    """Return diagonal and lower Cholesky blocks of A + jitter I.

    ``upper[t]`` connects times t and t+1; the last block is zero. Disconnected
    chains and singular off-diagonal blocks need no special treatment.
    """
    diagonal = symmetrize_with_jitter(diagonal, jitter=jitter)
    cross = jnp.concatenate((jnp.zeros_like(diagonal[:1]), upper[:-1]))
    _, _, schur = jax.lax.associative_scan(
        _compose_precision, (jnp.zeros_like(diagonal), cross, diagonal)
    )
    chol: jax.Array = jnp.linalg.cholesky(schur)
    lower = _transpose(jla.solve_triangular(chol[:-1], upper[:-1], lower=True))
    return chol, jnp.concatenate((jnp.zeros_like(chol[:1]), lower))


def solve_tridiagonal(chol: jax.Array, lower: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solve from Cholesky blocks, with a vector or matrix RHS at each time."""
    values = rhs.reshape((*rhs.shape[:2], -1))
    forward_matrix = -jla.solve_triangular(chol, lower, lower=True)
    forward_offset = jla.solve_triangular(chol, values, lower=True)
    _, forward = jax.lax.associative_scan(_compose_affine, (forward_matrix, forward_offset))
    following = jnp.concatenate((lower[1:], jnp.zeros_like(lower[:1])))
    backward_matrix = -jla.solve_triangular(_transpose(chol), _transpose(following), lower=False)
    backward_offset = jla.solve_triangular(_transpose(chol), forward, lower=False)
    backward_result: tuple[jax.Array, jax.Array] = jax.lax.associative_scan(
        _compose_affine, (backward_matrix, backward_offset), reverse=True
    )
    return backward_result[1].reshape(rhs.shape)


def inverse_subset_tridiagonal(chol: jax.Array, lower: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return diagonal and adjacent upper blocks of the inverse in parallel."""
    eye = jnp.broadcast_to(jnp.eye(chol.shape[-1], dtype=chol.dtype), chol.shape)
    inverse_chol = jla.solve_triangular(chol, eye, lower=True)
    following = jnp.concatenate((lower[1:], jnp.zeros_like(lower[:1])))
    backward = -jla.solve_triangular(_transpose(chol), _transpose(following), lower=False)
    conditional_covariance = _transpose(inverse_chol) @ inverse_chol
    _, diagonal = jax.lax.associative_scan(
        _compose_covariance, (backward, conditional_covariance), reverse=True
    )
    upper = backward[:-1] @ diagonal[1:]
    return (
        symmetrize(diagonal),
        jnp.concatenate((upper, jnp.zeros_like(diagonal[:1]))),
    )
