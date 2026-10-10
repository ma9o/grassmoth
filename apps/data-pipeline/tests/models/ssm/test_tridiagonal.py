"""Dense references for the time-parallel initialization precision algebra."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.models.ssm.inference.targets.laplace.shared import (
    _block_profile_inverse_subset_from_cholesky,
    _factor_block_profile_cholesky,
    _solve_block_profile_from_cholesky,
    precision_logdet,
)

pytestmark = pytest.mark.inference(concern="warmup")


@pytest.mark.parametrize(
    ("times", "dim", "dtype"),
    [(1, 2, "float64"), (2, 1, "float32"), (7, 3, "float64"), (16, 2, "float32")],
)
def test_parallel_precision_matches_dense_solutions_inverse_and_gradients(times, dim, dtype):
    """Cover singleton, odd/even length, disconnected chains and multiple RHSs."""
    rng = np.random.default_rng(407)
    root = np.zeros((times * dim, times * dim))
    for index in range(times):
        start = index * dim
        block = np.tril(rng.normal(scale=0.2, size=(dim, dim))) + 2 * np.eye(dim)
        root[start : start + dim, start : start + dim] = block
        if index and index != times // 2:
            root[start : start + dim, start - dim : start] = rng.normal(size=(dim, dim))
    dense = root @ root.T
    rows = np.arange(times)
    widths = np.minimum(1, times - 1 - rows)
    if times > 2:
        widths[times // 2 - 1] = 0
    with jax.enable_x64():
        diagonal = jnp.asarray(
            np.stack([dense[i * dim : (i + 1) * dim, i * dim : (i + 1) * dim] for i in rows]),
            dtype=dtype,
        )
        upper = np.zeros((1, times, dim, dim))
        for i in range(times - 1):
            upper[0, i] = dense[i * dim : (i + 1) * dim, (i + 1) * dim : (i + 2) * dim]
        upper = jnp.asarray(upper, dtype=dtype)
        row_upper = jnp.asarray(widths)
        row_lower = jnp.concatenate((jnp.zeros(1, dtype=row_upper.dtype), row_upper[:-1]))
        rhs = jnp.asarray(rng.normal(size=(times, dim, 3)), dtype=dtype)

        def evaluate(scale):
            diag = diagonal + scale * jnp.eye(dim)
            chol, lower = _factor_block_profile_cholesky(diag, upper, row_upper, row_lower)
            solution = _solve_block_profile_from_cholesky(chol, lower, rhs, row_upper, row_lower)
            vector = _solve_block_profile_from_cholesky(
                chol, lower, rhs[..., 0], row_upper, row_lower
            )
            inverse = _block_profile_inverse_subset_from_cholesky(chol, lower, row_upper, row_lower)
            logdet = precision_logdet(diag, upper, row_upper, row_lower)
            return logdet + jnp.sum(solution**2), (solution, vector, inverse, logdet)

        def dense_evaluate(scale):
            matrix = jnp.asarray(dense, dtype=dtype) + (scale + 1e-6) * jnp.eye(
                times * dim, dtype=dtype
            )
            solution = jnp.linalg.solve(matrix, rhs.reshape(times * dim, 3))
            logdet = jnp.linalg.slogdet(matrix)[1]
            return logdet + jnp.sum(solution**2)

        scale = jnp.asarray(0.2, dtype=dtype)
        (value, (solution, vector, (inv_diag, inv_upper), logdet)), gradient = jax.jit(
            jax.value_and_grad(evaluate, has_aux=True)
        )(scale)
        expected_value, expected_gradient = jax.value_and_grad(dense_evaluate)(scale)
        matrix = dense + (float(scale) + 1e-6) * np.eye(times * dim)
        expected_solution = np.linalg.solve(matrix, np.asarray(rhs).reshape(times * dim, 3))
        inverse = np.linalg.inv(matrix)
        tolerance = 3e-5 if dtype == "float32" else 1e-10
        for actual, expected in (
            (solution.reshape(times * dim, 3), expected_solution),
            (vector.reshape(-1), expected_solution[:, 0]),
            (value, expected_value),
            (gradient, expected_gradient),
            (logdet, np.linalg.slogdet(matrix)[1]),
        ):
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        for i in rows:
            np.testing.assert_allclose(
                inv_diag[i],
                inverse[i * dim : (i + 1) * dim, i * dim : (i + 1) * dim],
                rtol=tolerance,
                atol=tolerance,
            )
            if i + 1 < times:
                np.testing.assert_allclose(
                    inv_upper[0, i],
                    inverse[i * dim : (i + 1) * dim, (i + 1) * dim : (i + 2) * dim],
                    rtol=tolerance,
                    atol=tolerance,
                )
