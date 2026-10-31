import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.models.ssm.covariance import symmetric_eigh, symmetrize, symmetrize_with_jitter

pytestmark = pytest.mark.contract


def test_symmetrize_with_jitter_handles_batched_covariances():
    cov = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    result = symmetrize_with_jitter(cov, jitter=1e-3)

    expected = np.array(
        [
            [[1.001, 2.5], [2.5, 4.001]],
            [[5.001, 6.5], [6.5, 8.001]],
        ]
    )
    np.testing.assert_allclose(np.asarray(result), expected)


def test_symmetrize_handles_batched_covariances():
    cov = jnp.array(
        [
            [[0.0, 1.0], [3.0, 2.0]],
            [[4.0, 5.0], [7.0, 6.0]],
        ]
    )

    result = symmetrize(cov)

    expected = np.array(
        [
            [[0.0, 2.0], [2.0, 2.0]],
            [[4.0, 6.0], [6.0, 6.0]],
        ]
    )
    np.testing.assert_allclose(np.asarray(result), expected)


@pytest.mark.parametrize("dimension", [1, 3, 40])
def test_symmetric_eigenpairs_preserve_batched_indefinite_matrices(dimension):
    values = np.random.default_rng(17).normal(size=(2, dimension, dimension)).astype(np.float32)
    matrices = jnp.asarray((values + values.swapaxes(-1, -2)) / 2)
    eigenvalues, eigenvectors = jax.jit(symmetric_eigh)(matrices)
    reconstructed = (eigenvectors * eigenvalues[..., None, :]) @ jnp.swapaxes(eigenvectors, -1, -2)
    np.testing.assert_allclose(reconstructed, matrices, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(eigenvalues, np.linalg.eigvalsh(matrices), atol=2e-5, rtol=2e-5)
    assert eigenvalues.dtype == matrices.dtype == jnp.float32
