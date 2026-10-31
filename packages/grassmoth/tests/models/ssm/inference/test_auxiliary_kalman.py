"""Independent dense Gaussian reference for normalized parallel path proposals."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from grassmoth.models.ssm.inference.samplers.particle.blocked_kalman import (
    blocked_gaussian_proposal,
    prepare_block_geometry,
)
from grassmoth.models.ssm.inference.samplers.particle.kalman import (
    GaussianPathTerms,
    gaussian_path_proposal,
)
from grassmoth.models.ssm.inference.samplers.particle.twist import GaussianTwist

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize("steps", [1, 5])
def test_parallel_proposal_matches_dense_gaussian_law(steps):
    dimension = 2
    transition = np.array([[0.8, 0.3], [-0.2, 0.6]])
    matrices = np.broadcast_to(transition, (steps, dimension, dimension)).copy()
    matrices[0] = 0
    offsets = np.tile(np.array([0.3, -0.15]), (steps, 1))
    covariances = np.broadcast_to(np.array([[0.3, 0.08], [0.08, 0.2]]), matrices.shape).copy()
    covariances[0] *= 3
    information = np.broadcast_to(np.array([[1.8, -0.25], [-0.25, 0.9]]), matrices.shape)
    shift = np.tile(np.array([0.4, -0.1]), (steps, 1))
    # Dense precision assembled directly from the joint Markov density.
    operator = np.eye(steps * dimension)
    innovation_precision = np.zeros_like(operator)
    for time in range(steps):
        row = slice(time * dimension, (time + 1) * dimension)
        innovation_precision[row, row] = np.linalg.inv(covariances[time])
        if time:
            operator[row, slice((time - 1) * dimension, time * dimension)] = -transition
    precision = operator.T @ innovation_precision @ operator + np.kron(
        np.eye(steps), information[0]
    )
    covariance = np.linalg.inv(precision)
    mean = covariance @ (operator.T @ innovation_precision @ offsets.ravel() + shift.ravel())
    proposal = jax.jit(gaussian_path_proposal)(
        GaussianPathTerms(
            jnp.asarray(matrices),
            jnp.asarray(offsets),
            jnp.asarray(covariances),
            GaussianTwist(jnp.asarray(information), jnp.asarray(shift)),
        ),
        jnp.ones((steps, dimension), dtype=bool),
    )
    # Every shifted block agrees with the independently conditioned dense law,
    # including both endpoint factors and padded first/last blocks.
    reference = np.arange(steps * dimension).reshape(steps, dimension) * 0.1
    length, block_shift = 2, 1
    blocked = blocked_gaussian_proposal(
        GaussianPathTerms(
            jnp.asarray(matrices),
            jnp.asarray(offsets),
            jnp.asarray(covariances),
            GaussianTwist(jnp.asarray(information), jnp.asarray(shift)),
        ),
        jnp.ones((steps, dimension), dtype=bool),
        jnp.asarray(reference),
        length=length,
        starts=jnp.arange((steps + length - 1) // length + 1) * length - block_shift,
    )
    terms = GaussianPathTerms(
        jnp.asarray(matrices),
        jnp.asarray(offsets),
        jnp.asarray(covariances),
        GaussianTwist(jnp.asarray(information), jnp.asarray(shift)),
    )
    starts = jnp.arange((steps + length - 1) // length + 1) * length - block_shift
    free = jnp.ones((steps, dimension), dtype=bool)
    geometry = prepare_block_geometry(
        terms, free, jnp.asarray(reference), length=length, starts=starts
    )
    changed_reference = jnp.asarray(reference) * -0.6 + 0.7
    conditioned = jax.vmap(lambda block: block.condition(changed_reference))(geometry)
    rebuilt = blocked_gaussian_proposal(
        terms, free, changed_reference, length=length, starts=starts
    )
    for actual, expected in zip(conditioned[:3], rebuilt[:3], strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    for block in range(blocked.offsets.shape[0]):
        times = np.arange(length) + block * length - block_shift
        valid = (times >= 0) & (times < steps)
        if not valid.any():
            continue
        selected = np.array([t * dimension + d for t in times[valid] for d in range(dimension)])
        outside = np.setdiff1d(np.arange(steps * dimension), selected)
        conditional_covariance = np.linalg.inv(precision[np.ix_(selected, selected)])
        conditional_mean = mean[selected] - conditional_covariance @ precision[
            np.ix_(selected, outside)
        ] @ (reference.ravel()[outside] - mean[outside])
        law = jax.tree.map(lambda value, block=block: value[block], blocked)
        draw = law.sample(jax.random.key(900 + block))
        expected_log_density = multivariate_normal.logpdf(
            np.asarray(draw)[valid].ravel(), conditional_mean, conditional_covariance
        )
        np.testing.assert_allclose(law.log_density(draw), expected_log_density, atol=1e-5)

    key = jax.random.key(839)
    path = jax.jit(lambda: proposal.sample(key))()
    actual = proposal.log_density(path)
    expected = multivariate_normal.logpdf(np.asarray(path).ravel(), mean, covariance)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=5e-6)
    # The parallel draw must equal a serial recursion with the same innovations.
    noise = np.asarray(jax.random.normal(key, offsets.shape, dtype=jnp.float32))
    serial = []
    previous = np.zeros(dimension)
    for matrix, offset, chol, epsilon in zip(*proposal[:3], noise, strict=True):
        previous = np.asarray(matrix) @ previous + np.asarray(offset) + np.asarray(chol) @ epsilon
        serial.append(previous)
    np.testing.assert_allclose(path, serial, rtol=2e-6, atol=2e-6)


def test_fixed_coordinate_has_no_sampling_noise_or_density_cost():
    matrices = jnp.zeros((1, 2, 2))
    terms = GaussianPathTerms(
        matrices,
        jnp.array([[0.3, 2.1]]),
        jnp.array([[[0.4, 0.0], [0.0, 0.0]]]),
        GaussianTwist(matrices, jnp.zeros((1, 2))),
    )
    proposal = gaussian_path_proposal(terms, jnp.array([[True, False]]))
    path = proposal.sample(jax.random.key(42))
    assert path[0, 1] == terms.offsets[0, 1]
    expected = multivariate_normal.logpdf(np.asarray(path[0, :1]), mean=[0.3], cov=[[0.4]])
    np.testing.assert_allclose(proposal.log_density(path), expected, atol=1e-6)
