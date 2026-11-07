"""Independent dense Gaussian references for retained-component mGRAD weights."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from grassmoth.models.ssm.inference.samplers.particle.guided_geometry import (
    _diagonal_proposal_geometry,
    _proposal_geometry,
)
from grassmoth.models.ssm.inference.samplers.particle.marginal_mixture import (
    _gaussian_integral_weights,
    marginal_mixture_bank,
)
from grassmoth.models.ssm.inference.samplers.particle.twist import GaussianTwist

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize(
    ("dimension", "pinned"), [(1, False), (2, False), (3, True), (5, False), (40, True)]
)
@pytest.mark.parametrize("diagonal", [False, True])
def test_mixture_bank_matches_dense_integral(dimension, pinned, diagonal):
    rng = np.random.default_rng(31)
    count, labels = 5, 3
    factors = rng.normal(size=(labels, dimension, dimension))
    covariance = jnp.asarray(
        factors @ factors.swapaxes(-1, -2) / dimension + 0.3 * np.eye(dimension)
    )
    if diagonal:
        covariance = jax.vmap(jnp.diag)(jnp.diagonal(covariance, axis1=-2, axis2=-1))
    variance = jnp.linspace(0.3, 0.7, dimension)
    auxiliary = jnp.linspace(-0.4, 0.6, dimension)
    mask = jnp.arange(dimension) != (dimension - 1 if pinned else dimension)
    factory = _diagonal_proposal_geometry if diagonal else _proposal_geometry
    proposal = jax.vmap(factory, in_axes=(0, None, None, None, None))(
        covariance,
        GaussianTwist(jnp.diag(mask / variance), jnp.where(mask, auxiliary / variance, 0.0)),
        mask,
        jnp.zeros((dimension, dimension)),
        jnp.zeros(dimension),
    )
    means = jnp.asarray(rng.normal(size=(count, labels, dimension)))
    probabilities = jax.nn.log_softmax(jnp.asarray(rng.normal(size=(count, labels))), axis=-1)
    prior = jax.nn.log_softmax(jnp.asarray(rng.normal(size=labels)))
    bank = marginal_mixture_bank(
        proposal,
        proposal.center(means, mask),
        probabilities,
        prior,
        jax.random.normal(jax.random.key(1), (count - 1, dimension)),
        jax.random.uniform(jax.random.key(2), (count,)),
        jnp.linspace(0.2, -0.3, dimension),
        mask,
    )
    gradients = jnp.asarray(rng.normal(size=(count, dimension)))
    actual = np.asarray(bank.log_weights(gradients, variance, mask))
    free = np.flatnonzero(mask)
    factors = np.asarray(bank.geometry.chol)
    if diagonal:
        factors = np.asarray([np.diag(row) for row in factors])
    covariances = factors @ factors.swapaxes(-1, -2)
    covariances = covariances[:, free][:, :, free]
    free_variance = np.asarray(variance)[free]
    maps = covariances / free_variance[None, None, :]
    expected = []
    for index in range(count):
        others = np.arange(count) != index
        matrix = maps[others].reshape(-1, len(free))
        joint_covariance = (
            block_diag(*covariances[others]) + matrix @ np.diag(free_variance) @ matrix.T
        )
        center = np.asarray(bank.particles[index] + variance * gradients[index])[free]
        joint_mean = np.asarray(bank.prior_centers)[others][:, free].ravel() + matrix @ center
        expected.append(
            multivariate_normal.logpdf(
                np.asarray(bank.particles)[others][:, free].ravel(), joint_mean, joint_covariance
            )
            + float(bank.label_correction[index])
        )
    expected = np.asarray(expected)
    np.testing.assert_allclose(
        actual - actual[0], expected - expected[0], atol=2e-5 * np.sqrt(dimension)
    )


@pytest.mark.parametrize(("count", "dimension"), [(1, 2), (7, 3), (129, 8)])
def test_fused_gaussian_integral_matches_dense_solve(count, dimension):
    rng = np.random.default_rng(812)
    factors = rng.normal(size=(2, count, dimension, dimension))
    precision = factors @ factors.swapaxes(-1, -2) + np.eye(dimension)
    shift = rng.normal(size=(2, count, dimension))
    constant = rng.normal(size=(2, count))
    expected = constant + 0.5 * (
        np.sum(shift * np.linalg.solve(precision, shift[..., None])[..., 0], axis=-1)
        - np.linalg.slogdet(precision)[1]
    )
    actual = jax.jit(jax.vmap(_gaussian_integral_weights))(
        *(jnp.asarray(value, dtype=jnp.float32) for value in (precision, shift, constant))
    )
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "logits", [[0.0], [-jnp.inf, 0.0, jnp.log(4.0), -jnp.inf], [1000.0, 1001.0, 998.0], [0.0] * 8]
)
@pytest.mark.parametrize("carried", [False, True])
def test_uniform_component_selection_has_the_same_categorical_law(logits, carried):
    from scipy.special import softmax

    from grassmoth.models.ssm.inference.samplers.particle.math import (
        categorical_from_running,
        categorical_from_uniform,
        categorical_running_mass,
    )

    values = np.asarray(logits, dtype=np.float32)
    uniforms = (np.arange(32768, dtype=np.float32) + 0.5) / 32768
    draw = (
        (
            lambda scores, uniforms: categorical_from_running(
                categorical_running_mass(scores), uniforms
            )
        )
        if carried
        else categorical_from_uniform
    )
    selected = np.asarray(jax.jit(draw)(jnp.asarray(values), jnp.asarray(uniforms)))
    expected = np.searchsorted(
        np.cumsum(softmax(values.astype(np.float64))), uniforms, side="right"
    )
    np.testing.assert_array_equal(selected, expected)
    frequencies = np.bincount(selected, minlength=len(values)) / len(uniforms)
    np.testing.assert_allclose(
        frequencies, softmax(values.astype(np.float64)), atol=1 / len(uniforms), rtol=0
    )


def test_uniform_component_endpoints_never_choose_zero_mass():
    from grassmoth.models.ssm.inference.samplers.particle.math import (
        categorical_from_running,
        categorical_from_uniform,
        categorical_running_mass,
    )

    rng = np.random.default_rng(91)
    logits = rng.normal(size=(4096, 8)).astype(np.float32) * 10 - 1e6
    logits[rng.uniform(size=logits.shape) < 0.5] = -np.inf
    logits[:, 3] = -1e6
    logits[:, -1] = -np.inf
    uniforms = np.resize(np.array([0.0, 0.5, 1 - 2**-23], dtype=np.float32), len(logits))
    for draw in (
        categorical_from_uniform,
        lambda values, noise: categorical_from_running(categorical_running_mass(values), noise),
    ):
        selected = np.asarray(jax.jit(draw)(jnp.asarray(logits), jnp.asarray(uniforms)))
        assert np.isfinite(logits[np.arange(len(logits)), selected]).all()
        empty = jnp.array([[-jnp.inf] * 8, [jnp.nan] * 8])
        np.testing.assert_array_equal(jax.jit(draw)(empty, jnp.array([0.5, 0.5])), 0)
