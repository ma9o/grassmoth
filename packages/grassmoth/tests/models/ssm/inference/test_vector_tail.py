"""Vector tail differential against the unfused exact bank and conditional resampler."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.models.ssm.inference.samplers.particle.guided import (
    _killing_noise,
    _killing_reference,
)
from grassmoth.models.ssm.inference.samplers.particle.guided_geometry import (
    _DiagonalProposalGeometry,
)
from grassmoth.models.ssm.inference.samplers.particle.guided_tail import diagonal_mixture_tail
from grassmoth.models.ssm.inference.samplers.particle.marginal_mixture import MarginalMixtureBank
from grassmoth.models.ssm.inference.samplers.particle.math import (
    categorical_from_running,
    categorical_running_mass,
    gather_particles,
    label_logsumexp,
)

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize("dimension", [3, 8, 9, 16])
@pytest.mark.parametrize("gradient_enabled", [False, True])
def test_vector_tail_matches_unfused(dimension, gradient_enabled):
    rng = np.random.default_rng(75)

    def normal(shape):
        return jnp.asarray(rng.normal(size=shape).astype(np.float32))

    p, k = 7, 3
    x, prior = normal((p, dimension)), normal((p, dimension))
    chol = jnp.exp(normal((p, dimension)) * 0.2)
    correction = normal((p,)) * 0.1
    observed, parent, transition = normal((p, k)), normal((p, k)), normal((p, k))
    derivatives = normal((p, k, dimension))
    prior_labels = jax.nn.log_softmax(normal((k,)))
    reference_obs, reference_gradient = (
        normal((k,)),
        normal((dimension,)) if gradient_enabled else jnp.zeros(dimension),
    )
    variance = jnp.exp(normal((dimension,)) * 0.1) * 0.4
    mask = jnp.arange(dimension) % 3 != 1
    noise = _killing_noise(jax.random.key(31), p, x.dtype)
    actual = jax.jit(
        lambda: diagonal_mixture_tail(
            x,
            prior,
            chol,
            1 / chol,
            correction,
            observed,
            derivatives,
            parent,
            transition,
            prior_labels,
            reference_obs[None],
            reference_gradient[None],
            variance[None],
            mask[None],
            jax.tree.map(lambda value: value[None], noise),
            jnp.asarray(0),
            jnp.cumsum(jnp.exp(prior_labels)),
            jnp.full((1, x.shape[0]), 0.37),
            compute_gradient=gradient_enabled,
        )
    )()
    obs = observed.at[0].set(reference_obs)
    grad = (
        jnp.sum(jax.nn.softmax(obs + prior_labels, axis=-1)[..., None] * derivatives, axis=1)
        .at[0]
        .set(reference_gradient)
        if gradient_enabled
        else jnp.zeros_like(x)
    )
    geometry = _DiagonalProposalGeometry(
        jnp.zeros_like(x), jnp.zeros_like(x), chol, 1 / chol, jnp.zeros(p)
    )
    bank = MarginalMixtureBank(x, prior, prior, geometry, correction)
    raw = parent + transition + obs
    increment = label_logsumexp(raw)
    weights = increment + bank.log_weights(grad, variance, mask)
    scores = raw - increment[:, None]
    expected = (
        weights,
        scores,
        obs,
        _killing_reference(weights, noise),
        categorical_running_mass(raw),
        categorical_from_running(
            jnp.where(
                jnp.arange(p)[:, None] == 0,
                jnp.cumsum(jnp.exp(prior_labels)),
                gather_particles(categorical_running_mass(raw), _killing_reference(weights, noise)),
            ),
            jnp.full((p,), 0.37),
        ),
    )
    prepared_scores = gather_particles(scores, expected[3])
    prepared_correction = prior_labels[expected[5]] - prepared_scores[jnp.arange(p), expected[5]]
    expected = (*expected, prepared_scores, prepared_correction)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=2e-4, rtol=3e-5)
