"""Differential checks for the fused kernels, including batching and padded axes."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from grassmoth.models.ssm.inference.samplers.particle.guided import (
    _killing_fused,
    _killing_noise,
    _killing_reference,
)
from grassmoth.models.ssm.inference.samplers.particle.guided_selection import (
    _backward_from_moments,
)
from grassmoth.models.ssm.inference.samplers.particle.math import gather_particles, label_logsumexp

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize("axes", [(0, 0), (0, None), (None, 0)])
def test_flat_particle_gather_preserves_batched_indices(axes):
    values = jnp.arange(2 * 3 * 7 * 4).reshape(2, 3, 7, 4)
    indices = jnp.arange(2 * 3 * 7).reshape(2, 3, 7) % 7
    source = values[0] if axes[0] == 0 else values[0, 0]
    selected = indices[0] if axes[1] == 0 else indices[0, 0]
    actual = jax.jit(jax.vmap(gather_particles, in_axes=axes))(source, selected)
    expected = jax.jit(jax.vmap(lambda a, b: a[b], in_axes=axes))(source, selected)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(jax.vmap(gather_particles)))(values, indices),
        jax.jit(jax.vmap(jax.vmap(lambda a, b: a[b])))(values, indices),
    )


@pytest.mark.parametrize("count", [7, 16, 128, 512])
def test_fused_killing_preserves_conditional_ancestors(count):
    keys = jax.random.split(jax.random.key(38), 64)
    noise = jax.vmap(lambda key: _killing_noise(key, count, jnp.float32))(keys)
    logits = 3 * jax.random.normal(jax.random.key(29), (64, count))
    logits = logits.at[0].set(0.0)
    logits = logits.at[1].set(-jnp.inf).at[1, 0].set(0.0)
    logits = logits.at[2].set(jnp.linspace(-1000.0, 0.0, count))

    if count == 512:
        # Interval interiors avoid float32 CDF-rounding ambiguity at this larger size.
        cumulative = jnp.cumsum(jax.nn.softmax(logits), axis=-1)
        ranks = jax.vmap(jnp.searchsorted)(cumulative, noise.replacement)
        left = jnp.take_along_axis(
            jnp.concatenate((jnp.zeros((64, 1)), cumulative), axis=-1), ranks, axis=-1
        )
        right = jnp.take_along_axis(cumulative, jnp.minimum(ranks, count - 1), axis=-1)
        noise = noise._replace(replacement=(left + right) / 2)

    expected = jax.jit(jax.vmap(_killing_reference))(logits, noise)
    actual = jax.jit(jax.vmap(partial(_killing_fused, interpret=jax.default_backend() == "cpu")))(
        logits, noise
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:, 0], 0)


@pytest.mark.parametrize(
    ("steps", "count", "labels", "dimension", "tied"),
    [
        (1, 7, 3, 2, False),
        (5, 7, 3, 0, False),
        (5, 7, 3, 2, False),
        (17, 16, 8, 1, False),
        (7, 7, 3, 3, False),
        (5, 7, 3, 40, False),
        (4, 1, 1, 2, False),
        (5, 7, 3, 2, True),
    ],
)
@pytest.mark.parametrize("location", [0, 2**15])
@pytest.mark.parametrize("diagonal", [False, True])
def test_backward_from_moments_preserves_exact_selection_and_suffix(
    steps, count, labels, dimension, tied, location, diagonal
):
    rng = np.random.default_rng(320)
    chains = 2
    means = rng.normal(size=(chains, steps - 1, count, labels, dimension)).astype(np.float32)
    points = rng.normal(size=(chains, steps - 1, count, dimension)).astype(np.float32)
    means += location
    points += location
    whiteners = rng.normal(size=(chains, steps - 1, labels, dimension, dimension)).astype(
        np.float32
    )
    if diagonal:
        whiteners *= np.eye(dimension, dtype=np.float32)
    normalizers = rng.normal(size=(chains, steps - 1, labels)).astype(np.float32)
    weights = rng.normal(size=(chains, steps, count)).astype(np.float32)
    scores = rng.normal(size=(chains, steps, count, labels)).astype(np.float32)
    observations = rng.normal(size=scores.shape).astype(np.float32)
    gumbels = rng.gumbel(size=(chains, steps - 1, count)).astype(np.float32)
    if tied:
        for value in (
            means,
            points,
            whiteners,
            normalizers,
            weights,
            scores,
            observations,
            gumbels,
        ):
            value.fill(0)
    elif labels > 1:
        scores[..., -1] = -np.inf
        observations[..., -1] = -np.inf
    last = np.arange(chains, dtype=np.int32) % count
    expected_picks = np.empty((chains, steps), dtype=np.int32)
    expected_suffix = np.empty((chains, labels), dtype=np.float32)
    for chain in range(chains):
        following = last[chain]
        expected_picks[chain, -1] = following
        suffix = observations[chain, -1, following].copy()
        for index in range(steps - 2, -1, -1):
            residual = points[chain, index, following][None, None] - means[chain, index]
            white = np.sum(whiteners[chain, index][None] * residual[:, :, None, :], axis=-1)
            edge = normalizers[chain, index][None] - 0.5 * np.sum(white * white, axis=-1)
            logits = weights[chain, index] + logsumexp(
                scores[chain, index] + edge + suffix, axis=-1
            )
            following = np.argmax(logits + gumbels[chain, index])
            expected_picks[chain, index] = following
            suffix = suffix + edge[following] + observations[chain, index, following]
        expected_suffix[chain] = suffix
    actual = jax.jit(jax.vmap(_backward_from_moments))(
        *(
            jnp.asarray(value)
            for value in (
                weights,
                means,
                points,
                np.diagonal(whiteners, axis1=-2, axis2=-1) if diagonal else whiteners,
                normalizers,
                scores,
                observations,
                gumbels,
                last,
            )
        )
    )
    np.testing.assert_array_equal(actual[0], expected_picks)
    np.testing.assert_allclose(actual[1], expected_suffix, atol=5e-6, rtol=5e-6)


@pytest.mark.parametrize("labels", [1, 2, 3, 8, 17])
def test_small_label_reduction_preserves_zero_mass_and_large_log_densities(labels):
    values = np.array(
        [[-np.inf, -np.inf, -np.inf], [1000.0, 1001.0, -1000.0], [0.0, -np.inf, 0.0]],
        dtype=np.float32,
    )
    values = np.pad(values, ((0, 0), (0, max(0, labels - 3))), constant_values=-np.inf)[:, :labels]
    np.testing.assert_allclose(
        jax.jit(label_logsumexp)(jnp.asarray(values)),
        logsumexp(values, axis=-1),
        atol=2e-6,
        rtol=2e-6,
    )
    actual_gradient = jax.jit(jax.grad(lambda x: jnp.sum(label_logsumexp(x))))(values[1:])
    expected_gradient = jax.grad(lambda x: jnp.sum(jax.scipy.special.logsumexp(x, axis=-1)))(
        values[1:]
    )
    np.testing.assert_allclose(actual_gradient, expected_gradient, atol=2e-6, rtol=2e-6)
