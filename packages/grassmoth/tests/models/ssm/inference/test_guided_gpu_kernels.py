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
from grassmoth.models.ssm.inference.samplers.particle.guided_selection import _cached_backward
from grassmoth.models.ssm.inference.samplers.particle.math import label_logsumexp

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize("count", [7, 16, 128])
def test_fused_killing_preserves_conditional_ancestors(count):
    keys = jax.random.split(jax.random.key(38), 64)
    noise = jax.vmap(lambda key: _killing_noise(key, count, jnp.float32))(keys)
    logits = 3 * jax.random.normal(jax.random.key(29), (64, count))
    logits = logits.at[0].set(0.0)
    logits = logits.at[1].set(-jnp.inf).at[1, 0].set(0.0)
    logits = logits.at[2].set(jnp.linspace(-1000.0, 0.0, count))

    expected = jax.jit(jax.vmap(_killing_reference))(logits, noise)
    actual = jax.jit(jax.vmap(partial(_killing_fused, interpret=True)))(logits, noise)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:, 0], 0)


@pytest.mark.parametrize(
    ("steps", "count", "labels", "tied"),
    [
        (1, 7, 3, False),
        (5, 7, 3, False),
        (17, 16, 2, False),
        (4, 128, 1, False),
        (4, 1, 3, False),
        (5, 7, 3, True),
    ],
)
def test_cached_backward_preserves_selection_and_complete_label_suffix(steps, count, labels, tied):
    rng = np.random.default_rng(320)
    weights = rng.normal(size=(2, steps, count)).astype(np.float32)
    links = rng.normal(size=(2, steps - 1, count, count, labels)).astype(np.float32)
    scores = rng.normal(size=(2, steps, count, labels)).astype(np.float32)
    observations = rng.normal(size=(2, steps, count, labels)).astype(np.float32)
    gumbels = rng.gumbel(size=(2, steps - 1, count)).astype(np.float32)
    if tied:
        for value in (weights, links, scores, observations, gumbels):
            value.fill(0)
    elif labels > 1:
        scores[..., -1] = -np.inf
        observations[..., -1] = -np.inf
    last = np.arange(2, dtype=np.int32) % count

    expected_picks = np.empty((2, steps), dtype=np.int32)
    expected_suffix = np.empty((2, labels), dtype=np.float32)
    for chain in range(2):
        following = last[chain]
        expected_picks[chain, -1] = following
        suffix = observations[chain, -1, following].copy()
        for index in range(steps - 2, -1, -1):
            edge = links[chain, index, following]
            logits = weights[chain, index] + logsumexp(
                scores[chain, index] + edge + suffix, axis=-1
            )
            following = np.argmax(logits + gumbels[chain, index])
            expected_picks[chain, index] = following
            suffix = suffix + edge[following] + observations[chain, index, following]
        expected_suffix[chain] = suffix

    select = jax.jit(jax.vmap(_cached_backward))
    picks, suffix = select(
        *(jnp.asarray(v) for v in (weights, links, scores, observations, gumbels, last))
    )
    np.testing.assert_array_equal(picks, expected_picks)
    np.testing.assert_allclose(suffix, expected_suffix, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("labels", [1, 2, 3])
def test_small_label_reduction_preserves_zero_mass_and_large_log_densities(labels):
    values = np.array(
        [[-np.inf, -np.inf, -np.inf], [1000.0, 1001.0, -1000.0], [0.0, -np.inf, 0.0]],
        dtype=np.float32,
    )[:, :labels]
    np.testing.assert_allclose(
        jax.jit(label_logsumexp)(jnp.asarray(values)),
        logsumexp(values, axis=-1),
        atol=2e-6,
        rtol=2e-6,
    )
