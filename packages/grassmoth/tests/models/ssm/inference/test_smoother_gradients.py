"""Parameter batching preserves the chosen transition derivative under JIT."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.models.ssm.inference.samplers.particle.math import (
    _value_and_grad_by_param,
)

pytestmark = pytest.mark.inference(concern="sampling")


def test_parameter_batching_preserves_transition_argument_and_output_dtypes() -> None:
    contexts = {"scale": jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)}
    previous = jnp.array([0.5, 1.0])
    current = jnp.array([2.0, 3.0])
    following = jnp.array([5.0, 8.0])

    def _transition(context, left, right):
        return -0.5 * jnp.sum(context["scale"] * jnp.square(right - 2 * left))

    def _evaluate(left, value, right):
        current_terms = _value_and_grad_by_param(
            contexts,
            value,
            lambda context, particle: _transition(context, left, particle),
            value_dtype=jnp.float32,
            grad_dtype=jnp.float16,
        )
        next_terms = _value_and_grad_by_param(
            contexts,
            value,
            lambda context, particle: _transition(context, particle, right),
            value_dtype=jnp.float32,
            grad_dtype=jnp.float16,
        )
        return current_terms, next_terms

    (current_value, current_grad), (next_value, next_grad) = jax.jit(_evaluate)(
        previous, current, following
    )
    scale = np.asarray(contexts["scale"])
    current_residual = np.asarray(current - 2 * previous)
    next_residual = np.asarray(following - 2 * current)
    np.testing.assert_allclose(current_value, -0.5 * np.sum(scale * current_residual**2, axis=1))
    np.testing.assert_allclose(next_value, -0.5 * np.sum(scale * next_residual**2, axis=1))
    np.testing.assert_allclose(current_grad, -scale * current_residual)
    np.testing.assert_allclose(next_grad, 2 * scale * next_residual)
    assert current_value.dtype == next_value.dtype == jnp.float32
    assert current_grad.dtype == next_grad.dtype == jnp.float16
