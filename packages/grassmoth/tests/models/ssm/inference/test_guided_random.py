"""GPU counter mapping and transforms against JAX's native Threefry stream."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.models.ssm.inference.samplers.particle.guided_random import (
    PARTICLE_PRNG,
    _killing_draws_reference,
    _killing_pallas,
    _point_pallas,
    killing_draws,
    particle_gumbel,
    particle_normal,
    particle_uniform,
)

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize("shape", [(1,), (127, 1), (127, 3), (3000, 1), (0, 2)])
@pytest.mark.parametrize("kind", ["uniform", "normal", "gumbel"])
def test_point_noise_matches_native(shape, kind):
    keys = jax.random.split(jax.random.key(917, impl=PARTICLE_PRNG), (3, 5))
    shader = partial(
        _point_pallas,
        shape=shape,
        dtype=jnp.float32,
        kind=kind,
        interpret=jax.default_backend() == "cpu",
    )
    actual = jax.jit(jax.vmap(jax.vmap(shader)))(keys)
    expected = jax.jit(
        jax.vmap(jax.vmap(lambda key: getattr(jax.random, kind)(key, shape, dtype=jnp.float32)))
    )(keys)
    if kind == "uniform":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=5e-6)


@pytest.mark.parametrize("count", [1, 7, 128, 257])
def test_killing_noise_matches_native(count):
    keys = jax.random.split(jax.random.key(817, impl=PARTICLE_PRNG), (3, 5))
    shader = partial(
        _killing_pallas, count=count, dtype=jnp.float32, interpret=jax.default_backend() == "cpu"
    )
    actual = jax.jit(jax.vmap(jax.vmap(shader)))(keys)
    expected = jax.jit(
        jax.vmap(jax.vmap(partial(_killing_draws_reference, count=count, dtype=jnp.float32)))
    )(keys)
    for a, b in zip(actual[:2], expected[:2], strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(actual[2], expected[2], atol=5e-6, rtol=5e-6)


@pytest.mark.parametrize("draw", [particle_uniform, particle_normal, particle_gumbel])
def test_particle_noise_empty_batch(draw):
    actual = jax.jit(jax.vmap(lambda key: draw(key, (7,), jnp.float32)))(
        jax.random.split(jax.random.key(9, impl=PARTICLE_PRNG), 0)
    )
    assert actual.shape == (0, 7)


def test_cpu_noise_preserves_requested_dtype():
    if jax.default_backend() != "cpu":
        pytest.skip("Float64 is a CPU-only contract; all GPU computation is float32")
    with jax.enable_x64():
        key = jax.random.key(71, impl=PARTICLE_PRNG)
        for actual, expected in zip(
            killing_draws(key, 7, jnp.float64),
            _killing_draws_reference(key, 7, jnp.float64),
            strict=True,
        ):
            assert actual.dtype == jnp.float64
            np.testing.assert_array_equal(actual, expected)
        for draw, native in [
            (particle_uniform, jax.random.uniform),
            (particle_normal, jax.random.normal),
            (particle_gumbel, jax.random.gumbel),
        ]:
            actual = draw(key, (13,), jnp.float64)
            assert actual.dtype == jnp.float64
            np.testing.assert_array_equal(actual, native(key, (13,), jnp.float64))


def test_native_continuation_preserves_owned_key_implementation():
    from grassmoth.models.ssm.inference.continuation import restore_state, retain_state

    template = (jax.random.key(991, impl=PARTICLE_PRNG), jnp.array([1.0, 2.0]))
    saved = retain_state(template)
    with jax.default_prng_impl("rbg"):
        restored = restore_state(template, saved)
        assert isinstance(restored, tuple)
        assert jax.random.key_impl(restored[0]) == jax.random.key_impl(template[0])
        np.testing.assert_array_equal(
            jax.random.key_data(restored[0]), jax.random.key_data(template[0])
        )
        np.testing.assert_array_equal(
            jax.random.uniform(restored[0], (13,)), jax.random.uniform(template[0], (13,))
        )
