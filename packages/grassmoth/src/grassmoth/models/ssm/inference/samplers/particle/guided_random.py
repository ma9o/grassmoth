"""Counter-based particle draws without materialized, repeated GPU key banks."""

from __future__ import annotations

from functools import partial
from math import prod
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
from pydantic import Field

from grassmoth.models.ssm.inference.samplers.particle.math import TRITON_NUM_WARPS

if TYPE_CHECKING:
    from jax.typing import DTypeLike


class KillingNoise(NamedTuple):
    """Independent survival, replacement and cyclic-rotation draws for conditional killing."""

    survival: jax.Array
    replacement: jax.Array
    rotation: jax.Array


class TailRandomKeys(NamedTuple):
    """Per-time survival, replacement, rotation and label Threefry subkeys."""

    values: jax.Array
    reference: KillingNoise
    label_reference: jax.Array


MIXTURE_DRAW_FOLD_IN: Annotated[
    int,
    Field(
        description="Heuristic: fixed stream-separation tag 0x4D495854 for the mixture label draw, preserving the established proposal-key stream."
    ),
] = 0x4D495854


PARTICLE_PRNG: Annotated[
    Literal["threefry2x32"],
    Field(
        description="JAX's partitionable Threefry2x32-20 stream; explicit ownership keeps native particle kernels independent of the process default PRNG."
    ),
] = "threefry2x32"

RANDOM_TILE_SIZE: Annotated[
    int,
    Field(
        description="Heuristic: 256 adjacent counters per four-warp GPU tile, bounding registers while preserving coalesced stores."
    ),
] = 256


def _threefry(
    key0: jax.Array, key1: jax.Array, count0: jax.Array, count1: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Random123 Threefry2x32-20 (Salmon et al., SC11, section 4.2).

    The parity word, rotation schedule and five groups of four rounds are the
    Threefry2x32 specification, matching JAX's threefry2x32 implementation.
    All operands are uint32: wraparound and logical shifts are part of the law.
    """
    keys = (key0, key1, key0 ^ key1 ^ jnp.uint32(0x1BD11BDA))
    x, y = count0 + key0, count1 + key1
    for group in range(5):
        rotations = (13, 15, 26, 6) if group % 2 == 0 else (17, 29, 16, 24)
        for rotation in rotations:
            x = x + y
            y = ((y << jnp.uint32(rotation)) | (y >> jnp.uint32(32 - rotation))) ^ x
        x = x + keys[(group + 1) % 3]
        y = y + keys[(group + 2) % 3] + jnp.uint32(group + 1)
    return x, y


def _uniform_bits(high: jax.Array, low: jax.Array) -> jax.Array:
    """JAX float32 uniform: xor partitionable words, keep 23 mantissa bits, subtract one."""
    bits = ((high ^ low) >> jnp.uint32(9)) | jnp.uint32(0x3F800000)
    return jax.lax.bitcast_convert_type(bits, jnp.float32) - 1.0


def _gumbel(uniform: jax.Array) -> jax.Array:
    """JAX's inverse-CDF Gumbel transform on [float32.tiny, 1)."""
    return -jnp.log(-jnp.log(jnp.maximum(uniform, jnp.finfo(jnp.float32).tiny)))


def _killing_draws_reference(
    key: jax.Array, count: int, dtype: DTypeLike
) -> tuple[jax.Array, jax.Array, jax.Array]:
    survival, replacement, rotation = jax.random.split(key, 3)
    return (
        jax.random.uniform(survival, (count,), dtype=dtype),
        jax.random.uniform(replacement, (count,), dtype=dtype),
        jax.random.gumbel(rotation, (count,), dtype=dtype),
    )


def _killing_pallas(
    key: jax.Array, count: int, dtype: DTypeLike, *, interpret: bool = False
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fuse JAX's three-way split and two uniforms plus one Gumbel for float32 GPU work."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    block = pl.next_power_of_2(count)

    def kernel(key_ref, survival_ref, replacement_ref, rotation_ref):
        p = jnp.arange(block, dtype=jnp.uint32)
        k0, k1 = key_ref[0], key_ref[1]
        for component, out in enumerate((survival_ref, replacement_ref, rotation_ref)):
            a, b = _threefry(k0, k1, jnp.uint32(0), jnp.uint32(component))
            high, low = _threefry(a, b, jnp.zeros_like(p), p)
            value = _uniform_bits(high, low)
            if component == 2:
                value = _gumbel(value)
            pt.store(out.at[p], value.astype(dtype), mask=p < count)

    shape = jax.ShapeDtypeStruct((count,), dtype)
    return cast(
        "tuple[jax.Array, jax.Array, jax.Array]",
        pl.pallas_call(
            kernel,
            out_shape=(shape, shape, shape),
            interpret=interpret,
            compiler_params=pt.CompilerParams(num_warps=TRITON_NUM_WARPS),
            name="conditional_killing_threefry_noise",
        )(jax.random.key_data(key)),
    )


def killing_draws(key: jax.Array, count: int, dtype: DTypeLike) -> KillingNoise:
    """Preserve native CPU dtypes and the Threefry float32 GPU stream."""
    return KillingNoise(
        *cast(
            "tuple[jax.Array, jax.Array, jax.Array]",
            jax.lax.platform_dependent(
                key,
                cpu=partial(_killing_draws_reference, count=count, dtype=dtype),
                cuda=partial(_killing_pallas, count=count, dtype=dtype),
            ),
        )
    )


def _point_pallas(
    key: jax.Array,
    shape: tuple[int, ...],
    dtype: DTypeLike,
    *,
    kind: Literal["uniform", "normal", "gumbel"],
    interpret: bool = False,
) -> jax.Array:
    """Generate partitionable float32 counters directly at each output tile.

    The normal transform follows JAX: map to [nextafter(-1, 0), 1), then
    sqrt(2)*erf_inv. Its constants are float32 roundings of those exact values.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    size = prod(shape)
    if size == 0:
        return jnp.empty(shape, dtype=dtype)
    block = min(RANDOM_TILE_SIZE, pl.next_power_of_2(size))

    def kernel(key_ref, out):
        p = pl.program_id(0).astype(jnp.uint32) * jnp.uint32(block) + jnp.arange(
            block, dtype=jnp.uint32
        )
        high, low = _threefry(key_ref[0], key_ref[1], jnp.zeros_like(p), p)
        value = _uniform_bits(high, low)
        if kind == "gumbel":
            value = _gumbel(value)
        elif kind == "normal":
            lower = jnp.float32(-0.9999999403953552)
            value = jnp.maximum(lower, value * 2.0 + lower)
            value = jnp.float32(1.4142135623730951) * jax.lax.erf_inv(value)
        pt.store(out.at[p], value.astype(dtype), mask=p < size)

    return cast(
        "jax.Array",
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((size,), dtype),
            grid=(pl.cdiv(size, block),),
            interpret=interpret,
            compiler_params=pt.CompilerParams(num_warps=TRITON_NUM_WARPS),
            name="particle_threefry_" + kind,
        )(jax.random.key_data(key)).reshape(shape),
    )


def particle_uniform(key: jax.Array, shape: tuple[int, ...], dtype: DTypeLike) -> jax.Array:
    """Uniform particle noise, with the native CPU implementation and dtype."""
    return cast(
        "jax.Array",
        jax.lax.platform_dependent(
            key,
            cpu=partial(jax.random.uniform, shape=shape, dtype=dtype),
            cuda=partial(_point_pallas, shape=shape, dtype=dtype, kind="uniform"),
        ),
    )


def particle_normal(key: jax.Array, shape: tuple[int, ...], dtype: DTypeLike) -> jax.Array:
    """Normal particle noise, with the native CPU implementation and dtype."""
    return cast(
        "jax.Array",
        jax.lax.platform_dependent(
            key,
            cpu=partial(jax.random.normal, shape=shape, dtype=dtype),
            cuda=partial(_point_pallas, shape=shape, dtype=dtype, kind="normal"),
        ),
    )


def particle_gumbel(key: jax.Array, shape: tuple[int, ...], dtype: DTypeLike) -> jax.Array:
    """Gumbel particle noise, with the native CPU implementation and dtype."""
    return cast(
        "jax.Array",
        jax.lax.platform_dependent(
            key,
            cpu=partial(jax.random.gumbel, shape=shape, dtype=dtype),
            cuda=partial(_point_pallas, shape=shape, dtype=dtype, kind="gumbel"),
        ),
    )
