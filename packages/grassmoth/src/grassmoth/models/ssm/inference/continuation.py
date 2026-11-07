"""Native state round trips and transferable starts without executable serialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.continuation import ParticleTuning
from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure

if TYPE_CHECKING:
    from grassmoth.models.ssm.inference.warmup.initialization import ParticleInitialization


def retain_state(state: object) -> tuple[NumericalArray, ...]:
    """Retain numerical leaves only; code and tree structure stay with the kernel owner."""
    return tuple(
        NumericalArray.from_numpy(
            np.asarray(
                jax.random.key_data(value)
                if jnp.issubdtype(value.dtype, jax.dtypes.prng_key)
                else value
            )
        )
        for value in jax.tree.leaves(jax.tree.map(jnp.asarray, state))
    )


def restore_state[State](
    template: State, leaves: tuple[NumericalArray, ...]
) -> State | InitializationFailure:
    """Restore only the current native leaf shapes; incompatible native states are unsupported."""
    originals, structure = jax.tree.flatten(jax.tree.map(jnp.asarray, template))
    expected = tuple(
        jax.random.key_data(value) if jnp.issubdtype(value.dtype, jax.dtypes.prng_key) else value
        for value in originals
    )
    if len(expected) != len(leaves) or any(
        original.shape != np.asarray(saved.values).shape
        for original, saved in zip(expected, leaves, strict=True)
    ):
        return InitializationFailure.unavailable(
            "Native continuation state is incompatible with this runtime's state shape. Start a new fit."
        )
    restored = tuple(
        jax.random.wrap_key_data(jnp.asarray(saved.values), impl=jax.random.key_impl(original))
        if jnp.issubdtype(original.dtype, jax.dtypes.prng_key)
        else jnp.asarray(saved.values, dtype=original.dtype)
        for original, saved in zip(originals, leaves, strict=True)
    )
    return cast("State", jax.tree.unflatten(structure, restored))


def frozen_particle_tuning(initialization: ParticleInitialization) -> ParticleTuning:
    """Retain only geometry used by subsequent transitions, not an initialization checkpoint."""
    return ParticleTuning(
        parameter_chol=NumericalArray.from_numpy(np.asarray(initialization.parameter_chol)),
        parameter_reference_path=NumericalArray.from_numpy(np.asarray(initialization.paths[0])),
        initialization=initialization.diagnostics,
    )
