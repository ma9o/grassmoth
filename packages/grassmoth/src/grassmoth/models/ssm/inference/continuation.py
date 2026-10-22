"""Native state round trips and transferable starts without executable serialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.continuation import ParticleTuning

if TYPE_CHECKING:
    from grassmoth.models.ssm.inference.targets.gaussian_reference import GaussianReference
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


def restore_state[State](template: State, leaves: tuple[NumericalArray, ...]) -> State:
    """Reconstruct a kernel's native tree using its freshly bound structure."""
    originals, structure = jax.tree.flatten(jax.tree.map(jnp.asarray, template))
    restored = tuple(
        jax.random.wrap_key_data(jnp.asarray(saved.values))
        if jnp.issubdtype(original.dtype, jax.dtypes.prng_key)
        else jnp.asarray(saved.values, dtype=original.dtype)
        for original, saved in zip(originals, leaves, strict=True)
    )
    return cast("State", jax.tree.unflatten(structure, restored))


def frozen_particle_tuning(
    initialization: ParticleInitialization,
    reference: GaussianReference | None,
) -> ParticleTuning:
    """Retain only geometry used by subsequent transitions, not an initialization checkpoint."""
    return ParticleTuning(
        parameter_chol=NumericalArray.from_numpy(np.asarray(initialization.parameter_chol)),
        pilot_mean=NumericalArray.from_numpy(np.asarray(initialization.pilot_mean)),
        pilot_variance=NumericalArray.from_numpy(np.asarray(initialization.pilot_variance)),
        parameter_reference_path=NumericalArray.from_numpy(np.asarray(initialization.paths[0])),
        gaussian_reference=()
        if reference is None
        else retain_state(
            (
                reference.parameter_mean,
                reference.parameter_chol,
                reference.path_mean,
                reference.autoregression,
                reference.innovation_chol,
                reference.free_mask,
            )
        ),
        initialization=initialization.diagnostics,
    )
