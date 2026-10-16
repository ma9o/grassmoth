"""Native state round trips and transferable starts without executable serialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.arrays import NumericalArray
from nof1_causal_lab.artifacts.continuation import ParticleTuning, SamplerCheckpoint
from nof1_causal_lab.artifacts.reports.posterior_diagnostics import ContinuedInitializationDiagnostics
from nof1_causal_lab.models.ssm.inference.warmup.laps import LapsInitializationResult

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.targets.gaussian_reference import GaussianReference


def retain_state[State](state: State) -> tuple[NumericalArray, ...]:
    """Retain numerical leaves only; code and tree structure stay with the kernel owner."""
    return tuple(
        NumericalArray.from_numpy(np.asarray(
            jax.random.key_data(value) if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
            else value
        ))
        for value in jax.tree.leaves(jax.tree.map(jnp.asarray, state))
    )


def restore_state[State](template: State, leaves: tuple[NumericalArray, ...]) -> State:
    """Reconstruct a kernel's native tree using its freshly bound structure."""
    originals, structure = jax.tree.flatten(jax.tree.map(jnp.asarray, template))
    restored = tuple(
        jax.random.wrap_key_data(jnp.asarray(saved.values))
        if jax.dtypes.issubdtype(original.dtype, jax.dtypes.prng_key)
        else jnp.asarray(saved.values, dtype=original.dtype)
        for original, saved in zip(originals, leaves, strict=True)
    )
    return jax.tree.unflatten(structure, restored)


def frozen_particle_tuning(
    initialization: LapsInitializationResult,
    reference: GaussianReference | None,
) -> ParticleTuning:
    """Retain only geometry used by subsequent transitions, not an initialization checkpoint."""
    return ParticleTuning(
        parameter_chol=NumericalArray.from_numpy(np.asarray(initialization.parameter_chol)),
        pilot_mean=NumericalArray.from_numpy(np.asarray(initialization.pilot_mean)),
        pilot_variance=NumericalArray.from_numpy(np.asarray(initialization.pilot_variance)),
        parameter_reference_path=NumericalArray.from_numpy(np.asarray(initialization.paths[0])),
        gaussian_reference=() if reference is None else retain_state((
            reference.parameter_mean, reference.parameter_chol, reference.path_mean,
            reference.autoregression, reference.innovation_chol, reference.free_mask,
        )),
        initialization=initialization.diagnostics,
    )


def continued_particle_initialization(checkpoint: SamplerCheckpoint) -> LapsInitializationResult:
    """Start at retained values, using the particle kernel's saved proposal geometry."""
    tuning = checkpoint.tuning
    if not isinstance(tuning, ParticleTuning):
        positions = jnp.asarray(checkpoint.parameters.values)
        paths = jnp.asarray(checkpoint.paths.values)
        return LapsInitializationResult(
            positions=positions, paths=paths,
            parameter_chol=jnp.eye(positions.shape[-1], dtype=positions.dtype),
            pilot_mean=jnp.mean(paths, axis=0),
            pilot_variance=jnp.var(paths, axis=0),
            diagnostics=ContinuedInitializationDiagnostics(reused_tuning=False),
        )
    return LapsInitializationResult(
        positions=jnp.asarray(checkpoint.parameters.values),
        paths=jnp.asarray(checkpoint.paths.values),
        parameter_chol=jnp.asarray(tuning.parameter_chol.values),
        pilot_mean=jnp.asarray(tuning.pilot_mean.values),
        pilot_variance=jnp.asarray(tuning.pilot_variance.values),
        diagnostics=ContinuedInitializationDiagnostics(reused_tuning=True),
    )
