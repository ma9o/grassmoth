"""Resolve available point readings to their direct scientific state dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    import jax

    from nof1_causal_lab.models.ssm.compile.inputs import CompiledDynamicalModel


def compile_point_reading_mask(model: CompiledDynamicalModel, observations: jax.Array) -> jax.Array:
    """Mark direct dependencies only; an accumulator reading never expands to its drivers."""
    # The compiled predictor drops fixed zero loadings while retaining every
    # free loading, independently of its current numerical parameter value.
    dependencies = (
        np.asarray(model.loading_block.template) != 0
    ) | model.loading_block.free_support
    observed = np.asarray(observations)
    mask = np.zeros((len(observed), len(model.states)), dtype=bool)
    for channel, observation in enumerate(model.observations):
        if observation.support.support_kind != "point":
            continue
        for index in np.flatnonzero(dependencies[channel]):
            mask[:, index] |= ~np.isnan(observed[:, channel])
    return jnp.asarray(mask)
