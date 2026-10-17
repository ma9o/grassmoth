"""Resolve available point readings to their direct scientific state dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.model.expressions import expression_states

if TYPE_CHECKING:
    import jax

    from nof1_causal_lab.models.ssm.compile.inputs import CompiledDynamicalModel


def compile_point_reading_mask(
    model: CompiledDynamicalModel, observations: jax.Array
) -> jax.Array:
    """Mark direct dependencies only; an accumulator reading never expands to its drivers."""
    state_indices = {state.id: index for index, state in enumerate(model.states)}
    observed = np.asarray(observations)
    mask = np.zeros((len(observed), len(model.states)), dtype=bool)
    for channel, observation in enumerate(model.observations):
        if observation.support.support_kind != "point":
            continue
        dependencies = {
            state_indices[identity]
            for _, operand in observation.law.operands()
            for identity in expression_states(operand.expression)
        }
        for index in dependencies:
            mask[:, index] |= ~np.isnan(observed[:, channel])
    return jnp.asarray(mask)
