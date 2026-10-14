"""Draw compiler-resolved scientific laws while preserving their joint coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
from nof1_causal_lab.models.ssm.parameter_draws import assemble_parameter_draws

if TYPE_CHECKING:
    import jax

    from nof1_causal_lab.models.ssm.compile.inputs import CompiledDynamicalModel


def sample_model_laws(
    compiled_dynamical_model: CompiledDynamicalModel, *, draws: int, key: jax.Array
) -> JointPosteriorDraws:
    """Sample every active law once, retaining the compiler's joint event order."""
    values: dict[str, jnp.ndarray] = {}
    paths: dict[str, jnp.ndarray] = {}
    for law in compiled_dynamical_model.laws:
        shape = (
            (draws,)
            if law.distribution.batch_shape or law.distribution.event_shape
            else (draws, law.layout.width)
        )
        sampled = law.distribution.sample(random.fold_in(key, law.sample_index), sample_shape=shape)
        parameter_draws, trajectories = law.layout.unpack(jnp.asarray(sampled))
        values.update(parameter_draws.items())
        paths.update(trajectories.items())
    state_ids = tuple(state.id for state in compiled_dynamical_model.states if not state.is_input)
    return JointPosteriorDraws(
        parameters=assemble_parameter_draws(compiled_dynamical_model, values, count=draws),
        latent_paths=jnp.stack([paths[identity] for identity in state_ids], axis=-1)
        if paths
        else None,
        state_ids=state_ids,
    )
