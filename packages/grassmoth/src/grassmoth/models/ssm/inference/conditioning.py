"""Compile exact point measurements into fixed coordinates of the sampled path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from grassmoth.artifacts.model.expressions import StateExpression
from grassmoth.artifacts.model.likelihood import DeltaLawSpec
from grassmoth.models.ssm import compiled_access as numeric

if TYPE_CHECKING:
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel


@dataclass(frozen=True)
class ExactStateConstraints:
    """Observed state values on the model grid; NaN coordinates remain unknown."""

    values: jnp.ndarray

    @property
    def free_mask(self) -> jnp.ndarray:
        """State coordinates not fixed by exact observations, identified by NaN constraint values."""
        return jnp.isnan(self.values)

    def project(self, path: jnp.ndarray) -> jnp.ndarray:
        """Substitute observations in one path or a batch of initialization paths."""
        if path.shape[-2:] != self.values.shape:
            raise ValueError(
                "Exact observations and latent paths must use the same time/state grid"
            )
        return jnp.where(self.free_mask, path, self.values)


def compile_exact_state_constraints(
    compiled_dynamical_model: CompiledDynamicalModel,
    observations: jnp.ndarray,
    *,
    input_values: jnp.ndarray | None = None,
) -> ExactStateConstraints | None:
    """Condition direct state bindings without discarding their dynamics density."""
    exact = [
        (index, indicator)
        for index, indicator in enumerate(compiled_dynamical_model.observations)
        if isinstance(indicator.law, DeltaLawSpec)
    ]
    if not exact and input_values is None:
        return None
    observed = np.asarray(observations)
    values = np.full(
        (len(observed), numeric.n_states(compiled_dynamical_model)), np.nan, dtype=observed.dtype
    )
    state_indices = {
        identity: index
        for index, identity in enumerate(numeric.state_ids(compiled_dynamical_model))
    }
    for column, indicator in exact:
        if compiled_dynamical_model.states[indicator.state_index].is_input:
            if input_values is None:
                raise ValueError("Exogenous inputs require a replayed panel path")
            continue
        assert isinstance(indicator.law, DeltaLawSpec)
        expression = indicator.law.v.expression
        if indicator.support.support_kind != "point" or not isinstance(expression, StateExpression):
            raise ValueError(
                f"Delta indicator {indicator.name!r} requires a direct point binding "
                "Delta(v=state(...)) for particle inference; affine and interval constraints "
                "are not supported."
            )
        index = state_indices[expression.construct_id]
        readings = observed[:, column]
        if np.isinf(readings).any():
            raise ValueError(f"Exact observations for {indicator.name!r} must be finite or missing")
        present = ~np.isnan(readings)
        conflict = present & ~np.isnan(values[:, index]) & (readings != values[:, index])
        if conflict.any():
            raise ValueError(
                f"Conflicting exact observations for {expression.construct_id!r} "
                f"at model rows {np.flatnonzero(conflict).tolist()}"
            )
        values[present, index] = readings[present]
    if input_values is not None:
        inputs = numeric.input_mask(compiled_dynamical_model)
        values[:, inputs] = np.asarray(input_values)[:, inputs]
    for index, state in enumerate(compiled_dynamical_model.states):
        if state.time_invariant:
            readings = values[:, index][~np.isnan(values[:, index])]
            if readings.size:
                values[:, index] = readings[0]
    return ExactStateConstraints(jnp.asarray(values))
