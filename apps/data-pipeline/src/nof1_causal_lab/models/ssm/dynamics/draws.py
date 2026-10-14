"""Batched vector-field parameters reconstructed from canonical numerical sites."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.immutability import freeze_fields
from nof1_causal_lab.models.ssm.dynamics.spec import (
    compile_dynamics,
    pack_component_params_from_samples,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jax import Array

    from nof1_causal_lab.models.ssm.compile.inputs import CompiledDynamicalModel
    from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorField


@dataclass(frozen=True)
class DynamicsDraws:
    """One vector field and component parameters with a shared leading draw axis.

    The explicit count also describes fully fixed fields, whose parameter pytree
    has no leaves for ``vmap`` to infer a batch size from.
    """

    vector_field: VectorField
    parameters: tuple[Mapping[str, Array], ...]
    n_draws: int

    def __post_init__(self) -> None:
        """Freeze component parameter mappings and require a common leading draw axis."""
        object.__setattr__(
            self,
            "parameters",
            tuple(MappingProxyType(dict(component)) for component in self.parameters),
        )
        if self.n_draws < 0:
            raise ValueError("Dynamics draw count must be non-negative")
        if len(self.parameters) != len(self.vector_field.components):
            raise ValueError("Dynamics parameters must match the vector-field components")
        for component in self.parameters:
            for name, values in component.items():
                if values.ndim < 1 or values.shape[0] != self.n_draws:
                    raise ValueError(f"{name} must share the dynamics draw axis")
        freeze_fields(self)


def dynamics_from_samples(
    compiled_dynamical_model: CompiledDynamicalModel,
    samples: Mapping[str, Array],
    *,
    n_draws: int,
    prefix: str = "vf",
) -> DynamicsDraws:
    """Pack already batched site arrays without unstacking individual draws."""
    dynamics = compiled_dynamical_model.dynamics.spec
    return DynamicsDraws(
        vector_field=compile_dynamics(dynamics, prefix=prefix).vector_field,
        parameters=pack_component_params_from_samples(dynamics, samples, prefix=prefix),
        n_draws=n_draws,
    )
