"""Assemble execution tensors from aligned scientific parameter draws."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from nof1_causal_lab.models.ssm.parameterization import (
    assemble_deterministics_from_registry,
    build_site_registry,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.models.ssm.compile.inputs import CompiledDynamicalModel


def assemble_parameter_draws(
    compiled_dynamical_model: CompiledDynamicalModel,
    parameters: Mapping[str, jnp.ndarray],
    *,
    count: int,
) -> dict[str, jnp.ndarray]:
    """Assemble native tensors from aligned scientific parameter coordinates."""
    bindings, auxiliary = parameter_bindings(compiled_dynamical_model)
    expected = {identity for binding in bindings for identity in binding.coordinates}
    if set(parameters) != expected:
        raise ValueError("Draws do not match the scientific parameters of this DynamicalModelSpec")
    registry = build_site_registry(compiled_dynamical_model)
    # Padding has no scientific interpretation and is never read by an emission.
    # Its canonical completion is zero; every active coordinate is filled below.
    samples: dict[str, jnp.ndarray] = {}
    for site in registry:
        values = [
            parameters[identity]
            for binding in bindings
            for identity, coordinate in binding.coordinates.items()
            if coordinate.site_name == site.name
        ]
        dtype = jnp.result_type(*values) if values else jnp.float32
        samples[site.name] = jnp.zeros((count, *site.shape), dtype=dtype)
    covered = set(auxiliary)
    for binding in bindings:
        for identity, coordinate in binding.coordinates.items():
            samples[coordinate.site_name] = (
                samples[coordinate.site_name]
                .at[(slice(None), *coordinate.indices)]
                .set(parameters[identity])
            )
            covered.add(coordinate)
    from itertools import product

    from nof1_causal_lab.artifacts.model.parameter import ParameterCoordinate

    all_coordinates = {
        ParameterCoordinate(site_name=site.name, indices=indices)  # pyright: ignore[reportUnhashable] - Pydantic generates __hash__ from Value's inherited frozen configuration.
        for site in registry
        for indices in product(*(range(size) for size in site.shape))
    }
    if covered != all_coordinates:
        raise ValueError("Scientific draws must cover every active native coordinate")
    samples.update(
        assemble_deterministics_from_registry(samples, compiled_dynamical_model, n_draws=count)
    )
    return samples
