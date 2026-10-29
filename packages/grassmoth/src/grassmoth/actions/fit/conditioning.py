"""Condition one DynamicalModelSpec and derive numerical draws from its current joint law."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.probability.serialization import empirical_distribution
from grassmoth.models.ssm import compiled_access as numeric
from grassmoth.models.ssm.compile.bindings import joint_law_layout

if TYPE_CHECKING:
    from datetime import datetime

    from jax.typing import ArrayLike

    from grassmoth.artifacts.identity import DistributionId
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
    from grassmoth.models.ssm.inference.types import MCMCPosterior


def condition_model(
    dynamical_model_spec: DynamicalModelSpec,
    compiled_dynamical_model: CompiledDynamicalModel,
    result: MCMCPosterior,
    *,
    times: ArrayLike,
    time_origin: datetime,
) -> tuple[DynamicalModelSpec, DistributionId]:
    """Replace the current uncertainty with the engine's full joint empirical law.

    The original distributions survive in the input revision. No run metadata,
    posterior containers, compiler bindings, or independent fitted marginals
    are attached to the scientific model.
    """
    bindings = compiled_dynamical_model.bindings
    samples = result.get_samples()
    paths = result.draws.latent_paths
    state_ids = numeric.state_ids(compiled_dynamical_model)
    modeled_ids = [
        identity
        for identity in state_ids
        if dynamical_model_spec.get_construct(identity).role == "endogenous"
    ]
    grid = np.asarray(times)
    if paths is None or paths.shape[1:] != (len(grid), len(state_ids)):
        raise ValueError("Conditioning must retain the complete aligned latent trajectories")
    if result.draws.state_ids and tuple(result.draws.state_ids) != tuple(state_ids):
        raise ValueError("Engine latent trajectories do not match the model's state identities")
    conditioned_parameters = {binding.parameter_id for binding in bindings}
    layout = joint_law_layout(
        bindings,
        parameters=conditioned_parameters,
        constructs=modeled_ids,
        time_points=grid.tolist(),
        time_origin=time_origin,
    )
    # Relative input paths acquire the same calendar binding as the conditioned states.
    input_layouts = {
        key: value.revised(time_origin=time_origin)
        for key, value in dynamical_model_spec.law_layouts.items()
        if value.time_origin == "relative"
        and value.constructs
        and all(
            dynamical_model_spec.get_construct(member).role == "exogenous"
            for member in value.constructs
        )
    }
    joint = layout.pack(
        {
            identity: samples[coordinate.site_name][(slice(None), *coordinate.indices)]
            for binding in bindings
            for identity, coordinate in binding.coordinates.items()
        },
        {identity: paths[:, :, index] for index, identity in enumerate(state_ids)},
    )
    law = empirical_distribution(joint)
    identity = layout.distribution_id
    constructs = tuple(
        construct.with_distribution(
            identity
            if construct.id in modeled_ids
            else input_layouts[construct.distribution].distribution_id
            if construct.distribution is not None and construct.distribution in input_layouts
            else construct.distribution
        )
        for construct in dynamical_model_spec.constructs
    )
    # Constructs without edges, such as a lone latent, are retained through `constructs`.
    endpoints = {
        endpoint.id for edge in dynamical_model_spec.edges for endpoint in (edge.cause, edge.effect)
    }
    edges = replace_constructs(
        dynamical_model_spec.edges,
        tuple(construct for construct in constructs if construct.id in endpoints),
    )
    parameters = tuple(
        parameter if parameter.id not in conditioned_parameters else parameter.conditioned(identity)
        for parameter in dynamical_model_spec.parameters
    )
    retained_laws = {
        member.distribution
        for member in (
            *parameters,
            *constructs,
        )
        if member.distribution is not None
    }
    conditioned = dynamical_model_spec.with_entities(
        constructs=constructs,
        edges=edges,
        parameters=parameters,
        distributions={
            **{
                input_layouts[key].distribution_id if key in input_layouts else key: value
                for key, value in dynamical_model_spec.distributions.items()
                if (input_layouts[key].distribution_id if key in input_layouts else key)
                in retained_laws
            },
            identity: law,
        },
        law_layouts={
            **{
                input_layouts[key].distribution_id
                if key in input_layouts
                else key: input_layouts.get(key, value)
                for key, value in dynamical_model_spec.law_layouts.items()
                if (input_layouts[key].distribution_id if key in input_layouts else key)
                in retained_laws
            },
            identity: layout,
        },
    )
    return conditioned, identity
