"""Ownership of probability laws that holds before any compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import (
        DynamicalModelSpec,
        _ModelEntities,
    )


def validate_distribution_memberships(
    dynamical_model_spec: DynamicalModelSpec | _ModelEntities,
) -> None:
    """Own scalar membership and retained joint identities before compilation."""
    import numpy as np
    import numpyro.distributions as dist

    from nof1_causal_lab.artifacts.probability.serialization import (
        distribution_shape,
        materialize_distribution,
    )

    joint = {
        identity
        for identity, law in dynamical_model_spec.distributions.items()
        if distribution_shape(law) != ((), ())
    }
    if set(dynamical_model_spec.law_layouts) != joint:
        raise ValueError("Each joint law requires exactly one retained scientific layout")
    endogenous = {
        construct.id
        for construct in dynamical_model_spec.constructs
        if construct.role == "endogenous"
    }
    grids = {
        (layout.time_points, layout.time_origin)
        for layout in dynamical_model_spec.law_layouts.values()
        if set(layout.constructs) & endogenous
    }
    if len(grids) > 1:
        raise ValueError("Endogenous trajectories must share their retained time grid and origin")
    for identity, layout in dynamical_model_spec.law_layouts.items():
        parameters = tuple(
            sorted(
                parameter.id
                for parameter in dynamical_model_spec.parameters
                if parameter.distribution == identity
            )
        )
        constructs = tuple(
            sorted(
                construct.id
                for construct in dynamical_model_spec.constructs
                if construct.distribution == identity
            )
        )
        if (
            tuple(parameter for parameter, _ in layout.parameters) != parameters
            or layout.constructs != constructs
        ):
            raise ValueError("Joint layout membership must match its scientific quantities")
        if identity != layout.distribution_id or distribution_shape(
            dynamical_model_spec.distributions[identity]
        ) != ((), (layout.width,)):
            raise ValueError(
                "Joint law identity and event width must match its retained coordinates"
            )
        inputs = tuple(
            dynamical_model_spec.get_construct(key) for key in constructs if key not in endogenous
        )
        if inputs:
            if parameters or set(constructs) & endogenous:
                raise ValueError(
                    "Exogenous trajectory laws cannot contain parameters or endogenous states"
                )
            law = materialize_distribution(dynamical_model_spec.distributions[identity])
            if not isinstance(law, dist.Delta) or np.any(np.asarray(law.log_density) != 0):
                raise ValueError(
                    "Exogenous trajectories require a normalized deterministic Delta law"
                )
            values = np.asarray(law.v)
            if not np.isfinite(values).all():
                raise ValueError("Exogenous trajectories require finite values")
            for construct in inputs:
                path = values[layout.trajectory_slices[construct.id]]
                if construct.temporal_status == "time_invariant" and np.any(path != path[0]):
                    raise ValueError("Time-invariant input trajectories must be constant")
    for identity, law in dynamical_model_spec.distributions.items():
        if distribution_shape(law) != ((), ()):
            continue
        parameters = [
            parameter.id
            for parameter in dynamical_model_spec.parameters
            if parameter.distribution == identity
        ]
        if len(parameters) != 1 or any(
            construct.distribution == identity for construct in dynamical_model_spec.constructs
        ):
            raise ValueError("A scalar distribution must belong to exactly one parameter")
