"""Read parameter meaning from component slots; no parallel inventory is stored."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.model.parameter_context import CoefficientUse, iter_coefficient_uses
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.models.structure import selected_indicators, selected_state_ids

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nof1_causal_lab.artifacts.model.construct import ConstructSpec
    from nof1_causal_lab.artifacts.model.parameter_spec import ParameterSpec
    from nof1_causal_lab.models.structure import StructuralSelection


def execution_coefficient_uses(selection: StructuralSelection) -> Iterator[CoefficientUse]:
    """Exclude coefficients owned only by structure outside the numerical selection."""
    dynamical_model_spec = selection.dynamical_model_spec
    states = set(selected_state_ids(selection))
    roots = {
        edge.cause.id
        for edge in dynamical_model_spec.edges
        if edge.cause.id in selection.marginalized_construct_ids and edge.effect.id in states
    }
    edges = {
        edge.id
        for edge in dynamical_model_spec.edges
        if edge.cause.id in states | roots and edge.effect.id in states
    }
    active = {
        "construct": states | roots,
        "edge": edges,
        "indicator": {indicator.observation.id for indicator in selected_indicators(selection)},
    }
    for use in iter_coefficient_uses(dynamical_model_spec):
        if all(owner.kind == "mechanism" or owner.id in active[owner.kind] for owner in use.owners):
            yield use


def execution_parameters(selection: StructuralSelection) -> tuple[ParameterSpec, ...]:
    """Select the scientific coefficients used by the current retained structure."""
    referenced = {use.value for use in execution_coefficient_uses(selection)}
    return tuple(
        parameter
        for parameter in selection.dynamical_model_spec.parameters
        if parameter.id in referenced
    )


def require_priors(selection: StructuralSelection) -> None:
    """Only the coefficients the selection executes need prior laws."""
    missing = [
        parameter.id
        for parameter in execution_parameters(selection)
        if parameter.distribution is None
    ]
    if missing:
        raise IncompleteModelError(f"Compilation requires declared prior laws for {missing}")


def baseline_factor_groups(
    selection: StructuralSelection,
) -> tuple[tuple[ConstructSpec, ...], ...]:
    """A shared scale denotes one identifiable factor for marginalized baseline roots."""
    dynamical_model_spec = selection.dynamical_model_spec
    grouped: dict[str, list[ConstructSpec]] = {}
    states = set(selected_state_ids(selection))
    retained_parents = {
        edge.cause.id for edge in dynamical_model_spec.edges if edge.effect.id in states
    }
    for construct in dynamical_model_spec.constructs:
        if (
            construct.indicators
            or construct.id not in retained_parents
            or construct.temporal_status != "time_invariant"
            or construct.coefficient("initial_scale") is None
        ):
            continue
        coefficient = construct.coefficient("initial_scale")
        key = coefficient if isinstance(coefficient, str) else construct.id
        grouped.setdefault(key, []).append(construct)
    return tuple(tuple(group) for group in grouped.values())
