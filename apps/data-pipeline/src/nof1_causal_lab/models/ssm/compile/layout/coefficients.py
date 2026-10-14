"""Bind authored scalar quantities to execution coordinates and default templates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.model.parameter import SiteKind
from nof1_causal_lab.artifacts.model.parameter_context import coefficient_value
from nof1_causal_lab.models.parameters import execution_coefficient_uses
from nof1_causal_lab.models.ssm.compile.layout.axes import (
    observation_ids,
    state_ids,
    time_invariant_mask,
)
from nof1_causal_lab.models.ssm.compile.layout.errors import NumericalSupportError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId
    from nof1_causal_lab.artifacts.model.parameter_context import CoefficientUse
    from nof1_causal_lab.models.structure import StructuralSelection


def quantity_position(selection: StructuralSelection, parameter: CoefficientUse) -> tuple[int, ...]:
    """Locate a scientific scalar by its owners in the derived execution axes."""
    state = {key: i for i, key in enumerate(state_ids(selection))}
    observation = {key: i for i, key in enumerate(observation_ids(selection))}
    owners = {owner.id for owner in parameter.owners}

    def one(axis: Mapping[ConstructId, int] | Mapping[IndicatorId, int]) -> int:
        matches = [index for key, index in axis.items() if key in owners]
        if len(matches) != 1:
            raise NumericalSupportError(
                [f"Quantity {parameter.slot!r} needs one owner on this axis"]
            )
        return matches[0]

    kind = parameter.quantity
    if kind == SiteKind.LOADING:
        return (one(observation), one(state))
    if kind in {SiteKind.DIFFUSION_LOWER, SiteKind.T0_VAR_LOWER}:
        pair = sorted(index for key, index in state.items() if key in owners)
        if len(pair) != 2:
            raise NumericalSupportError(
                [f"Covariance quantity {parameter.slot!r} requires two distinct state owners"]
            )
        return (pair[1], pair[0])
    if kind in {SiteKind.MANIFEST_MEANS, SiteKind.MANIFEST_VAR_DIAG}:
        return (one(observation),)
    if kind == SiteKind.STATIC_STATE_SD:
        from nof1_causal_lab.models.parameters import baseline_factor_groups

        matches = [
            index
            for index, group in enumerate(baseline_factor_groups(selection))
            if owners & {construct.id for construct in group}
        ]
        if len(matches) != 1:
            raise NumericalSupportError(["A baseline scale must belong to one identifiable factor"])
        return (matches[0],)
    return (one(state),)


def _quantity_values(
    selection: StructuralSelection,
    kind: SiteKind,
    template: np.ndarray,
    support: np.ndarray,
    *,
    diagonal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply explicit scalar definitions to the scientific default policy."""
    values = np.array(template, dtype=float, copy=True)
    free = np.array(support, dtype=bool, copy=True)
    occupied = {}
    for parameter in execution_coefficient_uses(selection):
        if parameter.quantity != kind:
            continue
        if kind == SiteKind.T0_MEANS and not any(
            owner.id in state_ids(selection) for owner in parameter.owners
        ):
            continue
        position = quantity_position(selection, parameter)
        value = coefficient_value(parameter.value)
        if (
            kind in {SiteKind.DIFFUSION_DIAG, SiteKind.DIFFUSION_LOWER}
            and any(time_invariant_mask(selection)[index] for index in position)
            and value != 0.0
        ):
            raise NumericalSupportError(["Time-invariant constructs cannot have innovations"])
        if kind in {SiteKind.DIFFUSION_LOWER, SiteKind.T0_VAR_LOWER}:
            expected_kind = (
                "innovation_correlation"
                if kind == SiteKind.DIFFUSION_LOWER
                else "initial_state_correlation"
            )
            pair = {state_ids(selection)[index] for index in position}
            if not any(
                kind == expected_kind and {first, second} == pair
                for first, second, kind in selection.induced_dependencies
            ):
                raise NumericalSupportError(
                    [
                        "Correlated quantities require an explicit latent confounder in the scientific DAG"
                    ]
                )
        if (
            value is not None
            and kind
            in {
                SiteKind.DIFFUSION_DIAG,
                SiteKind.MANIFEST_VAR_DIAG,
                SiteKind.T0_VAR_DIAG,
                SiteKind.STATIC_STATE_SD,
            }
            and value < 0
        ):
            raise NumericalSupportError(["A fixed standard deviation cannot be negative"])
        if kind == SiteKind.STATIC_STATE_SD and occupied.get(position) == parameter.value:
            continue
        if position in occupied:
            raise NumericalSupportError(
                [f"Multiple scientific definitions for {kind.value} at {position}"]
            )
        occupied[position] = parameter.value
        index = (position[0], position[0]) if diagonal else position
        support_index = index if free.ndim == len(index) else position
        free[support_index] = value is None
        if value is not None:
            values[index] = value
    return values, free
