"""Ephemeral bindings between scientific identities and numerical coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, assert_never

import numpy as np
from pydantic import Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import ParameterElementId, ParameterId
from nof1_causal_lab.artifacts.model.parameter import ParameterCoordinate, PriorAuthoringTransform
from nof1_causal_lab.artifacts.probability.joint_layout import JointLawLayout
from nof1_causal_lab.models.ssm.structure.sites import (
    CompiledBindingTarget,
    CompiledEdgeTarget,
    RowSiteSelection,
    ScalarSiteSelection,
    SiteDescriptor,
    WholeSiteSelection,
    site_size,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from datetime import datetime

    from nof1_causal_lab.artifacts.identity import ConstructId
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledDynamicalModel
    from nof1_causal_lab.models.ssm.structure.sites import SiteSelection


class CompiledSiteCoordinate(Value):
    """A native coordinate and its resolved offset in the site's flattened array."""

    coordinate: ParameterCoordinate
    flat_index: int = Field(ge=0)


def resolve_site_selection(
    site: SiteDescriptor, selection: SiteSelection
) -> tuple[CompiledSiteCoordinate, ...]:
    """Resolve a compiler-owned selection once, before binding or prior attachment."""
    match selection:
        case ScalarSiteSelection(flat_index=flat_index):
            indices = (flat_index,)
        case RowSiteSelection(row=row):
            indices = tuple(
                int(np.ravel_multi_index((row, column), site.shape))
                for column in range(site.shape[1])
            )
        case WholeSiteSelection():
            indices = tuple(range(site_size(site.shape)))
        case _:
            assert_never(selection)
    return tuple(
        CompiledSiteCoordinate(
            coordinate=ParameterCoordinate(
                site_name=site.name,
                indices=tuple(int(index) for index in np.unravel_index(flat_index, site.shape)),
            ),
            flat_index=flat_index,
        )
        for flat_index in indices
    )


class CompiledParameterBinding(Value):
    """Semantic parameter-to-runtime-site binding."""

    parameter_id: ParameterId
    parameter_name: str
    coordinates: Mapping[ParameterElementId, ParameterCoordinate] = Field(min_length=1)
    elements: Mapping[ParameterElementId, str] = Field(min_length=1)
    native_coordinates: tuple[CompiledSiteCoordinate, ...] = Field(min_length=1)
    site: SiteDescriptor
    target: CompiledBindingTarget
    transform: PriorAuthoringTransform

    @property
    def flat_index(self) -> int:
        """The first native coordinate; the constructor owns nonempty selection."""
        return self.native_coordinates[0].flat_index


class CompiledEffectInterval(Value):
    """An interval-effect prior carries its resolved edge owner."""

    target: CompiledEdgeTarget
    days: float


def parameter_bindings(
    compiled_dynamical_model: CompiledDynamicalModel,
) -> tuple[tuple[CompiledParameterBinding, ...], tuple[ParameterCoordinate, ...]]:
    """Read scientific bindings already resolved by the compiler."""
    return compiled_dynamical_model.bindings, compiled_dynamical_model.auxiliary_coordinates


def joint_law_layout(
    bindings: Collection[CompiledParameterBinding],
    *,
    parameters: Collection[ParameterId],
    constructs: Collection[ConstructId],
    time_points: Sequence[float],
    time_origin: datetime | Literal["relative"] = "relative",
) -> JointLawLayout:
    """Establish scientific coordinates and display labels once at production."""
    by_id = {binding.parameter_id: binding for binding in bindings}
    members = tuple(sorted(parameters))
    return JointLawLayout(
        parameters=tuple(
            (identity, tuple(sorted(by_id[identity].coordinates))) for identity in members
        ),
        constructs=tuple(sorted(constructs)),
        time_points=tuple(float(value) for value in time_points),
        time_origin=time_origin,
        labels={
            element: label
            for identity in members
            for element, label in by_id[identity].elements.items()
        },
    )
