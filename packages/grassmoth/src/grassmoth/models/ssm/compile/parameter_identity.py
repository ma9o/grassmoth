"""Derive scalar finding identities from canonical parameters and native bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.artifacts.identity import scientific_id
from grassmoth.artifacts.model.expressions import COEFFICIENT_MEANINGS
from grassmoth.artifacts.model.likelihood import OBSERVATION_FAMILY_SPECS
from grassmoth.artifacts.model.parameter import SiteKind
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.models.ssm.compile.layout import axes
from grassmoth.models.ssm.structure.sites import WholeSiteSelection

if TYPE_CHECKING:
    from grassmoth.artifacts.identity import ParameterElementId
    from grassmoth.artifacts.model.parameter_spec import ParameterSpec
    from grassmoth.models.ssm.compile.bindings import CompiledSiteCoordinate
    from grassmoth.models.ssm.structure.sites import SiteDescriptor, SiteSelection
    from grassmoth.models.structure import StructuralSelection


SHARED_OBSERVATION_FAMILIES = {
    COEFFICIENT_MEANINGS[role].quantity: law.family.value
    for law in OBSERVATION_FAMILY_SPECS
    if law.family != DistributionFamily.ORDERED_LOGISTIC
    for role in law.parameter_roles
}


def component_identity(
    parameter: ParameterSpec,
    coordinate: CompiledSiteCoordinate,
    selection: SiteSelection,
    site: SiteDescriptor,
    structure: StructuralSelection,
) -> tuple[ParameterElementId, str] | None:
    """Identify category components by labels and covariance components by their ordered basis.

    Padded likelihood coordinates are execution-only and have no scientific component.
    """
    indices = coordinate.coordinate.indices
    kind = site.site_kind
    key: object = "scalar"
    label = parameter.name
    if kind in {
        SiteKind.OBS_ORDERED_BASE,
        SiteKind.OBS_ORDERED_GAPS,
        SiteKind.OBS_CAT_INTERCEPTS,
        SiteKind.OBS_CAT_SLOPES,
    }:
        indicators = {
            item.observation.name: item for item in structure.dynamical_model_spec.indicators
        }
        indicator = indicators[axes.observation_names(structure)[indices[0]]]
        levels = (
            indicator.observation.ordinal_levels
            if kind in {SiteKind.OBS_ORDERED_BASE, SiteKind.OBS_ORDERED_GAPS}
            else indicator.observation.categorical_levels
        )
        if levels is None:
            return None
        if kind == SiteKind.OBS_ORDERED_BASE:
            key = [indicator.observation.id, "first_cutpoint", levels[:2]]
            label = f"{indicator.observation.name}: {levels[0]} | {levels[1]}"
        elif kind == SiteKind.OBS_ORDERED_GAPS:
            column = indices[1]
            if column >= len(levels) - 2:
                return None
            key = [indicator.observation.id, "cutpoint_gap", levels[column : column + 3]]
            label = f"{indicator.observation.name}: gap {levels[column]} / {levels[column + 1]} / {levels[column + 2]}"
        else:
            column = indices[1]
            if column >= len(levels) - 1:
                return None
            key = [indicator.observation.id, "category_contrast", levels[column + 1], levels[0]]
            label = f"{indicator.observation.name}: {levels[column + 1]} vs {levels[0]}"
    elif kind in {
        SiteKind.DIFFUSION_LOWER,
        SiteKind.DIFFUSION_DIAG,
        SiteKind.T0_VAR_LOWER,
        SiteKind.T0_VAR_DIAG,
    }:
        # A Cholesky entry is conditional on the preceding ordered basis. Reordering
        # that basis changes the quantity even if the endpoint labels survive.
        constructs = {item.name: item.id for item in structure.dynamical_model_spec.constructs}
        position = site.positions[coordinate.flat_index]
        row = position[0] if isinstance(position, tuple) else position
        basis = [constructs[name] for name in axes.state_names(structure)[: row + 1]]
        key = ["cholesky", basis]
    elif any(n > 1 for n in site.shape) and isinstance(selection, WholeSiteSelection):
        raise ValueError(f"Parameter {parameter.name!r} needs explicit scientific component axes")
    return scientific_id("element", [parameter.id, key]), label
