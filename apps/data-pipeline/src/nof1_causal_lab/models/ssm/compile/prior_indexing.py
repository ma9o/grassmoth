"""Scientific parameter IDs bound to native sample sites through explicit ownership."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import ParameterId
from nof1_causal_lab.artifacts.model.parameter import SiteKind
from nof1_causal_lab.compilation_errors import AggregatedCompileError
from nof1_causal_lab.models.ssm.compile.layout import axes, static_factors
from nof1_causal_lab.models.ssm.structure.sites import (
    CompiledBlockTarget,
    CompiledEdgeTarget,
    CompiledNodeTarget,
    CompiledSiteBinding,
    RowSiteSelection,
    ScalarSiteSelection,
    WholeSiteSelection,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model.parameter_spec import ParameterSpec
    from nof1_causal_lab.models.ssm.dynamics.expression import ExpressionComponentSpec
    from nof1_causal_lab.models.ssm.structure.sites import (
        SiteDescriptor,
        SitePosition,
        SiteSelection,
    )
    from nof1_causal_lab.models.structure import StructuralSelection


class PriorIndexingError(AggregatedCompileError):
    """Aggregate independent scientific-ownership binding failures."""

    header = "Prior index binding failed"


class CompiledBindingRegistry(Value):
    """Parameter-ID keyed bindings; runtime aliases are display metadata only."""

    by_parameter: Mapping[ParameterId, CompiledSiteBinding]


def _native_dynamics_bindings(
    components: tuple[ExpressionComponentSpec, ...],
    sites: Mapping[str, SiteDescriptor],
) -> dict[ParameterId, CompiledSiteBinding]:
    """Bind coefficient references within the native component emitted by their own term."""
    result = {}
    for index, component in enumerate(components):
        target = (
            CompiledNodeTarget(target_index=component.target)
            if component.source is None
            else CompiledEdgeTarget(
                target_index=component.target,
                source_index=component.source,
            )
        )
        for identity, site in component.parameter_sites(f"vf_{index}"):
            if identity in result:
                raise PriorIndexingError(
                    ["One parameter cannot own multiple independent runtime sites"]
                )
            result[identity] = CompiledSiteBinding(
                site=sites[site.name],
                selection=ScalarSiteSelection(flat_index=0),
                target=target,
            )
    return result


def build_site_bindings(
    structure: StructuralSelection,
    sites: tuple[SiteDescriptor, ...],
    components: tuple[ExpressionComponentSpec, ...],
    parameters: tuple[ParameterSpec, ...],
) -> CompiledBindingRegistry:
    """Bind by mechanism coefficient references, quantities, and scientific owner IDs."""
    from nof1_causal_lab.models.ssm.compile.parameter_identity import SHARED_OBSERVATION_FAMILIES

    dynamical_model_spec = structure.dynamical_model_spec
    bindings = _native_dynamics_bindings(components, {site.name: site for site in sites})
    latent = {identity: index for index, identity in enumerate(axes.state_ids(structure))}
    manifest = {identity: index for index, identity in enumerate(axes.observation_ids(structure))}
    errors: list[str] = []

    for parameter in parameters:
        if parameter.id in bindings:
            continue
        kind = dynamical_model_spec.parameter_context(parameter.id).quantity
        matches: list[tuple[SiteDescriptor, SiteSelection]] = []
        owners = dynamical_model_spec.parameter_context(parameter.id).owners
        construct_ids = {owner.id for owner in owners if owner.kind == "construct"}
        indicator_ids = {owner.id for owner in owners if owner.kind == "indicator"}
        state_indices = {latent[key] for key in construct_ids if key in latent}
        indicator_indices = {manifest[key] for key in indicator_ids if key in manifest}
        position: SitePosition | None = None
        if kind in SHARED_OBSERVATION_FAMILIES or kind == SiteKind.PROC_DF:
            matches = [(site, WholeSiteSelection()) for site in sites if site.site_kind == kind]
        elif kind in {SiteKind.OBS_ORDERED_BASE, SiteKind.OBS_ORDERED_GAPS}:
            if len(indicator_indices) == 1:
                matches = [
                    (
                        site,
                        RowSiteSelection(row=next(iter(indicator_indices)))
                        if kind == SiteKind.OBS_ORDERED_GAPS
                        else ScalarSiteSelection(flat_index=next(iter(indicator_indices))),
                    )
                    for site in sites
                    if site.site_kind == kind
                ]
        else:
            if kind == SiteKind.STATIC_STATE_SD:
                factor_ids = construct_ids & set(static_factors.static_factor_ids(structure))
                if len(factor_ids) == 1:
                    position = {
                        identity: index
                        for index, identity in enumerate(
                            static_factors.static_factor_ids(structure)
                        )
                    }[next(iter(factor_ids))]
            elif kind == SiteKind.LOADING:
                if len(indicator_indices) == 1 and len(state_indices) == 1:
                    position = (next(iter(indicator_indices)), next(iter(state_indices)))
            elif kind in {SiteKind.DIFFUSION_LOWER, SiteKind.T0_VAR_LOWER}:
                if len(state_indices) == 2:
                    position = (max(state_indices), min(state_indices))
            elif kind in {SiteKind.MANIFEST_MEANS, SiteKind.MANIFEST_VAR_DIAG}:
                if len(indicator_indices) == 1:
                    position = next(iter(indicator_indices))
            elif (
                kind in {SiteKind.DIFFUSION_DIAG, SiteKind.T0_MEANS, SiteKind.T0_VAR_DIAG}
                and len(state_indices) == 1
            ):
                position = next(iter(state_indices))
            if position is not None:
                matches = [
                    (site, ScalarSiteSelection(flat_index=index))
                    for site in sites
                    if site.site_kind == kind
                    for index, candidate in enumerate(site.positions)
                    if candidate == position
                ]
        if len(matches) != 1:
            errors.append(
                f"Parameter {parameter.name!r} ({parameter.id}, {kind.value}) must bind to "
                f"one active site through its scientific owners; found {len(matches)}"
            )
            continue
        site, selection = matches[0]
        bindings[parameter.id] = CompiledSiteBinding(
            site=site,
            selection=selection,
            target=CompiledBlockTarget(),
        )
    if errors:
        raise PriorIndexingError(errors)
    return CompiledBindingRegistry(by_parameter=MappingProxyType(bindings))


__all__ = [
    "PriorIndexingError",
    "CompiledBindingRegistry",
    "build_site_bindings",
]
