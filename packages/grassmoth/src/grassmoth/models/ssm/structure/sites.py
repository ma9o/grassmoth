"""Canonical sample-site descriptor and support enums.

Lives in ``structure`` so block specs can return them from
``iter_sites()`` without circular imports with ``parameterization.py``.

The descriptor is the canonical identity of a sample site: name, shape,
support class, semantic kind, assembly grouping, and the various prior
binding keys used by compile-time and runtime layers.
"""

from __future__ import annotations

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.model.parameter import SiteKind, SupportClass

type SitePosition = int | tuple[int, ...]


class SiteDescriptor(Value):
    """Metadata for a single sample site.

    ``positions`` is the free-entry index list owned by the originating
    block: ``list[int]`` for vector-shaped sites, ``list[tuple[int, int]]``
    for matrix-shaped sites. Compile-time consumers use it to translate
    flat-vector parameter indices back to structural ``(i, j)`` positions.
    """

    name: str
    shape: tuple[int, ...]
    support: SupportClass
    assembly_group: str
    site_kind: SiteKind
    positions: tuple[SitePosition, ...] = ()
    prior_field: str | None = None


class ScalarSiteSelection(Value):
    """Selection of one coordinate by its flat index within a sampling site."""

    flat_index: int


class RowSiteSelection(Value):
    """Selection of an entire leading-axis row within a sampling site."""

    row: int


class WholeSiteSelection(Value):
    """Select every native coordinate, including execution-only padding."""


type SiteSelection = ScalarSiteSelection | RowSiteSelection | WholeSiteSelection


class CompiledBlockTarget(Value):
    """A parameter whose placement is owned by its block's site coordinates."""


class CompiledNodeTarget(Value):
    """A node expression's component and owning state coordinate."""

    target_index: int


class CompiledEdgeTarget(Value):
    """An edge expression's component and resolved effect/cause coordinates."""

    target_index: int
    source_index: int


type CompiledBindingTarget = CompiledBlockTarget | CompiledNodeTarget | CompiledEdgeTarget


class CompiledSiteBinding(Value):
    """A selected native site and its resolved structural owner."""

    site: SiteDescriptor
    selection: SiteSelection
    target: CompiledBindingTarget


def site_size(shape: tuple[int, ...]) -> int:
    """Number of scalar elements in a site shape."""
    size = 1
    for d in shape:
        size *= d
    return size
