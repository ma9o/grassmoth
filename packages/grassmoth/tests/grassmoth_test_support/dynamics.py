"""Expression constructors and explicit linear reference components for numerical tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from grassmoth.artifacts.identity import ConstructId, scientific_id
from grassmoth.artifacts.model.expressions import (
    coefficient,
    hill,
    restoring_force,
    restoring_potential,
    state,
)
from grassmoth.models.ssm.dynamics.expression import ExpressionComponentSpec
from grassmoth.models.ssm.dynamics.vector_field import ComponentDrift, VectorField

if TYPE_CHECKING:
    from collections.abc import Mapping

    from grassmoth.models.ssm.dynamics.edges import VectorFieldComponent

_ZERO = 0


def reference_vector_field(
    n_latent: int,
    components: tuple[VectorFieldComponent, ...],
    potential_indices: tuple[int, ...] = (),
) -> VectorField:
    """Bind independent numerical reference components to the explicit drift program."""

    return VectorField(
        n_latent, components, ComponentDrift(components, potential_indices), potential_indices
    )


def _coefficient(slot, role):
    return slot if isinstance(slot, (int, float)) else scientific_id("parameter", role)


def _state(index):
    return state(ConstructId(f"construct:state{index}"))


def _term(expression, target, source=None, *, kind="drift", max_index=None):
    last = max(target, source if source is not None else target, max_index or 0)
    return ExpressionComponentSpec(
        expression=expression,
        target=target,
        source=source,
        state_ids=tuple(ConstructId(f"construct:state{i}") for i in range(last + 1)),
        kind=kind,
    )


def decay_term(target, decay=None):
    return _term(
        restoring_force(
            _state(target).construct_id,
            center=_ZERO,
            stiffness=_coefficient(decay, "decay"),
            quartic=_ZERO,
        ),
        target,
    )


def intercept_term(target, cint=None):
    return _term(coefficient(_coefficient(cint, "intercept"), "intercept"), target)


def potential_term(target, center=None, stiffness=None, quartic=_ZERO):
    return _term(
        restoring_potential(
            _state(target).construct_id,
            center=_coefficient(center, "center"),
            stiffness=_coefficient(stiffness, "decay"),
            quartic=_coefficient(quartic, "quartic"),
        ),
        target,
        kind="potential",
    )


def linear_term(source, target, weight=None):
    return _term(
        coefficient(_coefficient(weight, "weight"), "weight") * _state(source), target, source
    )


def hill_term(source, target, emax=None, ec50=None, n=None):
    return _term(
        hill(
            _state(source),
            emax=_coefficient(emax, "emax"),
            ec50=_coefficient(ec50, "ec50"),
            n=_coefficient(n, "exponent"),
        ),
        target,
        source,
    )


def interaction_term(source_a, source_b, target, weight=None):
    return _term(
        coefficient(_coefficient(weight, "weight"), "weight") * _state(source_a) * _state(source_b),
        target,
        source_a,
        max_index=source_b,
    )


# ---------------------------------------------------------------------------
# Full-vector terms
# ---------------------------------------------------------------------------


class DenseLinear(eqx.Module):
    """``A @ η + c`` contribution as a single matmul.

    Reads ``params['drift']`` (``(n, n)``) and optional ``params['cint']``
    (``(n,)``). This test reference consumes matrix coefficients directly.
    Edge-input overrides go through ``eta_per_edge`` to exercise surgical
    interventions on specific ``j → i`` couplings.
    """

    def contribute(
        self,
        accumulator: Array,
        _eta: Array,
        eta_per_edge: Array,
        _t: Array,
        params: Mapping[str, Array],
    ) -> Array:
        """Add dense linear drift and its intercept using the edge-adjusted source states."""
        A = params["drift"]
        cint = params.get("cint", jnp.zeros(A.shape[0], dtype=accumulator.dtype))
        return accumulator + (A * eta_per_edge).sum(axis=1) + cint


class DiagonalDecay(eqx.Module):
    """Per-latent relaxation term ``-decay · η``.

    Used in the vector-field primitive case where the user has explicit
    natural timescales (decay rate ``ρ_i`` for each latent). For the
    effect-compartment pattern, set ``decay[target] = k_e0`` to match
    a ``LinearEdge`` with weight ``k_e0`` and obtain
    ``dC_e/dt = k_e0 · (C_p − C_e)``.
    """

    def contribute(
        self,
        accumulator: Array,
        eta: Array,
        _eta_per_edge: Array,
        _t: Array,
        params: Mapping[str, Array],
    ) -> Array:
        """Add independent negative decay terms for every state coordinate."""
        return accumulator + (-params["decay"] * eta)


class StateDecay(eqx.Module):
    """Single-latent relaxation term ``-decay * eta[target]``."""

    target: int = eqx.field(static=True)

    def contribute(
        self,
        accumulator: Array,
        eta: Array,
        _eta_per_edge: Array,
        _t: Array,
        params: Mapping[str, Array],
    ) -> Array:
        """Add negative decay to this component's target state only."""
        return accumulator.at[self.target].add(-params["decay"] * eta[self.target])


class Intercept(eqx.Module):
    """Per-latent constant intercept ``c``."""

    def contribute(
        self,
        accumulator: Array,
        _eta: Array,
        _eta_per_edge: Array,
        _t: Array,
        params: Mapping[str, Array],
    ) -> Array:
        """Add the constant drift vector to the accumulated derivative."""
        return accumulator + params["cint"]


class StateIntercept(eqx.Module):
    """Single-latent constant intercept contribution."""

    target: int = eqx.field(static=True)

    def contribute(
        self,
        accumulator: Array,
        _eta: Array,
        _eta_per_edge: Array,
        _t: Array,
        params: Mapping[str, Array],
    ) -> Array:
        """Add a constant drift contribution to this component's target state only."""
        return accumulator.at[self.target].add(params["cint"])


# ---------------------------------------------------------------------------
# Single-target edges
# ---------------------------------------------------------------------------


class LinearEdge(eqx.Module):
    """``w · η[source]`` contribution at ``target``.

    Also expresses *effect compartments* — set the target's
    ``DiagonalDecay`` rate to the same value as this edge's weight and
    you get ``dC_e/dt = w · (C_p − C_e)``.
    """

    source: int = eqx.field(static=True)
    target: int = eqx.field(static=True)

    def contribute(
        self,
        accumulator: Array,
        _eta: Array,
        eta_per_edge: Array,
        _t: Array,
        params: Mapping[str, Array],
    ) -> Array:
        """Add the weighted edge-adjusted source state to the target derivative."""
        contribution = params["weight"] * eta_per_edge[self.target, self.source]
        return accumulator.at[self.target].add(contribution)
