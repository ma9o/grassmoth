"""Compile shared nonlinear drift expressions with explicit operand ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from sympy import Basic, Symbol, evaluate, lambdify

from grassmoth.artifacts.model.expressions import (
    CoefficientExpression,
    StateExpression,
    symbolic_expression,
)
from grassmoth.models.ssm.compile.expressions import structural_cse

if TYPE_CHECKING:
    from collections.abc import Mapping

    from grassmoth.artifacts.identity import ParameterId
    from grassmoth.models.ssm.dynamics.expression import ExpressionComponentSpec
    from grassmoth.models.ssm.dynamics.spec import DynamicsSpec
    from grassmoth.models.ssm.dynamics.vector_field import DriftProgram


@dataclass(frozen=True)
class _StateRead:
    index: int


@dataclass(frozen=True)
class _EdgeRead:
    target: int
    index: int


@dataclass(frozen=True)
class _ParameterRead:
    component: int
    identity: ParameterId


@dataclass(frozen=True)
class _LiteralRead:
    value: float


type _DriftRead = _StateRead | _EdgeRead | _ParameterRead | _LiteralRead


def _bind_operand(
    index: int,
    component: ExpressionComponentSpec,
    operand: StateExpression | CoefficientExpression,
) -> _DriftRead:
    match operand:
        case StateExpression():
            state = component.state_index[operand.construct_id]
            return _EdgeRead(component.target, state) if component.edge_owned else _StateRead(state)
        case CoefficientExpression():
            reference = component.coefficients[operand]
            return (
                _ParameterRead(index, reference)
                if isinstance(reference, str)
                else _LiteralRead(reference)
            )


def _read_operand(
    operand: _DriftRead,
    eta: jax.Array,
    eta_per_edge: jax.Array,
    params: tuple[Mapping[str, jax.Array], ...],
) -> jax.Array:
    match operand:
        case _StateRead(index):
            return eta[index]
        case _EdgeRead(target, index):
            return eta_per_edge[target, index]
        case _ParameterRead(component, identity):
            return params[component][identity]
        case _LiteralRead(value):
            return jnp.asarray(value)


def compile_shared_drift(spec: DynamicsSpec) -> DriftProgram:
    """Share structural subexpressions across drift outputs while preserving parameter slices."""
    roots: list[Basic] = []
    providers: dict[_DriftRead, Symbol] = {}
    targets: list[int] = []
    for index, component in enumerate(spec.components):
        if component.kind == "potential":
            continue
        symbolic = symbolic_expression(component.expression)
        renamed: dict[Symbol, Symbol] = {}
        for symbol, operand in symbolic.operands.items():
            owner = _bind_operand(index, component, operand)
            if owner not in providers:
                providers[owner] = Symbol(f"operand_{len(providers)}")
            renamed[symbol] = providers[owner]
        with evaluate(False):
            roots.append(symbolic.root.xreplace(renamed))
        targets.append(component.target)
    entries = tuple(providers)
    with evaluate(False):
        numerical = lambdify(
            tuple(providers.values()),
            tuple(roots),
            modules=[
                {
                    "maximum": jnp.maximum,
                    "sigmoid": jax.nn.sigmoid,
                    "normal_cdf": jax.scipy.special.ndtr,
                },
                "jax",
            ],
            cse=structural_cse,  # pyright: ignore[reportArgumentType] -- SymPy supports callable CSE; its annotation only declares bool.
            dummify=True,
            docstring_limit=0,
        )
    contributions_by_state = tuple(
        tuple(index for index, target in enumerate(targets) if target == state)
        for state in range(spec.n_latent)
    )

    def drift(
        _t: jax.Array,
        eta: jax.Array,
        eta_per_edge: jax.Array,
        params: tuple[Mapping[str, jax.Array], ...],
    ) -> jax.Array:
        values = tuple(_read_operand(operand, eta, eta_per_edge, params) for operand in entries)
        contributions = tuple(jnp.asarray(value, dtype=eta.dtype) for value in numerical(*values))
        # Ownership is static. Sum scalar terms before stacking state outputs so
        # particle/time batching never materializes a contribution bank to scatter.
        return jnp.stack(
            tuple(
                sum((contributions[index] for index in indices), jnp.zeros((), dtype=eta.dtype))
                for indices in contributions_by_state
            )
        )

    return drift
