"""Compile the shared symbolic formula to native, differentiable JAX arithmetic."""

from __future__ import annotations

from collections import Counter
from types import MappingProxyType
from typing import TYPE_CHECKING, cast, overload

import jax
import jax.numpy as jnp
from sympy import Basic, Dummy, Symbol, evaluate, lambdify, postorder_traversal, preorder_traversal

from grassmoth.artifacts.model.expressions import symbolic_expression

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from grassmoth.artifacts.model.expressions import (
        CoefficientExpression,
        Expression,
        StateExpression,
    )


type Subexpressions = tuple[tuple[Symbol, Basic], ...]


@overload
def structural_cse(expressions: Basic) -> tuple[Subexpressions, Basic]: ...


@overload
def structural_cse(expressions: Sequence[Basic]) -> tuple[Subexpressions, tuple[Basic, ...]]: ...


def structural_cse(
    expressions: Basic | Sequence[Basic],
) -> tuple[Subexpressions, Basic | tuple[Basic, ...]]:
    """Share identical nodes without algebraic rewriting of authored domains."""
    roots = (expressions,) if isinstance(expressions, Basic) else tuple(expressions)
    counts = Counter(node for root in roots for node in preorder_traversal(root) if node.args)
    ordered = tuple(
        dict.fromkeys(node for root in roots for node in postorder_traversal(root) if node.args)
    )
    substitutions: dict[Basic, Symbol] = {}
    assignments: list[tuple[Symbol, Basic]] = []
    with evaluate(False):
        for node in ordered:
            if counts[node] > 1:
                symbol = Dummy()
                assignments.append((symbol, node.xreplace(substitutions)))
                substitutions[node] = symbol
        outputs = tuple(cast("Basic", root.xreplace(substitutions)) for root in roots)
    return tuple(assignments), outputs[0] if isinstance(expressions, Basic) else outputs


def compile_expression(
    value: Expression,
    *,
    substitution: Expression | None = None,
    functions: Mapping[str, Callable[..., jax.Array]] = MappingProxyType({}),
) -> tuple[tuple[StateExpression | CoefficientExpression, ...], Callable[..., jax.Array]]:
    """Return ordered operand owners and a formula whose first argument is the predictor."""
    symbolic = symbolic_expression(value)
    predictor = Symbol("predictor")
    root = symbolic.root
    if substitution is not None:
        # xreplace must preserve authored domains such as x/x at x=0.
        with evaluate(False):
            root = root.xreplace({symbolic_expression(substitution).root: predictor})
    symbols = tuple(
        dict.fromkeys(
            node
            for node in preorder_traversal(root)
            if isinstance(node, Symbol) and node != predictor
        )
    )
    operands = tuple(symbolic.operands[symbol] for symbol in symbols)
    with evaluate(False):
        numerical = lambdify(
            (predictor, *symbols),
            root,
            modules=[
                {
                    "maximum": jnp.maximum,
                    "sigmoid": jax.nn.sigmoid,
                    "normal_cdf": jax.scipy.special.ndtr,
                    **functions,
                },
                "jax",
            ],
            dummify=True,
            docstring_limit=0,
            cse=structural_cse,  # pyright: ignore[reportArgumentType] -- SymPy supports callable CSE; its annotation only declares bool.
        )

    def outlined(*arguments: jax.Array | float) -> jax.Array:
        return jnp.asarray(numerical(*arguments))

    return operands, jax.jit(outlined, inline=jax.Inline.JAX_EARLY)
