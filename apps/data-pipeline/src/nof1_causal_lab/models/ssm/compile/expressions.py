"""Compile the shared symbolic formula to native, differentiable JAX arithmetic."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from sympy import Symbol, evaluate, lambdify, preorder_traversal

from nof1_causal_lab.artifacts.model.expressions import symbolic_expression

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from nof1_causal_lab.artifacts.model.expressions import (
        CoefficientExpression,
        Expression,
        StateExpression,
    )


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
        )
    return operands, numerical
