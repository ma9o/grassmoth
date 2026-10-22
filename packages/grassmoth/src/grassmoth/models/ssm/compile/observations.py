"""Bind native observation operands to compiled coordinates and the cached predictor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from grassmoth.artifacts.model.expressions import CoefficientExpression, Expression
from grassmoth.artifacts.model.likelihood import Law, LinkFunction, map_law
from grassmoth.models.ssm.compile.expressions import compile_expression
from grassmoth.models.ssm.dynamics.expression import BoundExpression

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from grassmoth.artifacts.identity import ParameterId
    from grassmoth.artifacts.model.likelihood import LikelihoodAnalysis
    from grassmoth.artifacts.model.parameter import ParameterCoordinate
    from grassmoth.models.ssm.compile.bindings import CompiledParameterBinding
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
    from grassmoth.models.ssm.dynamics.expression import OperandEvaluator


def _inverse(eta: jax.Array) -> jax.Array:
    valid = jnp.isfinite(eta) & (eta > 0.0)
    return jnp.where(valid, 1.0 / jnp.where(valid, eta, 1.0), jnp.nan)


def _identity(eta: jax.Array) -> jax.Array:
    return eta


_RESPONSES: dict[LinkFunction, Callable[[jax.Array], jax.Array]] = {
    LinkFunction.IDENTITY: _identity,
    LinkFunction.LOG: jnp.exp,
    LinkFunction.INVERSE: _inverse,
    LinkFunction.LOGIT: jax.nn.sigmoid,
    LinkFunction.PROBIT: jax.scipy.special.ndtr,
    LinkFunction.CUMULATIVE_LOGIT: _identity,
    LinkFunction.SOFTMAX: _identity,
}


def _positive_domain(raw: OperandEvaluator) -> OperandEvaluator:
    """The inverse response's positive predictor domain is a scientific boundary."""

    def evaluate(eta: jax.Array, scale: jax.Array, values: tuple[jax.Array, ...]) -> jax.Array:
        valid = jnp.isfinite(eta) & (eta > 0.0)
        result = raw(jnp.where(valid, eta, 1.0), scale, values)
        return jnp.where(valid, result, jnp.nan)

    return evaluate


def bind_observation_law(
    law: Law[Expression],
    terms: LikelihoodAnalysis,
    bindings: Mapping[ParameterId, CompiledParameterBinding],
    *,
    channel: int,
    category_count: int,
    sampling_count: int,
    categorical_anchor: bool,
) -> Law[BoundExpression]:
    """Bind all IDs once; execution evaluates only resolved gathers and arithmetic."""
    response = _RESPONSES[terms.link]

    def bind(expression: Expression) -> BoundExpression:
        coordinates: list[ParameterCoordinate] = []

        def coefficient(operand: CoefficientExpression) -> OperandEvaluator:
            if operand.role == "observation_scale":
                return lambda _eta, scale, _values: scale
            value = operand.value
            if value is None:
                raise ValueError("Compilation requires assigned observation operands")
            if isinstance(value, (int, float)):
                return lambda _eta, _scale, _values: jnp.asarray(value)
            binding = bindings[value]
            selected = sorted(
                binding.coordinates.values(), key=lambda coordinate: coordinate.indices
            )
            vector = operand.role in {"cutpoint_gaps", "category_intercepts", "category_slopes"}
            width = category_count - (2 if operand.role == "cutpoint_gaps" else 1)
            if operand.role in {
                "cutpoint_base",
                "cutpoint_gaps",
                "category_intercepts",
                "category_slopes",
            }:
                selected = [
                    coordinate for coordinate in selected if coordinate.indices[0] == channel
                ]
            # An anchored category's first slope is pinned to one; its sampled slot is unread.
            pinned = categorical_anchor and operand.role == "category_slopes"
            if vector:
                selected = [
                    coordinate
                    for coordinate in selected
                    if int(pinned) <= coordinate.indices[-1] < width
                ]
            start = len(coordinates)
            coordinates.extend(selected)
            stop = len(coordinates)
            if not vector:
                if stop - start != 1:
                    raise ValueError(f"{operand.role} must bind to one native coordinate")
                return lambda _eta, _scale, values: values[start]
            if stop - start != width - int(pinned):
                raise ValueError(f"{operand.role} must bind every declared category coordinate")

            def gather(
                _eta: jax.Array, _scale: jax.Array, values: tuple[jax.Array, ...]
            ) -> jax.Array:
                result = jnp.stack(values[start:stop]) if stop > start else jnp.empty((0,))
                return (
                    jnp.concatenate((jnp.ones((1,), dtype=result.dtype), result))
                    if pinned
                    else result
                )

            return gather

        def cutpoints(base: jax.Array, gaps: jax.Array) -> jax.Array:
            base = jnp.atleast_1d(base)
            if category_count == 2:
                return base
            return jnp.concatenate((base, base + jnp.cumsum(gaps)))

        def logits(predictor: jax.Array, intercepts: jax.Array, slopes: jax.Array) -> jax.Array:
            return jnp.concatenate(
                (jnp.zeros((1,), dtype=predictor.dtype), intercepts + slopes * predictor)
            )

        operands, numerical = compile_expression(
            expression,
            substitution=terms.predictor,
            functions={"ordered_cutpoints": cutpoints, "category_logits": logits},
        )
        evaluators = tuple(
            coefficient(operand)
            if isinstance(operand, CoefficientExpression)
            else (lambda eta, _scale, _values: eta)
            for operand in operands
        )

        def evaluate(eta: jax.Array, scale: jax.Array, values: tuple[jax.Array, ...]) -> jax.Array:
            return jnp.asarray(
                numerical(eta, *(operand(eta, scale, values) for operand in evaluators))
            )

        evaluate_fn: OperandEvaluator = evaluate
        if terms.link == LinkFunction.INVERSE:
            evaluate_fn = _positive_domain(evaluate_fn)
        event_size = (
            category_count if law.distribution == "Categorical" and expression == law.logits else 0
        )
        if law.distribution == "OrderedLogistic" and expression == law.cutpoints:
            event_size = category_count - 1
        return BoundExpression(
            expression,
            evaluate_fn,
            tuple(coordinates),
            (),
            event_size,
            sampling_count,
            response,
            terms.link,
        )

    return map_law(law, bind)


def materialize_observation_laws(
    compiled_dynamical_model: CompiledDynamicalModel, samples: Mapping[str, jax.Array]
) -> tuple[Law[BoundExpression], ...]:
    """Gather each law's dynamic leaves once from canonical parameter-site draws."""

    def bind(operand: BoundExpression) -> BoundExpression:
        return operand.bind(samples)

    return tuple(
        map_law(observation.law, bind) for observation in compiled_dynamical_model.observations
    )
