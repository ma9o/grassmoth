"""Persistent additive contributions to continuous-time dynamics."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import MechanismId
from grassmoth.artifacts.model.expressions import (
    CallExpression,
    CoefficientExpression,
    Expression,
    walk_expression,
)


class _MechanismSpec(Value):
    """A symbolic specification of an additive drift term or a node potential.

    A node potential contributes its negative gradient to the drift.
    """

    model_config = ConfigDict(revalidate_instances="always")

    id: MechanismId
    expression: Expression

    @model_validator(mode="after")
    def validate_drift_operands(self) -> Self:
        for node in walk_expression(self.expression):
            if isinstance(node, CoefficientExpression):
                if node.role not in {
                    "center",
                    "decay",
                    "quartic",
                    "intercept",
                    "weight",
                    "emax",
                    "ec50",
                    "exponent",
                }:
                    raise ValueError(
                        f"{node.role} is an observation coefficient, not a drift operand"
                    )
                if node.value is None:
                    raise ValueError("A declared drift contribution requires assigned coefficients")
            if isinstance(node, CallExpression) and node.function in {
                "ordered_cutpoints",
                "category_logits",
            }:
                raise ValueError(f"{node.function} requires an observation category context")
        return self


class DriftMechanismSpec(_MechanismSpec):
    """An additive drift contribution on a construct or directed edge."""

    kind: Literal["drift"] = "drift"


class PotentialMechanismSpec(_MechanismSpec):
    """A construct potential whose negative gradient contributes to its drift."""

    kind: Literal["potential"] = "potential"


type DynamicsMechanismSpec = Annotated[
    DriftMechanismSpec | PotentialMechanismSpec, Field(discriminator="kind")
]
