"""Read parameter meaning from component slots; no parallel inventory is stored."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from grassmoth.artifacts.identity import ConstructRef, EdgeRef, IndicatorRef, MechanismRef
from grassmoth.artifacts.model.construct import CausalEdgeSpec
from grassmoth.artifacts.model.expressions import expression_coefficients, expression_states
from grassmoth.artifacts.model.parameter import SiteKind

if TYPE_CHECKING:
    from collections.abc import Iterator

    from grassmoth.artifacts.identity import EntityRef, ParameterId
    from grassmoth.artifacts.model.dynamical_model_spec import (
        DynamicalModelSpec,
        _ModelEntities,
    )


@dataclass(frozen=True)
class CoefficientUse:
    """One derived use of a literal or parameter within a scientific component."""

    quantity: SiteKind
    owners: tuple[EntityRef, ...]
    value: float | ParameterId
    slot: str


@dataclass(frozen=True)
class ParameterContext:
    """A derived reverse index of the slots referencing one parameter."""

    uses: tuple[CoefficientUse, ...]

    @property
    def quantity(self) -> SiteKind:
        """Shared coefficient quantity, rejecting uses with incompatible scientific meanings."""
        kinds = {use.quantity for use in self.uses}
        if len(kinds) != 1:
            raise ValueError("A shared parameter must have compatible coefficient meanings")
        return self.uses[0].quantity

    @property
    def owners(self) -> tuple[EntityRef, ...]:
        """Distinct scientific owners of all coefficient uses, preserving first occurrence."""
        return tuple({owner.id: owner for use in self.uses for owner in use.owners}.values())


def iter_coefficient_uses(
    dynamical_model_spec: DynamicalModelSpec | _ModelEntities,
) -> Iterator[CoefficientUse]:
    """Yield model coefficient uses with their quantity meanings, owners, and authored locations."""
    for owner, mechanism in dynamical_model_spec.iter_mechanisms():
        refs: list[EntityRef] = [MechanismRef(id=mechanism.id)]
        target = owner.effect.id if isinstance(owner, CausalEdgeSpec) else owner.id
        refs.append(ConstructRef(id=target))
        if isinstance(owner, CausalEdgeSpec):
            refs.append(EdgeRef(id=owner.id))
        refs.extend(
            ConstructRef(id=identity)
            for identity in sorted(expression_states(mechanism.expression) - {target})
        )
        for index, operand in enumerate(expression_coefficients(mechanism.expression)):
            if operand.value is None:
                continue
            quantity = operand.meaning.quantity
            yield CoefficientUse(
                quantity, tuple(refs), operand.value, f"{mechanism.id}.expression.{index}"
            )
    for construct in dynamical_model_spec.constructs:
        ref = ConstructRef(id=construct.id)
        for operand in construct.coefficients:
            if operand.value is None:
                continue
            quantity = operand.meaning.quantity
            if (
                quantity == SiteKind.T0_VAR_DIAG
                and not construct.indicators
                and construct.temporal_status == "time_invariant"
            ):
                quantity = SiteKind.STATIC_STATE_SD
            yield CoefficientUse(
                quantity,
                (ref, *(ConstructRef(id=identity) for identity in operand.construct_ids)),
                operand.value,
                ".".join((construct.id, "coefficients", operand.role, *operand.construct_ids)),
            )
        for indicator in construct.indicators:
            if indicator.likelihood is None:
                continue
            terms = indicator.likelihood.parsed
            for identity, operand in terms.loadings.items():
                if operand.value is not None:
                    yield CoefficientUse(
                        operand.meaning.quantity,
                        (ConstructRef(id=identity), IndicatorRef(id=indicator.observation.id)),
                        operand.value,
                        f"{indicator.observation.id}.likelihood.loading.{identity}",
                    )
            observation_refs = (ref, IndicatorRef(id=indicator.observation.id))
            for operand in (terms.intercept, *terms.auxiliary):
                if operand.value is not None:
                    yield CoefficientUse(
                        operand.meaning.quantity,
                        observation_refs,
                        operand.value,
                        f"{indicator.observation.id}.likelihood.{operand.role}",
                    )


def parameter_contexts(
    dynamical_model_spec: DynamicalModelSpec | _ModelEntities,
) -> dict[ParameterId, ParameterContext]:
    """Group symbolic coefficient uses by parameter identity to establish each parameter's context."""
    grouped: dict[ParameterId, list[CoefficientUse]] = {}
    for use in iter_coefficient_uses(dynamical_model_spec):
        if isinstance(use.value, str):
            grouped.setdefault(use.value, []).append(use)
    return {identity: ParameterContext(tuple(uses)) for identity, uses in grouped.items()}


def coefficient_value(coefficient: float | ParameterId) -> float | None:
    """Resolve a literal; named parameters always represent uncertain quantities."""
    return None if isinstance(coefficient, str) else coefficient
