"""Reject coefficients that the whole model uses only through a sum or a product.

Every equation is expanded into a sum of terms, each a literal scale times parameter
powers times opaque factors (states, functions, and anything not expanded). If a set
of parameters appears only as single linear factors, the model depends on them only
through each term group's linear combination; if they appear only as factors, only
through their monomials. A rank deficit in either system is an exact invariance of
every equation, so no data can separate the parameters involved. The test is
syntactic: it misses redundancies it cannot see and never reports one that is absent.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from math import prod
from typing import TYPE_CHECKING, cast

from sympy import (
    Add,
    Expr,
    Function,
    ImmutableMatrix,
    Mul,
    Pow,
    Rational,
    Symbol,
    evaluate,
    expand_mul,
)
from sympy.core.traversal import bottom_up

from nof1_causal_lab.artifacts.model.construct import CausalEdgeSpec
from nof1_causal_lab.artifacts.model.expressions import (
    CoefficientExpression,
    LiteralExpression,
    symbolic_expression,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from nof1_causal_lab.artifacts.identity import ParameterId
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import (
        DynamicalModelSpec,
        _ModelEntities,
    )
    from nof1_causal_lab.artifacts.model.expressions import Expression

_MAX_TERMS = 64
_PARAMETER = re.compile(r"parameter:[0-9a-f]{64}")


@dataclass(frozen=True)
class _Term:
    """A literal scale times parameter powers times opaque factors."""

    scale: Fraction
    parameters: tuple[tuple[ParameterId, int], ...] = ()
    factors: tuple[tuple[str, int], ...] = ()


def _algebra(node: Expression) -> Expr:
    """Share symbolic lowering, retaining the conservative expansion budget."""
    symbolic = symbolic_expression(node)
    replacements = {
        symbol: (
            Rational(operand.value)
            if isinstance(operand.value, (int, float))
            else Symbol(operand.value)
        )
        for symbol, operand in symbolic.nodes.items()
        if isinstance(operand, CoefficientExpression) and operand.value is not None
    }
    replacements.update(
        {
            symbol: Rational(operand.value)
            for symbol, operand in symbolic.nodes.items()
            if isinstance(operand, LiteralExpression)
        }
    )
    with evaluate(False):
        root = symbolic.root.xreplace(replacements)

    def bounded(value: Expr) -> Expr:
        if isinstance(value, Function):
            return Symbol(str(value))
        if isinstance(value, Pow) and (
            not value.exp.is_Integer
            or (value.exp != -1 and value.exp < 1)
            or len(Add.make_args(value.base)) != 1
            or value.base == 0
        ):
            return Symbol(str(value))
        if isinstance(value, Add) and len(value.args) > _MAX_TERMS:
            return Symbol(str(value))
        if isinstance(value, Mul):
            if prod(len(Add.make_args(factor)) for factor in value.args) > _MAX_TERMS:
                return Symbol(str(value))
            return cast("Expr", expand_mul(value))
        return value

    return cast("Expr", bottom_up(root, bounded))


def _terms(value: Expr, parameters: Mapping[str, ParameterId]) -> tuple[_Term, ...]:
    terms = []
    for term in Add.make_args(value):
        scale, factors = term.as_coeff_Mul()
        if scale == 0:
            continue
        numerator, denominator = scale.as_numer_denom()
        powers = tuple(
            sorted(
                (str(symbol), int(power))
                for symbol, power in factors.as_powers_dict().items()
                if symbol != 1
            )
        )
        terms.append(
            _Term(
                Fraction(int(numerator), int(denominator)),
                tuple((parameters[name], power) for name, power in powers if name in parameters),
                tuple((name, power) for name, power in powers if name not in parameters),
            )
        )
    return tuple(terms)


def _equations(
    dynamical_model_spec: DynamicalModelSpec | _ModelEntities,
) -> Iterator[tuple[_Term, ...]]:
    """One expanded sum per scalar equation in which coefficients combine."""
    drift: dict[tuple[str, str], list[Expr]] = defaultdict(list)
    parameters = {str(parameter.id): parameter.id for parameter in dynamical_model_spec.parameters}
    for owner, mechanism in dynamical_model_spec.iter_mechanisms():
        target = owner.effect.id if isinstance(owner, CausalEdgeSpec) else owner.id
        drift[target, mechanism.kind].append(_algebra(mechanism.expression))
    for expressions in drift.values():
        yield _terms(Add(*expressions), parameters)
    for construct in dynamical_model_spec.constructs:
        for operand in construct.coefficients:
            yield _terms(_algebra(operand), parameters)
    for _, likelihood in dynamical_model_spec.iter_likelihoods():
        for _, operand in likelihood.law.operands():
            yield _terms(_algebra(operand), parameters)


def _nested(factor: str) -> frozenset[str]:
    """Parameters referenced inside an opaque factor's canonical form."""
    return frozenset(match.group() for match in _PARAMETER.finditer(factor))


def _rank(rows: Sequence[Mapping[ParameterId, Fraction]], columns: Sequence[ParameterId]) -> int:
    return int(
        ImmutableMatrix(
            [[row.get(column, Fraction(0)) for column in columns] for row in rows]
        ).rank()
    )


def _deficient_groups(
    rows: Sequence[Mapping[ParameterId, Fraction]], qualifying: frozenset[ParameterId]
) -> Iterator[tuple[list[ParameterId], list[dict[ParameterId, Fraction]]]]:
    """Connected parameter groups whose forms have fewer dimensions than parameters."""
    restricted = [
        {name: value for name, value in row.items() if name in qualifying} for row in rows
    ]
    restricted = [row for row in restricted if row]
    parent = {name: name for row in restricted for name in row}

    def root(name: ParameterId) -> ParameterId:
        while parent[name] != name:
            name = parent[name]
        return name

    for row in restricted:
        first, *rest = row
        for name in rest:
            parent[root(name)] = root(first)
    groups: dict[ParameterId, list[ParameterId]] = defaultdict(list)
    for name in sorted(parent):
        groups[root(name)].append(name)
    for members in groups.values():
        forms = [row for row in restricted if root(next(iter(row))) == root(members[0])]
        if len(members) > 1 and _rank(forms, members) < len(members):
            yield members, forms


def coefficient_redundancies(
    dynamical_model_spec: DynamicalModelSpec | _ModelEntities,
) -> tuple[str, ...]:
    """Describe each exact redundancy among the model's free coefficients."""
    equations = list(_equations(dynamical_model_spec))
    terms = [term for equation in equations for term in equation]
    nested = frozenset().union(*(_nested(name) for term in terms for name, _ in term.factors))
    nonlinear = {
        name
        for term in terms
        if len(term.parameters) > 1 or any(power != 1 for _, power in term.parameters)
        for name, _ in term.parameters
    }
    names = {parameter.id: parameter.name for parameter in dynamical_model_spec.parameters}
    found: list[str] = []
    linear_rows: list[Mapping[ParameterId, Fraction]] = []
    for equation in equations:
        groups: dict[tuple[tuple[str, int], ...], dict[ParameterId, Fraction]] = defaultdict(dict)
        for term in equation:
            if len(term.parameters) == 1 and term.parameters[0][1] == 1:
                groups[term.factors][term.parameters[0][0]] = term.scale
        linear_rows.extend(groups.values())
    for members, forms in _deficient_groups(linear_rows, frozenset(names) - nested - nonlinear):
        found.append(
            f"{_list(members, names)} enter the model only through "
            + ", ".join(_linear_form(form, names) for form in _distinct(forms))
        )
    product_rows: list[Mapping[ParameterId, Fraction]] = [
        {name: Fraction(power) for name, power in term.parameters} for term in terms
    ]
    for members, forms in _deficient_groups(product_rows, frozenset(names) - nested):
        found.append(
            f"{_list(members, names)} enter the model only through "
            + ", ".join(_product_form(form, names) for form in _distinct(forms))
        )
    return tuple(dict.fromkeys(found))


def _distinct(
    forms: Sequence[Mapping[ParameterId, Fraction]],
) -> list[Mapping[ParameterId, Fraction]]:
    return list({tuple(sorted(form.items())): form for form in forms}.values())


def _list(members: list[ParameterId], names: Mapping[ParameterId, str]) -> str:
    labels = sorted(names[name] for name in members)
    return ", ".join(labels[:-1]) + f" and {labels[-1]}"


def _linear_form(form: Mapping[ParameterId, Fraction], names: Mapping[ParameterId, str]) -> str:
    text = ""
    for index, (name, scale) in enumerate(sorted(form.items(), key=lambda item: names[item[0]])):
        sign = "-" if scale < 0 else "+" if index else ""
        size = "" if abs(scale) == 1 else f"{float(abs(scale)):g}·"
        text += f" {sign} {size}{names[name]}" if index else f"{sign}{size}{names[name]}"
    return text


def _product_form(form: Mapping[ParameterId, Fraction], names: Mapping[ParameterId, str]) -> str:
    return "·".join(
        names[name] if power == 1 else f"{names[name]}^{power}"
        for name, power in sorted(form.items(), key=lambda item: names[item[0]])
    )


def validate_coefficient_redundancy(
    dynamical_model_spec: DynamicalModelSpec | _ModelEntities,
) -> None:
    """A redundancy the observed-data law cannot separate is an authoring error."""
    if found := coefficient_redundancies(dynamical_model_spec):
        raise ValueError(
            "; ".join(found)
            + ". No data can separate them: merge each combination into one parameter "
            "with its own law."
        )
