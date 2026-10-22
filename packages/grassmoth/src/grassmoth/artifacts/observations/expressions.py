"""Owned window grammar: parse once, retain immutable terms, inspect and fold."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, assert_never, override

from pydantic_core import core_schema

from grassmoth.artifacts.observations.semantics import SummaryOperator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema

type WindowScalar = str | int | float | bool | None


class WindowOperator(StrEnum):
    """Permitted scalar, comparison, and aggregation operators in computed measurement expressions."""

    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    MODULO = "%"
    POWER = "**"
    NEGATE = "negate"
    POSITIVE = "positive"
    NOT = "not"
    AND = "and"
    OR = "or"
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    IN = "in"
    NOT_IN = "not in"
    IS = "is"
    IS_NOT = "is not"
    IF = "if"
    ABS = "abs"
    ALL = "all"
    ANY = "any"
    COALESCE = "coalesce"
    CONTAINS = "contains"
    CONTAINS_ANY = "contains_any"
    COUNT_NON_NULL = "count_non_null"
    COUNT_TRUE = "count_true"
    FIRST = "first"
    LAST = "last"
    LOWER = "lower"
    MAX = "max"
    MEAN = "mean"
    MIN = "min"
    STD = "std"
    SUM = "sum"


_FUNCTIONS = frozenset(
    operator
    for operator in WindowOperator
    if operator.name
    in {
        "ABS",
        "ALL",
        "ANY",
        "COALESCE",
        "CONTAINS",
        "CONTAINS_ANY",
        "COUNT_NON_NULL",
        "COUNT_TRUE",
        "FIRST",
        "LAST",
        "LOWER",
        "MAX",
        "MEAN",
        "MIN",
        "STD",
        "SUM",
    }
)
_REDUCTIONS: Mapping[WindowOperator, SummaryOperator] = {
    WindowOperator.FIRST: SummaryOperator.FIRST,
    WindowOperator.LAST: SummaryOperator.LAST,
    WindowOperator.SUM: SummaryOperator.SUM,
    WindowOperator.MEAN: SummaryOperator.MEAN,
    WindowOperator.STD: SummaryOperator.STD,
    WindowOperator.COUNT_TRUE: SummaryOperator.COUNT,
    WindowOperator.COUNT_NON_NULL: SummaryOperator.COUNT,
}
_WINDOW_FUNCTIONS = frozenset(_REDUCTIONS) | {
    WindowOperator.ANY,
    WindowOperator.ALL,
    WindowOperator.MIN,
    WindowOperator.MAX,
}
_AST_OPERATORS: Mapping[type[ast.AST], WindowOperator] = {
    ast.Add: WindowOperator.ADD,
    ast.Sub: WindowOperator.SUBTRACT,
    ast.Mult: WindowOperator.MULTIPLY,
    ast.Div: WindowOperator.DIVIDE,
    ast.Mod: WindowOperator.MODULO,
    ast.Pow: WindowOperator.POWER,
    ast.USub: WindowOperator.NEGATE,
    ast.UAdd: WindowOperator.POSITIVE,
    ast.Not: WindowOperator.NOT,
    ast.And: WindowOperator.AND,
    ast.Or: WindowOperator.OR,
    ast.Eq: WindowOperator.EQ,
    ast.NotEq: WindowOperator.NE,
    ast.Lt: WindowOperator.LT,
    ast.LtE: WindowOperator.LE,
    ast.Gt: WindowOperator.GT,
    ast.GtE: WindowOperator.GE,
    ast.In: WindowOperator.IN,
    ast.NotIn: WindowOperator.NOT_IN,
    ast.Is: WindowOperator.IS,
    ast.IsNot: WindowOperator.IS_NOT,
}


@dataclass(frozen=True)
class WindowLiteral:
    """A fixed scalar, including a missing value, within a parsed measurement expression."""

    value: WindowScalar


@dataclass(frozen=True)
class WindowColumn:
    """A named source-column reference within a parsed measurement expression."""

    name: str


@dataclass(frozen=True)
class WindowCollection:
    """A literal collection used by membership and helper operations in a measurement expression."""

    values: tuple[WindowScalar, ...]


@dataclass(frozen=True)
class WindowOperation:
    """A permitted measurement operator applied to its ordered child expressions."""

    operator: WindowOperator
    arguments: tuple[WindowTerm, ...]


type WindowTerm = WindowLiteral | WindowColumn | WindowCollection | WindowOperation


def _operator(node: ast.AST) -> WindowOperator:
    operator = _AST_OPERATORS.get(type(node))
    if operator is None:
        raise ValueError(f"Unsupported computed_rule operator: {type(node).__name__}")
    return operator


def _literal(node: ast.AST) -> WindowScalar:
    if not isinstance(node, ast.Constant) or not isinstance(
        node.value, (str, int, float, type(None))
    ):
        raise ValueError("computed_rule literal collections support only scalar constant values")
    return node.value


def _parse_term(node: ast.AST) -> WindowTerm:
    match node:
        case ast.Constant():
            return WindowLiteral(_literal(node))
        case ast.Name():
            return WindowColumn(node.id)
        case ast.List() | ast.Tuple() | ast.Set():
            return WindowCollection(tuple(_literal(item) for item in node.elts))
        case ast.BinOp():
            return WindowOperation(
                _operator(node.op), (_parse_term(node.left), _parse_term(node.right))
            )
        case ast.UnaryOp():
            return WindowOperation(_operator(node.op), (_parse_term(node.operand),))
        case ast.BoolOp():
            return WindowOperation(
                _operator(node.op), tuple(_parse_term(item) for item in node.values)
            )
        case ast.Compare():
            operands = (node.left, *node.comparators)
            comparisons = []
            for left, op, right in zip(operands[:-1], node.ops, node.comparators, strict=True):
                operator = _operator(op)
                if operator in {WindowOperator.IS, WindowOperator.IS_NOT} and not (
                    isinstance(right, ast.Constant) and right.value is None
                ):
                    raise ValueError("computed_rule only supports 'is None' and 'is not None'")
                if operator in {WindowOperator.IN, WindowOperator.NOT_IN} and not isinstance(
                    right, (ast.List, ast.Tuple, ast.Set)
                ):
                    raise ValueError(
                        "computed_rule 'in' comparisons require a literal list/tuple/set"
                    )
                comparisons.append(
                    WindowOperation(operator, (_parse_term(left), _parse_term(right)))
                )
            return (
                comparisons[0]
                if len(comparisons) == 1
                else WindowOperation(WindowOperator.AND, tuple(comparisons))
            )
        case ast.IfExp():
            return WindowOperation(
                WindowOperator.IF,
                tuple(_parse_term(item) for item in (node.test, node.body, node.orelse)),
            )
        case ast.Call():
            if not isinstance(node.func, ast.Name):
                raise ValueError("computed_rule only supports simple function calls")
            operator = next((item for item in _FUNCTIONS if item == node.func.id), None)
            if operator is None:
                raise ValueError(f"Unsupported computed_rule function '{node.func.id}'")
            if node.keywords:
                raise ValueError("computed_rule does not support keyword arguments")
            count = len(node.args)
            if operator == WindowOperator.COALESCE:
                if count < 2:
                    raise ValueError(
                        "computed_rule function 'coalesce' expects at least 2 arguments"
                    )
            elif count != (
                2 if operator in {WindowOperator.CONTAINS, WindowOperator.CONTAINS_ANY} else 1
            ):
                raise ValueError(f"computed_rule function '{operator}' has invalid argument count")
            if operator == WindowOperator.CONTAINS and not isinstance(_literal(node.args[1]), str):
                raise ValueError(
                    "computed_rule function 'contains' requires a literal string pattern"
                )
            arguments = tuple(_parse_term(item) for item in node.args)
            if operator == WindowOperator.CONTAINS_ANY and not (
                isinstance(arguments[1], WindowCollection)
                and all(isinstance(value, str) for value in arguments[1].values)
            ):
                raise ValueError(
                    "computed_rule function 'contains_any' requires a literal list of strings"
                )
            return WindowOperation(operator, arguments)
        case _:
            raise ValueError(f"Unsupported computed_rule syntax: {type(node).__name__}")


def _walk(term: WindowTerm) -> Iterator[WindowTerm]:
    yield term
    if isinstance(term, WindowOperation):
        for argument in term.arguments:
            yield from _walk(argument)


def _scalar(term: WindowTerm) -> bool:
    match term:
        case WindowColumn():
            return False
        case WindowOperation(operator, arguments):
            return operator in _WINDOW_FUNCTIONS or all(_scalar(argument) for argument in arguments)
        case WindowLiteral() | WindowCollection():
            return True


def _number(term: WindowTerm) -> bool:
    match term:
        case WindowLiteral(value):
            return type(value) in (int, float)
        case WindowOperation(operator, (argument,)) if operator in {
            WindowOperator.NEGATE,
            WindowOperator.POSITIVE,
        }:
            return _number(argument)
        case _:
            return False


def _summary(term: WindowTerm) -> SummaryOperator:
    match term:
        case WindowOperation(operator, (argument,)):
            if operator in _REDUCTIONS and not any(
                isinstance(child, WindowOperation) and child.operator in _WINDOW_FUNCTIONS
                for child in _walk(argument)
            ):
                return _REDUCTIONS[operator]
            if operator in {WindowOperator.NEGATE, WindowOperator.POSITIVE}:
                return _summary(argument)
        case WindowOperation(operator, (left, right)) if operator in {
            WindowOperator.ADD,
            WindowOperator.SUBTRACT,
            WindowOperator.MULTIPLY,
            WindowOperator.DIVIDE,
        }:
            if _number(right):
                return _summary(left)
            if _number(left) and operator != WindowOperator.DIVIDE:
                return _summary(right)
        case WindowOperation(WindowOperator.IF, (guard, body, otherwise)) if _scalar(guard):
            if body == WindowLiteral(None):
                return _summary(otherwise)
            if otherwise == WindowLiteral(None):
                return _summary(body)
        case _:
            pass
    raise ValueError(
        "computed_rule requires one supported window summary (first, last, sum, mean, std, "
        "count_true, or count_non_null), optionally with unit conversion or a missingness guard. "
        "Put row transforms inside the summary; other window formulas have no supported observation semantics."
    )


@dataclass(frozen=True, init=False)
class WindowExpression:
    """Canonical source and its single owned parse, including source dependencies."""

    source: str
    root: WindowTerm
    dependencies: frozenset[str]
    summary_operator: SummaryOperator

    def __init__(self, source: str) -> None:
        """Parse a computed rule into the restricted expression tree and record its column dependencies."""
        try:
            parsed = ast.parse(source, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid computed_rule: {exc.msg}") from exc
        root = _parse_term(parsed.body)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "root", root)
        object.__setattr__(
            self,
            "dependencies",
            frozenset(term.name for term in _walk(root) if isinstance(term, WindowColumn)),
        )
        object.__setattr__(self, "summary_operator", _summary(root))

    @override
    def __str__(self) -> str:
        return self.source

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: object, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Accept parsed window expressions or source strings and serialize the original rule text."""
        parsed = core_schema.no_info_after_validator_function(cls, core_schema.str_schema())
        return core_schema.json_or_python_schema(
            ref=f"{cls.__module__}.{cls.__qualname__}",
            metadata={
                "pydantic_js_updates": {
                    "description": (
                        "Deterministic support-window expression that returns one scalar per window. "
                        "Use Python-like syntax over source_columns with arithmetic, comparisons, "
                        "if/else, and helper functions such as any(), sum(), mean(), std(), "
                        "first(), last(), count_true(), count_non_null(), lower(), contains(), "
                        "and contains_any(). Use None for missing values."
                    )
                }
            },
            json_schema=parsed,
            python_schema=core_schema.union_schema([core_schema.is_instance_schema(cls), parsed]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                str, return_schema=core_schema.str_schema()
            ),
        )

    def fold[T](
        self,
        *,
        literal: Callable[[WindowScalar], T],
        column: Callable[[str], T],
        collection: Callable[[tuple[WindowScalar, ...]], T],
        operation: Callable[[WindowOperator, tuple[T, ...]], T],
    ) -> T:
        """Interpret the parsed expression by supplying one operation for each node kind.

        Args:
            literal: Interpreter for scalar constants, including missing values.
            column: Interpreter for a named source-column reference.
            collection: Interpreter for a tuple of literal values.
            operation: Interpreter receiving an operator and its already interpreted
                arguments in expression order.

        Returns:
            The root result after recursively interpreting its children.
        """

        def _visit(term: WindowTerm) -> T:
            match term:
                case WindowLiteral(value):
                    return literal(value)
                case WindowColumn(name):
                    return column(name)
                case WindowCollection(values):
                    return collection(values)
                case WindowOperation(operator, arguments):
                    return operation(operator, tuple(_visit(argument) for argument in arguments))
                case _:
                    assert_never(term)

        return _visit(self.root)
