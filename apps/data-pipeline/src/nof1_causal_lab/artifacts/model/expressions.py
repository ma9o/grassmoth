"""Scientific expressions shared by dynamics, conditional laws, bindings, and equations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Literal, cast

from pydantic import ConfigDict, Field, FiniteFloat, model_validator
from sympy import Add, Basic, Expr, Float, Function, Mul, Pow, Symbol

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import ConstructId, ParameterId
from nof1_causal_lab.artifacts.model.parameter import SiteKind, SupportClass

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

type CoefficientRole = Literal[
    "center",
    "decay",
    "quartic",
    "intercept",
    "weight",
    "emax",
    "ec50",
    "exponent",
    "loading",
    "observation_intercept",
    "observation_scale",
    "degrees_of_freedom",
    "shape",
    "dispersion",
    "concentration",
    "cutpoint_base",
    "cutpoint_gaps",
    "category_intercepts",
    "category_slopes",
    "diffusion_scale",
    "diffusion_loading",
    "process_degrees_of_freedom",
    "initial_mean",
    "initial_scale",
    "initial_correlation",
]
CONSTRUCT_COEFFICIENT_ROLES: frozenset[CoefficientRole] = frozenset(
    {
        "diffusion_scale",
        "diffusion_loading",
        "process_degrees_of_freedom",
        "initial_mean",
        "initial_scale",
        "initial_correlation",
    }
)
type BinaryOperator = Literal["add", "subtract", "multiply", "divide", "power", "maximum"]
type ExpressionFunction = Literal[
    "exp", "sigmoid", "normal_cdf", "ordered_cutpoints", "category_logits"
]


@dataclass(frozen=True)
class CoefficientMeaning:
    """Scientific meaning and support of one scalar operand."""

    quantity: SiteKind
    support: SupportClass
    allows_zero: bool = False


COEFFICIENT_MEANINGS: Mapping[CoefficientRole, CoefficientMeaning] = {
    "center": CoefficientMeaning(SiteKind.DYNAMICS_POTENTIAL_CENTER, SupportClass.REAL),
    "decay": CoefficientMeaning(SiteKind.DYNAMICS_DECAY, SupportClass.POSITIVE),
    "quartic": CoefficientMeaning(
        SiteKind.DYNAMICS_POTENTIAL_QUARTIC, SupportClass.POSITIVE, allows_zero=True
    ),
    "intercept": CoefficientMeaning(SiteKind.DYNAMICS_CINT, SupportClass.REAL),
    "weight": CoefficientMeaning(SiteKind.DYNAMICS_WEIGHT, SupportClass.REAL),
    "emax": CoefficientMeaning(SiteKind.HILL_EMAX, SupportClass.POSITIVE),
    "ec50": CoefficientMeaning(SiteKind.HILL_EC50, SupportClass.POSITIVE),
    "exponent": CoefficientMeaning(SiteKind.HILL_N, SupportClass.POSITIVE),
    "loading": CoefficientMeaning(SiteKind.LOADING, SupportClass.REAL),
    "observation_intercept": CoefficientMeaning(SiteKind.MANIFEST_MEANS, SupportClass.REAL),
    "observation_scale": CoefficientMeaning(
        SiteKind.MANIFEST_VAR_DIAG, SupportClass.POSITIVE, allows_zero=True
    ),
    "degrees_of_freedom": CoefficientMeaning(SiteKind.OBS_DF, SupportClass.POSITIVE),
    "shape": CoefficientMeaning(SiteKind.OBS_SHAPE, SupportClass.POSITIVE),
    "dispersion": CoefficientMeaning(SiteKind.OBS_R, SupportClass.POSITIVE),
    "concentration": CoefficientMeaning(SiteKind.OBS_CONCENTRATION, SupportClass.POSITIVE),
    "cutpoint_base": CoefficientMeaning(SiteKind.OBS_ORDERED_BASE, SupportClass.REAL),
    "cutpoint_gaps": CoefficientMeaning(SiteKind.OBS_ORDERED_GAPS, SupportClass.POSITIVE),
    "category_intercepts": CoefficientMeaning(SiteKind.OBS_CAT_INTERCEPTS, SupportClass.REAL),
    "category_slopes": CoefficientMeaning(SiteKind.OBS_CAT_SLOPES, SupportClass.REAL),
    "diffusion_scale": CoefficientMeaning(
        SiteKind.DIFFUSION_DIAG, SupportClass.POSITIVE, allows_zero=True
    ),
    "diffusion_loading": CoefficientMeaning(SiteKind.DIFFUSION_LOWER, SupportClass.REAL),
    "process_degrees_of_freedom": CoefficientMeaning(SiteKind.PROC_DF, SupportClass.POSITIVE),
    "initial_mean": CoefficientMeaning(SiteKind.T0_MEANS, SupportClass.REAL),
    "initial_scale": CoefficientMeaning(
        SiteKind.T0_VAR_DIAG, SupportClass.POSITIVE, allows_zero=True
    ),
    "initial_correlation": CoefficientMeaning(SiteKind.T0_VAR_LOWER, SupportClass.CORRELATION),
}


class _ExpressionValue(Value):
    """An immutable scalar expression, with arithmetic for scientific constructors."""

    model_config = ConfigDict(revalidate_instances="always")

    def __add__(self: Expression, other: Expression | float) -> BinaryExpression:  # pyright: ignore[reportGeneralTypeIssues] -- This private operator mixin binds only to the closed Expression variants.
        return BinaryExpression(operator="add", left=self, right=expression(other))

    def __sub__(self: Expression, other: Expression | float) -> BinaryExpression:  # pyright: ignore[reportGeneralTypeIssues] -- This private operator mixin binds only to the closed Expression variants.
        return BinaryExpression(operator="subtract", left=self, right=expression(other))

    def __mul__(self: Expression, other: Expression | float) -> BinaryExpression:  # pyright: ignore[reportGeneralTypeIssues] -- This private operator mixin binds only to the closed Expression variants.
        return BinaryExpression(operator="multiply", left=self, right=expression(other))

    def __truediv__(self: Expression, other: Expression | float) -> BinaryExpression:  # pyright: ignore[reportGeneralTypeIssues] -- This private operator mixin binds only to the closed Expression variants.
        return BinaryExpression(operator="divide", left=self, right=expression(other))

    def __pow__(self: Expression, other: Expression | float) -> BinaryExpression:  # pyright: ignore[reportGeneralTypeIssues] -- This private operator mixin binds only to the closed Expression variants.
        return BinaryExpression(operator="power", left=self, right=expression(other))

    def __neg__(self: Expression) -> BinaryExpression:  # pyright: ignore[reportGeneralTypeIssues] -- This private operator mixin binds only to the closed Expression variants.
        return LiteralExpression(value=-1) * self


class LiteralExpression(_ExpressionValue):
    """A finite scalar constant in a model equation."""

    kind: Literal["literal"] = "literal"
    value: FiniteFloat


class StateExpression(_ExpressionValue):
    """A construct's state or declared known input, referenced by identity."""

    kind: Literal["state"] = "state"
    construct_id: ConstructId


class CoefficientExpression(_ExpressionValue):
    """A scientifically typed coefficient operand, literal or parameter reference."""

    kind: Literal["coefficient"] = "coefficient"
    role: CoefficientRole
    value: FiniteFloat | ParameterId | None = Field(
        default=None,
        description=(
            "Fixed coefficients are finite literals; uncertain coefficients reference a "
            "persistent parameter ID. Null leaves the operand unassigned."
        ),
    )
    construct_ids: tuple[ConstructId, ...] = Field(
        default=(), description="Additional constructs participating in this coefficient use."
    )

    @property
    def meaning(self) -> CoefficientMeaning:
        """Role-owned coefficient semantics, including its admissible numeric support."""
        return COEFFICIENT_MEANINGS[self.role]

    def validate_value(self, value: float) -> None:
        """Reject a numeric coefficient outside the support declared by its semantic role."""
        meaning = self.meaning
        if meaning.support == SupportClass.POSITIVE and (
            value < 0 if meaning.allows_zero else value <= 0
        ):
            requirement = "non-negative" if meaning.allows_zero else "positive"
            raise ValueError(f"{self.role} must be {requirement}; got {value}")

    @model_validator(mode="after")
    def validate_fixed_support(self) -> CoefficientExpression:
        """Check numeric coefficient literals at construction; symbolic values retain their owner."""
        if isinstance(self.value, (int, float)):
            self.validate_value(self.value)
        return self


class BinaryExpression(_ExpressionValue):
    """A supported scalar operation composing two expressions."""

    kind: Literal["binary"] = "binary"
    operator: BinaryOperator
    left: Expression
    right: Expression


class CallExpression(_ExpressionValue):
    """A supported mathematical function, including explicit discrete contrasts."""

    kind: Literal["call"] = "call"
    function: ExpressionFunction
    arguments: tuple[Expression, ...]

    @model_validator(mode="after")
    def validate_arity(self) -> CallExpression:
        """Reject calls whose argument count differs from the expression function's declared arity."""
        expected = {"ordered_cutpoints": 2, "category_logits": 3}.get(self.function, 1)
        if len(self.arguments) != expected:
            raise ValueError(f"{self.function} requires {expected} arguments")
        return self


type Expression = Annotated[
    LiteralExpression | StateExpression | CoefficientExpression | BinaryExpression | CallExpression,
    Field(discriminator="kind"),
]


def expression(value: Expression | float) -> Expression:
    """Wrap numeric constants as literal expression nodes and preserve existing expression owners."""
    if isinstance(value, (float, int)):
        return LiteralExpression(value=value)
    return value


def state(identity: ConstructId) -> StateExpression:
    """Build an expression referring to the latent state of the named construct."""
    return StateExpression(construct_id=identity)


def coefficient(
    value: float | ParameterId | None,
    role: CoefficientRole,
    *,
    construct_ids: tuple[ConstructId, ...] = (),
) -> CoefficientExpression:
    """Build a role-bearing coefficient from a fixed value, parameter reference, or unresolved value."""
    return CoefficientExpression(role=role, value=value, construct_ids=construct_ids)


def restoring_force(
    target: ConstructId,
    *,
    center: float | ParameterId | None,
    stiffness: float | ParameterId | None,
    quartic: float | ParameterId | None,
) -> Expression:
    """Restoring drift -stiffness * (x - center) - quartic * (x - center)^3."""
    delta = state(target) - coefficient(center, "center")
    return -coefficient(stiffness, "decay") * delta - coefficient(quartic, "quartic") * delta**3


def restoring_potential(
    target: ConstructId,
    *,
    center: float | ParameterId | None,
    stiffness: float | ParameterId | None,
    quartic: float | ParameterId | None,
) -> Expression:
    """Scalar node energy; Dynestyx differentiates it to obtain the restoring drift."""
    delta = state(target) - coefficient(center, "center")
    return (
        coefficient(stiffness, "decay") * delta**2 / 2
        + coefficient(quartic, "quartic") * delta**4 / 4
    )


def hill(
    source: Expression,
    *,
    emax: float | ParameterId | None,
    ec50: float | ParameterId | None,
    n: float | ParameterId | None,
) -> Expression:
    """The native non-negative Hill response, including its numerical denominator term."""
    dose = BinaryExpression(operator="maximum", left=source, right=LiteralExpression(value=0.0))
    exponent = coefficient(n, "exponent")
    powered = dose**exponent
    return (
        coefficient(emax, "emax")
        * powered
        / (coefficient(ec50, "ec50") ** exponent + powered + 1e-12)
    )


class SymbolicExpression(Value):
    """Unevaluated SymPy arithmetic with the authored nodes as its identity legend."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    root: Expr
    nodes: Mapping[Basic, Expression]
    operands: Mapping[Symbol, StateExpression | CoefficientExpression]


def symbolic_call(name: str, *arguments: Expr) -> Expr:
    """SymPy returns a callable undefined-function class from its dynamic factory."""
    return cast("Callable[..., Expr]", Function(name))(*arguments)


def symbolic_expression(value: Expression) -> SymbolicExpression:
    """Lower once without cancelling operands, reordering uses, or erasing their roles."""
    nodes: dict[Basic, Expression] = {}
    operands: dict[Symbol, StateExpression | CoefficientExpression] = {}

    def lower(node: Expression) -> Expr:
        match node:
            case LiteralExpression():
                result = Float(node.value)
            case StateExpression() | CoefficientExpression():
                result = Symbol(node.model_dump_json())
                operands[result] = node
            case BinaryExpression():
                left, right = lower(node.left), lower(node.right)
                match node.operator:
                    case "add":
                        result = Add(left, right, evaluate=False)
                    case "subtract":
                        result = Add(left, Mul(-1, right, evaluate=False), evaluate=False)
                    case "multiply":
                        result = Mul(left, right, evaluate=False)
                    case "divide":
                        result = Mul(left, Pow(right, -1, evaluate=False), evaluate=False)
                    case "power":
                        result = Pow(left, right, evaluate=False)
                    case "maximum":
                        result = symbolic_call("maximum", left, right)
            case CallExpression():
                result = symbolic_call(node.function, *(lower(arg) for arg in node.arguments))
        nodes[result] = node
        return result

    root = lower(value)
    return SymbolicExpression(
        root=root, nodes=MappingProxyType(nodes), operands=MappingProxyType(operands)
    )


def walk_expression(value: Expression) -> Iterator[Expression]:
    """Yield authored expression nodes in preorder without lowering to symbolic algebra."""
    yield value
    match value:
        case LiteralExpression() | StateExpression() | CoefficientExpression():
            return
        case BinaryExpression(left=left, right=right):
            yield from walk_expression(left)
            yield from walk_expression(right)
        case CallExpression(arguments=arguments):
            for argument in arguments:
                yield from walk_expression(argument)


def expression_states(value: Expression) -> frozenset[ConstructId]:
    """Collect the distinct construct identities referenced by state nodes in an expression."""
    return frozenset(
        node.construct_id for node in walk_expression(value) if isinstance(node, StateExpression)
    )


def expression_coefficients(value: Expression) -> tuple[CoefficientExpression, ...]:
    """One use per distinct operand; formulas can mention a coefficient repeatedly."""
    return tuple(
        dict.fromkeys(
            node for node in walk_expression(value) if isinstance(node, CoefficientExpression)
        )
    )


def linear_coefficient(value: Expression, source: ConstructId) -> float | ParameterId:
    """Recognize the scalar linear form supported by input and projection matrices."""
    if isinstance(value, BinaryExpression) and value.operator == "multiply":
        for lhs, rhs in ((value.left, value.right), (value.right, value.left)):
            if (
                isinstance(lhs, CoefficientExpression)
                and lhs.role == "weight"
                and rhs == state(source)
            ):
                if lhs.value is None:
                    from nof1_causal_lab.compilation_errors import IncompleteModelError

                    raise IncompleteModelError("Linear expression requires its coefficient")
                return lhs.value
    raise ValueError("This execution boundary requires one scalar linear coefficient")


def restoring_coefficients(
    value: Expression, target: ConstructId, *, kind: Literal["drift", "potential"] = "drift"
) -> tuple[CoefficientExpression, ...]:
    """Recognize restoring terms for the existing anchoring and scale checks.

    A role annotation alone is not a certificate. The complete term must equal
    the scientific constructor before it can establish an anchor.
    """
    operands = expression_coefficients(value)
    roles = {node.role: node for node in operands}
    constructor = restoring_potential if kind == "potential" else restoring_force
    if (
        len(operands) == 3
        and roles.keys() == {"center", "decay", "quartic"}
        and value
        == constructor(
            target,
            center=roles["center"].value,
            stiffness=roles["decay"].value,
            quartic=roles["quartic"].value,
        )
    ):
        return operands
    if isinstance(value, BinaryExpression) and value.operator == "add":
        return (
            *restoring_coefficients(value.left, target, kind=kind),
            *restoring_coefficients(value.right, target, kind=kind),
        )
    return ()


def hill_applications(
    value: Expression,
) -> Iterator[tuple[Expression, CoefficientExpression, CoefficientExpression]]:
    """Recognize complete Hill applications for saturation diagnostics, including compositions."""
    for node in walk_expression(value):
        if not isinstance(node, BinaryExpression) or node.operator != "divide":
            continue
        numerator = node.left
        if not isinstance(numerator, BinaryExpression) or numerator.operator != "multiply":
            continue
        emax, power = numerator.left, numerator.right
        if not isinstance(emax, CoefficientExpression) or emax.role != "emax":
            continue
        if not isinstance(power, BinaryExpression) or power.operator != "power":
            continue
        dose, exponent = power.left, power.right
        if not isinstance(exponent, CoefficientExpression) or exponent.role != "exponent":
            continue
        if not isinstance(dose, BinaryExpression) or dose.operator != "maximum":
            continue
        for operand in expression_coefficients(node.right):
            if operand.role == "ec50" and node == hill(
                dose.left, emax=emax.value, ec50=operand.value, n=exponent.value
            ):
                yield dose.left, operand, exponent
