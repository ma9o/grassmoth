"""Scientific constructs own observations, intrinsic dynamics, and authored usage."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import (
    ConfigDict,
    Field,
    PlainSerializer,
    ValidatorFunctionWrapHandler,
    WrapValidator,
    model_validator,
)

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import (
    ConstructId,
    ConstructRef,
    DistributionId,
    EdgeId,
    ParameterId,
)
from grassmoth.artifacts.model.evidence import LiteratureSource
from grassmoth.artifacts.model.expressions import (
    CONSTRUCT_COEFFICIENT_ROLES,
    CoefficientExpression,
    CoefficientRole,
    StateExpression,
)
from grassmoth.artifacts.model.indicator import IndicatorSpec
from grassmoth.artifacts.model.mechanism import DriftMechanismSpec, DynamicsMechanismSpec
from grassmoth.artifacts.probability.families import DistributionFamily

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class Role(StrEnum):
    """Whether a construct is modeled as endogenous or supplied as an exogenous input."""

    ENDOGENOUS = "endogenous"
    EXOGENOUS = "exogenous"


class TemporalStatus(StrEnum):
    """Temporal status states whether a construct varies within the individual over time."""

    TIME_VARYING = "time_varying"
    TIME_INVARIANT = "time_invariant"


class ConstructSpec(Value):
    """A specification of a theoretical entity in the scientific causal model."""

    model_config = ConfigDict(revalidate_instances="always")

    id: ConstructId = Field(description="Persistent identity. Preserve when revising or renaming.")
    name: str = Field(description="Construct name (e.g., 'stress', 'sleep_quality')")
    description: str = Field(description="What this theoretical construct represents")
    indicators: tuple[IndicatorSpec, ...] = ()
    dynamics: tuple[DynamicsMechanismSpec, ...] = ()
    coefficients: tuple[CoefficientExpression, ...] = ()
    innovation_family: Literal[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T] = (
        DistributionFamily.GAUSSIAN
    )
    distribution: DistributionId | None = Field(
        default=None,
        description="Trajectory law in DynamicalModelSpec.distributions, with coordinates in its law_layouts entry. Exogenous trajectories use Delta and hold each value until the next point, including after the last point.",
    )
    role: Role = Field(
        description="'endogenous' means modeled, with or without parents; 'exogenous' means supplied by a deterministic trajectory law, without endogenous dynamics or noise."
    )
    temporal_status: TemporalStatus = Field(
        description="'time_varying' (changes over time) or 'time_invariant' (fixed)"
    )

    def with_distribution(self, identity: DistributionId | None) -> ConstructSpec:
        """Return a new construct selecting the supplied trajectory distribution, or none."""
        return self.revised(distribution=identity)

    @property
    def is_dynamic(self) -> bool:
        """Whether this state advances across time slices, regardless of causal role."""
        return self.temporal_status == TemporalStatus.TIME_VARYING

    def coefficient(
        self, role: CoefficientRole, *, construct_ids: tuple[ConstructId, ...] = ()
    ) -> float | ParameterId | None:
        """Read one scalar use by its role and scientific references."""
        return next(
            (
                operand.value
                for operand in self.coefficients
                if operand.role == role and operand.construct_ids == construct_ids
            ),
            None,
        )

    @model_validator(mode="after")
    def validate_coefficients(self) -> ConstructSpec:
        """Reject coefficients and dynamics that conflict with the construct's role or innovation law."""
        if self.role == Role.EXOGENOUS:
            if self.dynamics or self.coefficients:
                raise ValueError(
                    "Exogenous constructs have no dynamics, diffusion or initial coefficients"
                )
            for indicator in self.indicators:
                likelihood = indicator.likelihood
                if (
                    likelihood is None
                    or likelihood.law.distribution != "Delta"
                    or not isinstance(expression := likelihood.law.v, StateExpression)
                    or expression.construct_id != self.id
                    or likelihood.standardized
                ):
                    raise ValueError(
                        "Exogenous readings require Delta(v=state(the owning construct)) in recorded units"
                    )
        seen = set()
        for operand in self.coefficients:
            if operand.role not in CONSTRUCT_COEFFICIENT_ROLES:
                raise ValueError(f"{operand.role} belongs in a dynamics or likelihood expression")
            key = (operand.role, operand.construct_ids)
            if key in seen:
                raise ValueError(f"Duplicate construct coefficient {key}")
            seen.add(key)
            coupled = operand.role in {"diffusion_loading", "initial_correlation"}
            if len(operand.construct_ids) != int(coupled) or self.id in operand.construct_ids:
                raise ValueError(
                    "Joint coefficients require one other construct; scalar uses require none"
                )
            if (
                operand.role == "process_degrees_of_freedom"
                and self.innovation_family != DistributionFamily.STUDENT_T
            ):
                raise ValueError("Only Student-t innovations have a degrees-of-freedom coefficient")
        return self


@dataclass
class _EndpointScope:
    definitions: dict[str, object] = field(default_factory=dict)
    constructs: dict[ConstructId, ConstructSpec] = field(default_factory=dict)


_endpoint_scope: ContextVar[_EndpointScope | None] = ContextVar("endpoint_scope", default=None)
_serialized_endpoints: ContextVar[set[ConstructId] | None] = ContextVar(
    "serialized_endpoints", default=None
)


def _validate_endpoint(value: object, handler: ValidatorFunctionWrapHandler) -> ConstructSpec:
    scope = _endpoint_scope.get()
    if isinstance(value, ConstructRef) or (
        isinstance(value, dict) and value.get("kind") == "construct"
    ):
        reference = value if isinstance(value, ConstructRef) else ConstructRef(**value)
        if scope is None or reference.id not in scope.definitions:
            raise ValueError(f"Undefined construct endpoint {reference.id!r}")
        if reference.id in scope.constructs:
            return scope.constructs[reference.id]
        value = scope.definitions[reference.id]
    construct: ConstructSpec = handler(value)
    if scope is None:
        return construct
    if construct.id in scope.constructs:
        canonical = scope.constructs[construct.id]
        if canonical.model_dump(mode="json") != construct.model_dump(mode="json"):
            raise ValueError(f"Conflicting definitions of construct {construct.id!r}")
        return canonical
    scope.constructs[construct.id] = construct
    return construct


def _serialize_endpoint(value: ConstructSpec) -> ConstructSpec | ConstructRef:
    seen = _serialized_endpoints.get()
    if seen is not None:
        if value.id in seen:
            return ConstructRef(id=value.id)
        seen.add(value.id)
    return value


ConstructEndpoint = Annotated[
    ConstructSpec,
    WrapValidator(_validate_endpoint, json_schema_input_type=ConstructSpec | ConstructRef),
    PlainSerializer(
        _serialize_endpoint, return_type=ConstructSpec | ConstructRef, when_used="json"
    ),
]


class CausalEdgeSpec(Value):
    """A specification of a directed causal relationship between two constructs."""

    model_config = ConfigDict(revalidate_instances="always")

    id: EdgeId = Field(description="Persistent identity. Preserve when revising the same edge.")
    mechanisms: tuple[DriftMechanismSpec, ...] = ()
    cause: ConstructEndpoint = Field(
        description="Cause construct; shared endpoints have one identity."
    )
    effect: ConstructEndpoint = Field(
        description="Effect construct; shared endpoints have one identity."
    )
    description: str = Field(description="Theoretical justification for this causal link")
    sources: tuple[LiteratureSource, ...] = Field(
        default_factory=tuple,
        description="Literature sources supporting this causal link",
    )

    def with_endpoints(self, cause: ConstructSpec, effect: ConstructSpec) -> Self:
        """Rebuild the edge with new endpoint owners while retaining its mechanisms and identity."""
        return type(self)(
            id=self.id,
            mechanisms=self.mechanisms,
            cause=cause,
            effect=effect,
            description=self.description,
            sources=self.sources,
        )


@contextmanager
def endpoint_validation_scope(
    values: object, *, endpoints: Iterable[ConstructSpec] = ()
) -> Iterator[None]:
    """Resolve forward and shared JSON references within exactly one causal graph."""
    scope = _EndpointScope()
    for endpoint in endpoints:
        scope.definitions[endpoint.id] = endpoint
        scope.constructs[endpoint.id] = endpoint
    if isinstance(values, (list, tuple)):
        for edge in values:
            if isinstance(edge, CausalEdgeSpec):
                edge_endpoints: tuple[object, object] = (edge.cause, edge.effect)
            elif isinstance(edge, dict):
                edge_endpoints = (edge.get("cause"), edge.get("effect"))
            else:
                continue  # The edge validator reports malformed edge values.
            for endpoint in edge_endpoints:
                if isinstance(endpoint, ConstructSpec):
                    scope.definitions.setdefault(endpoint.id, endpoint)
                elif (
                    isinstance(endpoint, dict)
                    and "name" in endpoint
                    and isinstance(identity := endpoint.get("id"), str)
                ):
                    scope.definitions.setdefault(identity, endpoint)
    token = _endpoint_scope.set(scope)
    try:
        yield
    finally:
        _endpoint_scope.reset(token)


@contextmanager
def endpoint_serialization_scope(referenced: Iterable[ConstructId] = ()) -> Iterator[None]:
    """Write each endpoint definition once, followed by references to its identity."""
    token = _serialized_endpoints.set(set(referenced))
    try:
        yield
    finally:
        _serialized_endpoints.reset(token)


def replace_constructs(
    edges: Iterable[CausalEdgeSpec], replacements: Iterable[ConstructSpec]
) -> tuple[CausalEdgeSpec, ...]:
    """Replace endpoint values throughout a graph without changing its membership."""
    edges = tuple(edges)
    by_id = {construct.id: construct for construct in replacements}
    identities = {endpoint.id for edge in edges for endpoint in (edge.cause, edge.effect)}
    if unknown := by_id.keys() - identities:
        raise ValueError(f"Cannot replace constructs absent from the graph: {sorted(unknown)}")
    return tuple(
        edge.with_endpoints(
            by_id.get(edge.cause.id, edge.cause), by_id.get(edge.effect.id, edge.effect)
        )
        for edge in edges
    )


def _check_edge_constraint(edge: CausalEdgeSpec) -> str | None:
    """Check a single edge against shared latent-structure constraints."""
    cause_construct = edge.cause
    effect_construct = edge.effect

    if effect_construct.role == Role.EXOGENOUS:
        return f"Exogenous construct '{effect_construct.name}' cannot be an effect"

    if (
        cause_construct.temporal_status == TemporalStatus.TIME_VARYING
        and effect_construct.temporal_status == TemporalStatus.TIME_INVARIANT
    ):
        return (
            f"Time-varying construct '{cause_construct.name}' cannot be a cause of "
            f"time-invariant construct '{effect_construct.name}'. Time-invariant constructs "
            "are fixed within person and cannot have time-varying parents."
        )

    return None


def _check_global_constraints(
    constructs: tuple[ConstructSpec, ...],
    edges: tuple[CausalEdgeSpec, ...],
) -> list[str]:
    """Check latent-structure global constraints."""
    errors: list[str] = []

    import networkx as nx

    graph = nx.Graph((edge.cause.id, edge.effect.id) for edge in edges)
    graph.add_nodes_from(construct.id for construct in constructs)
    if constructs and not nx.is_connected(graph):
        errors.append("The scientific model must be one connected causal graph")

    static_edges = [
        (edge.cause.id, edge.effect.id)
        for edge in edges
        if not edge.cause.is_dynamic and not edge.effect.is_dynamic
    ]
    if static_edges:
        graph = nx.DiGraph(static_edges)
        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            errors.append(
                f"Time-invariant edges form cycle(s): {cycles}. "
                "Feedback requires time-varying states."
            )

    return errors
