"""The scientific model: stable entities enriched by successive validated revisions."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, cast, override

from pydantic import (
    ConfigDict,
    Field,
    FiniteFloat,
    GetJsonSchemaHandler,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    computed_field,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import (
    ConstructId,
    DistributionId,
    EdgeId,
    IndicatorId,
    MechanismId,
    ParameterId,
)
from nof1_causal_lab.artifacts.model.construct import (
    CausalEdgeSpec,
    ConstructSpec,
    Role,
    TemporalStatus,
    _check_edge_constraint,
    _check_global_constraints,
    endpoint_serialization_scope,
    endpoint_validation_scope,
)
from nof1_causal_lab.artifacts.model.document import (
    diff_fields,
    document_schema,
    entity_document,
    entity_fields,
    merge_fields,
    parameter_references,
)
from nof1_causal_lab.artifacts.model.expressions import expression_states
from nof1_causal_lab.artifacts.model.parameter_spec import ParameterSpec
from nof1_causal_lab.artifacts.observations.duration import Duration
from nof1_causal_lab.artifacts.probability.joint_layout import JointLawLayout
from nof1_causal_lab.artifacts.probability.serialization import (
    NumPyroDistribution,
    NumPyroObject,
    serialize_constructor,
)
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.json_types import JsonObject

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import CoreSchema

    from nof1_causal_lab.artifacts.model.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.model.likelihood import LikelihoodSpec
    from nof1_causal_lab.artifacts.model.mechanism import DynamicsMechanismSpec
    from nof1_causal_lab.artifacts.model.parameter_context import ParameterContext


class _ModelEntities(Value):
    """A connected causal graph with owned scientific detail, built to answer the study question."""

    constructs: tuple[ConstructSpec, ...] = ()
    edges: tuple[CausalEdgeSpec, ...] = ()
    parameters: tuple[ParameterSpec, ...] = ()
    distributions: Mapping[DistributionId, NumPyroDistribution] = Field(
        default_factory=dict,
        description=(
            "All explicit probability laws. Members are the parameters and constructs referring to each ID. "
            "Event coordinates are parameters by ID and element ID, then constructs by ID "
            "and time point. A scalar law belongs to one parameter and applies independently to its elements."
        ),
    )
    law_layouts: Mapping[DistributionId, JointLawLayout] = Field(
        default_factory=dict,
        description="Scientific coordinates and production labels of each joint law, beside its native atoms.",
    )

    @model_validator(mode="before")
    @classmethod
    def parse_document(cls, value: object) -> object:
        """Expand document identities inside the complete-model parsing boundary."""
        if not isinstance(value, DynamicalModelSpec):
            return value
        return {
            "constructs": tuple(
                entity_fields(identity, value)
                for identity, value in value.construct_definitions.items()
                if value is not None
            ),
            "edges": tuple(
                {
                    **entity_fields(identity, value),
                    **{
                        name: {"kind": "construct", "id": value[name]}
                        for name in ("cause", "effect")
                        if name in value
                    },
                }
                for identity, value in value.edge_definitions.items()
                if value is not None
            ),
            "parameters": tuple(
                entity_fields(identity, value)
                for identity, value in value.parameter_definitions.items()
                if value is not None
            ),
            "distributions": value.distribution_definitions,
            "law_layouts": value.layout_definitions,
            "measurement_clock": value.measurement_clock,
        }

    @computed_field
    @property
    def time_points(self) -> tuple[FiniteFloat, ...]:
        """Endogenous trajectory coordinates own the model's retained state grid."""
        return next(
            (
                layout.time_points
                for layout in self.law_layouts.values()
                if any(self.get_construct(key).role == Role.ENDOGENOUS for key in layout.constructs)
            ),
            (),
        )

    measurement_clock: Duration | None = None

    @field_validator("edges", mode="wrap")
    @classmethod
    def resolve_endpoints(
        cls, value: object, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
    ) -> tuple[CausalEdgeSpec, ...]:
        """Resolve edge endpoints within a shared construction scope so edges retain their owners."""
        if "constructs" not in info.data:
            raise ValueError("Edges require valid construct definitions")
        with endpoint_validation_scope(value, endpoints=info.data["constructs"]):
            edges: tuple[CausalEdgeSpec, ...] = handler(value)
            return edges

    @field_serializer("edges", mode="wrap")
    def serialize_edges(  # noqa: ANN202 -- Pydantic wrap serialization must retain the edge field schema; a return annotation replaces it.
        self, value: tuple[CausalEdgeSpec, ...], handler: SerializerFunctionWrapHandler
    ):
        """Serialize edges within the endpoint scope while preserving their declared field schema."""
        with endpoint_serialization_scope():
            edges: tuple[CausalEdgeSpec, ...] = handler(value)
            return edges

    @override
    def __eq__(self, other: object) -> bool:
        """Scientific equality excludes derived caches and their references back to this model."""
        return isinstance(other, _ModelEntities) and bool(
            self.model_dump(mode="json") == other.model_dump(mode="json")
        )

    @cached_property
    def _constructs(self) -> Mapping[ConstructId, ConstructSpec]:
        return MappingProxyType({construct.id: construct for construct in self.constructs})

    @cached_property
    def _edges(self) -> Mapping[EdgeId, CausalEdgeSpec]:
        return MappingProxyType({item.id: item for item in self.edges})

    @cached_property
    def _indicators(self) -> Mapping[IndicatorId, IndicatorSpec]:
        return MappingProxyType({item.observation.id: item for _, item in self.iter_indicators()})

    @cached_property
    def _indicator_owners(self) -> Mapping[IndicatorId, ConstructSpec]:
        return MappingProxyType(
            {item.observation.id: owner for owner, item in self.iter_indicators()}
        )

    @cached_property
    def _parameters(self) -> Mapping[ParameterId, ParameterSpec]:
        return MappingProxyType({item.id: item for item in self.parameters})

    @cached_property
    def _mechanisms(self) -> Mapping[MechanismId, DynamicsMechanismSpec]:
        return MappingProxyType({item.id: item for _, item in self.iter_mechanisms()})

    @cached_property
    def _parameter_contexts(self) -> Mapping[ParameterId, ParameterContext]:
        from nof1_causal_lab.artifacts.model.parameter_context import parameter_contexts

        return MappingProxyType(dict(parameter_contexts(self)))

    def parameter_context(self, identity: ParameterId) -> ParameterContext:
        """Look up a parameter's quantity context and scientific owners by its model-local identity."""
        return self._parameter_contexts[identity]

    def get_construct(self, identity: ConstructId) -> ConstructSpec:
        """Look up the construct owned by this model; an unknown identity raises ``KeyError``."""
        return self._constructs[identity]

    def distribution_for(self, identity: ParameterId | ConstructId) -> NumPyroDistribution | None:
        """Resolve the law a quantity participates in, retaining its full joint dependence."""
        quantity = (
            self.parameter(cast("ParameterId", identity))
            if identity.startswith("parameter:")
            else self.get_construct(cast("ConstructId", identity))
        )
        return (
            self.distributions[quantity.distribution] if quantity.distribution is not None else None
        )

    def edge(self, identity: EdgeId) -> CausalEdgeSpec:
        """Look up the causal edge owned by this model; an unknown identity raises ``KeyError``."""
        return self._edges[identity]

    def indicator(self, identity: IndicatorId) -> IndicatorSpec:
        """Look up the measurement indicator by observation ID; unknown IDs raise ``KeyError``."""
        return self._indicators[identity]

    def indicator_owner(self, identity: IndicatorId) -> ConstructSpec:
        """Look up the construct that owns the named observation indicator."""
        return self._indicator_owners[identity]

    def parameter(self, identity: ParameterId) -> ParameterSpec:
        """Look up the parameter definition by its scientific identity."""
        return self._parameters[identity]

    def mechanism(self, identity: MechanismId) -> DynamicsMechanismSpec:
        """Look up the dynamics mechanism owned by this model using its mechanism identity."""
        return self._mechanisms[identity]

    def iter_indicators(self) -> Iterator[tuple[ConstructSpec, IndicatorSpec]]:
        """Yield each indicator paired with its owning construct in authored order."""
        for construct in self.constructs:
            for indicator in construct.indicators:
                yield construct, indicator

    def iter_likelihoods(self) -> Iterator[tuple[IndicatorSpec, LikelihoodSpec]]:
        """Yield indicators with declared likelihoods, paired with their likelihood definitions."""
        for _, indicator in self.iter_indicators():
            if indicator.likelihood is not None:
                yield indicator, indicator.likelihood

    def iter_mechanisms(
        self,
    ) -> Iterator[tuple[ConstructSpec | CausalEdgeSpec, DynamicsMechanismSpec]]:
        """Yield construct dynamics followed by edge mechanisms, each paired with its owner."""
        for construct in self.constructs:
            for mechanism in construct.dynamics:
                yield construct, mechanism
        for edge in self.edges:
            for mechanism in edge.mechanisms:
                yield edge, mechanism

    def parameters_for(
        self, identity: ConstructId | EdgeId | IndicatorId | MechanismId
    ) -> tuple[ParameterSpec, ...]:
        """Select parameters whose recorded ownership includes the named scientific entity."""
        return tuple(
            parameter
            for parameter in self.parameters
            if any(owner.id == identity for owner in self.parameter_context(parameter.id).owners)
        )

    @property
    def indicators(self) -> tuple[IndicatorSpec, ...]:
        """All observation indicators in construct order and each construct's authored order."""
        return tuple(indicator for _, indicator in self.iter_indicators())

    @model_validator(mode="after")
    def validate_references(self) -> _ModelEntities:
        """Enforce model-wide reference, distribution-membership, and coefficient invariants."""
        references = {
            entity.distribution
            for entity in (*self.parameters, *self.constructs)
            if entity.distribution is not None
        }
        if references != set(self.distributions):
            raise ValueError("Distributions must be referenced and every reference must exist")
        observations = tuple(item.observation for item in self.indicators)
        for label, items in (
            ("construct", self.constructs),
            ("edge", self.edges),
            ("indicator", observations),
            ("parameter", self.parameters),
            ("mechanism", tuple(item for _, item in self.iter_mechanisms())),
        ):
            ids = [item.id for item in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate {label} IDs")
        for label, items in (("construct", self.constructs), ("indicator", observations)):
            if len({item.name for item in items}) != len(items):
                raise ValueError(f"Duplicate {label} names")
        endpoint_pairs = [(edge.cause.id, edge.effect.id) for edge in self.edges]
        if len(endpoint_pairs) != len(set(endpoint_pairs)):
            raise ValueError("Declare one causal edge per endpoint pair and compose its mechanisms")
        for edge in self.edges:
            if error := _check_edge_constraint(edge):
                raise ValueError(error)
        if errors := _check_global_constraints(self.constructs, self.edges):
            raise ValueError(errors[0])
        for construct in self.constructs:
            if construct.temporal_status == TemporalStatus.TIME_INVARIANT and construct.dynamics:
                raise ValueError("Time-invariant constructs cannot have intrinsic drift")
        for owner, mechanism in self.iter_mechanisms():
            dependencies = expression_states(mechanism.expression)
            if unknown := dependencies - self._constructs.keys():
                raise ValueError(f"Expression references unknown constructs: {sorted(unknown)}")
            if isinstance(owner, CausalEdgeSpec):
                if owner.cause.id not in dependencies:
                    raise ValueError("An edge expression must reference its causal source")
                allowed = {owner.effect.id} | {
                    edge.cause.id for edge in self.edges if edge.effect.id == owner.effect.id
                }
                if dependencies - allowed:
                    raise ValueError(
                        "Expression dependencies require explicit causal edges to their effect"
                    )
            elif dependencies - {owner.id}:
                raise ValueError("Intrinsic dynamics may reference only their owning construct")
        for indicator, likelihood in self.iter_likelihoods():
            owners = set(likelihood.parsed.loadings)
            unknown = owners - self._constructs.keys()
            if unknown:
                raise ValueError(f"Likelihood references unknown constructs: {sorted(unknown)}")
            if self.indicator_owner(indicator.observation.id).id not in owners:
                raise ValueError(
                    f"Likelihood {indicator.observation.id!r} must include its measured construct"
                )

        from nof1_causal_lab.artifacts.model.parameter_context import iter_coefficient_uses

        for construct in self.constructs:
            unknown = {
                identity for operand in construct.coefficients for identity in operand.construct_ids
            } - self._constructs.keys()
            if unknown:
                raise ValueError(
                    f"Construct coefficients reference unknown constructs: {sorted(unknown)}"
                )
            if any(
                operand.role in {"diffusion_loading", "initial_correlation"}
                and any(
                    self.get_construct(identity).role == Role.EXOGENOUS
                    for identity in operand.construct_ids
                )
                for operand in construct.coefficients
            ):
                raise ValueError("Noise and initial correlations cannot reference exogenous inputs")
        for use in iter_coefficient_uses(self):
            if isinstance(use.value, str) and use.value not in self._parameters:
                raise ValueError(f"Coefficient {use.slot!r} references an undeclared parameter")
        if unused := self._parameters.keys() - self._parameter_contexts.keys():
            raise ValueError(f"Parameters are not referenced by component slots: {sorted(unused)}")
        for identity, context in self._parameter_contexts.items():
            quantity = context.quantity
            parameter = self.parameter(identity)
            if (
                parameter.transform.kind == "dt_persistence_to_ct_decay"
                and quantity.value != "dynamics_decay"
            ):
                raise ValueError("Persistence coordinates describe a dynamics decay quantity")
        if self.distributions or self.law_layouts:
            from nof1_causal_lab.artifacts.probability.membership import (
                validate_distribution_memberships,
            )

            validate_distribution_memberships(self)
        from nof1_causal_lab.artifacts.model.coefficient_redundancy import (
            validate_coefficient_redundancy,
        )

        validate_coefficient_redundancy(self)
        return self

    def require_measurements(self) -> None:
        """Require both a measurement clock and indicators before preparing or binding observations."""
        if self.measurement_clock is None or not self.indicators:
            raise IncompleteModelError("Measurements require a measurement clock and indicators")


class DynamicalModelSpec(Value):
    """One identity-addressed scientific document, usable for creation and partial edits.

    Omission carries no update. Null entity entries are deletion instructions. The
    editing boundary materializes and checks the complete document before publication.
    Scientific consumers resolve its owned entities at that boundary.
    """

    model_config = ConfigDict(serialize_by_alias=True, validate_by_name=True)

    construct_definitions: Mapping[ConstructId, JsonObject | None] = Field(
        default_factory=dict, alias="constructs"
    )
    edge_definitions: Mapping[EdgeId, JsonObject | None] = Field(
        default_factory=dict, alias="edges"
    )
    parameter_definitions: Mapping[ParameterId, JsonObject | None] = Field(
        default_factory=dict, alias="parameters"
    )
    distribution_definitions: Mapping[
        DistributionId, NumPyroObject | NumPyroDistribution | None
    ] = Field(default_factory=dict, alias="distributions")
    layout_definitions: Mapping[DistributionId, JsonObject | None] = Field(
        default_factory=dict, alias="law_layouts"
    )
    measurement_clock: Duration | None = None

    @field_serializer("distribution_definitions", mode="wrap")
    def serialize_laws(  # noqa: ANN201 -- The scientific owner supplies the native constructor schema.
        self,
        value: Mapping[DistributionId, NumPyroObject | NumPyroDistribution | None],
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ):
        """Serialize partial constructor documents through the same numerical boundary as native laws."""
        context = info.context or {}
        return handler(
            {
                identity: serialize_constructor(
                    law,
                    binary=bool(context.get("binary_arrays")),
                    array_loader=context.get("distribution_array_loader"),
                )
                if isinstance(law, Mapping)
                else law
                for identity, law in value.items()
            }
        )

    @model_serializer(mode="wrap")  # noqa: V105 -- Pydantic invokes the registered serialization hook.
    def serialize_supplied(  # noqa: ANN201 -- Preserve the owner-derived document schema instead of replacing it with the serializer mapping type.
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ):
        """Preserve omission across HTTP, call identity, Temporal, and saved requests."""
        fields = {
            name if info.by_alias is False else type(self).model_fields[name].alias or name
            for name in self.model_fields_set
        }
        serialized = handler(self)
        return {key: value for key, value in serialized.items() if key in fields}

    @classmethod
    @override
    def __get_pydantic_json_schema__(
        cls, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Publish the same document fields for new models, edits, and complete saved models."""
        result = handler.resolve_ref_schema(handler(schema))
        resolved = handler.resolve_ref_schema(handler(_ModelEntities.__pydantic_core_schema__))
        properties = document_schema(resolved, handler)["properties"]
        properties.pop("time_points", None)
        edge_schema = properties["edges"]["additionalProperties"]
        if handler.mode == "validation":
            edge_schema = edge_schema["anyOf"][0]
        for name in ("cause", "effect"):
            edge_schema["properties"][name] = properties["constructs"]["propertyNames"]
        result["properties"] = properties
        if handler.mode == "serialization":
            result["required"] = list(properties)
        else:
            result.pop("required", None)
        return result

    @cached_property
    def _entities(self) -> _ModelEntities:
        """Resolve the complete document once through its scientific owners."""
        return self._resolve()

    def _resolve(self, context: object = None) -> _ModelEntities:
        """Parse complete fields, retaining the storage boundary's lazy numerical laws."""
        return _ModelEntities.model_validate(self, context=context)

    @override
    def __eq__(self, other: object) -> bool:
        """Compare supplied scientific fields independently of resolved caches."""
        return isinstance(other, DynamicalModelSpec) and bool(
            self.model_dump(mode="json") == other.model_dump(mode="json")
        )

    @classmethod
    def from_entities(
        cls,
        *,
        edges: tuple[CausalEdgeSpec, ...] = (),
        constructs: tuple[ConstructSpec, ...] | None = None,
        parameters: tuple[ParameterSpec, ...] = (),
        distributions: Mapping[DistributionId, NumPyroDistribution] | None = None,
        law_layouts: Mapping[DistributionId, JointLawLayout] | None = None,
        measurement_clock: Duration | str | None = None,
    ) -> DynamicalModelSpec:
        """Build a complete document from resolved scientific values at their owner."""
        values = _ModelEntities.model_validate(
            {
                "constructs": constructs
                if constructs is not None
                else tuple(
                    {
                        endpoint.id: endpoint
                        for edge in edges
                        for endpoint in (edge.cause, edge.effect)
                    }.values()
                ),
                "edges": edges,
                "parameters": parameters,
                "distributions": distributions or {},
                "law_layouts": law_layouts or {},
                "measurement_clock": measurement_clock,
            }
        )
        return cls._from_entities(values)

    @classmethod
    def _from_entities(cls, values: _ModelEntities) -> DynamicalModelSpec:
        """Serialize entity definitions while retaining native laws and their array owners."""
        serialized = values.model_dump(mode="json")
        return cls(
            constructs={
                item.id: entity_document(item.model_dump(mode="json")) for item in values.constructs
            },
            edges={
                item.id: {
                    **entity_document(item.model_dump(mode="json", exclude={"cause", "effect"})),
                    "cause": item.cause.id,
                    "effect": item.effect.id,
                }
                for item in values.edges
            },
            parameters={
                item.id: entity_document(item.model_dump(mode="json")) for item in values.parameters
            },
            distributions=values.distributions,
            law_layouts=serialized["law_layouts"],
            measurement_clock=values.measurement_clock,
        )

    def with_entities(
        self,
        *,
        edges: tuple[CausalEdgeSpec, ...] | None = None,
        constructs: tuple[ConstructSpec, ...] | None = None,
        parameters: tuple[ParameterSpec, ...] | None = None,
        distributions: Mapping[DistributionId, NumPyroDistribution] | None = None,
        law_layouts: Mapping[DistributionId, JointLawLayout] | None = None,
    ) -> DynamicalModelSpec:
        """Rebuild a complete document after a scientific transformation."""
        selected_edges = self.edges if edges is None else edges
        return type(self).from_entities(
            constructs=constructs
            if constructs is not None
            else tuple(
                {
                    endpoint.id: endpoint
                    for edge in selected_edges
                    for endpoint in (edge.cause, edge.effect)
                }.values()
            )
            if edges is not None
            else self.constructs,
            edges=selected_edges,
            parameters=self.parameters if parameters is None else parameters,
            distributions=self.distributions if distributions is None else distributions,
            law_layouts=self.law_layouts if law_layouts is None else law_layouts,
            measurement_clock=self.measurement_clock,
        )

    @property
    def constructs(self) -> tuple[ConstructSpec, ...]:
        """Read the explicit construct definitions in authored order."""
        return self._entities.constructs

    @property
    def edges(self) -> tuple[CausalEdgeSpec, ...]:
        """Read the resolved edges."""
        return self._entities.edges

    @property
    def parameters(self) -> tuple[ParameterSpec, ...]:
        """Read the resolved parameters."""
        return self._entities.parameters

    @property
    def distributions(self) -> Mapping[DistributionId, NumPyroDistribution]:
        """Read the resolved distributions."""
        return self._entities.distributions

    @property
    def law_layouts(self) -> Mapping[DistributionId, JointLawLayout]:
        """Read the resolved law layouts."""
        return self._entities.law_layouts

    @property
    def time_points(self) -> tuple[FiniteFloat, ...]:
        """Joint trajectory coordinates own the model's retained time grid."""
        return self._entities.time_points

    @property
    def _constructs(self) -> Mapping[ConstructId, ConstructSpec]:
        """Read the resolved constructs."""
        return self._entities._constructs

    @property
    def indicators(self) -> tuple[IndicatorSpec, ...]:
        """All observation indicators in construct order and each construct's authored order."""
        return self._entities.indicators

    def parameter_context(self, identity: ParameterId) -> ParameterContext:
        """Look up a parameter's quantity context and scientific owners by its model-local identity."""
        return self._entities.parameter_context(identity)

    def get_construct(self, identity: ConstructId) -> ConstructSpec:
        """Look up the construct owned by this model; an unknown identity raises ``KeyError``."""
        return self._entities.get_construct(identity)

    def distribution_for(self, identity: ParameterId | ConstructId) -> NumPyroDistribution | None:
        """Resolve the law a quantity participates in, retaining its full joint dependence."""
        return self._entities.distribution_for(identity)

    def edge(self, identity: EdgeId) -> CausalEdgeSpec:
        """Look up the causal edge owned by this model; an unknown identity raises ``KeyError``."""
        return self._entities.edge(identity)

    def indicator(self, identity: IndicatorId) -> IndicatorSpec:
        """Look up the measurement indicator by observation ID; unknown IDs raise ``KeyError``."""
        return self._entities.indicator(identity)

    def indicator_owner(self, identity: IndicatorId) -> ConstructSpec:
        """Look up the construct that owns the named observation indicator."""
        return self._entities.indicator_owner(identity)

    def parameter(self, identity: ParameterId) -> ParameterSpec:
        """Look up the parameter definition by its scientific identity."""
        return self._entities.parameter(identity)

    def mechanism(self, identity: MechanismId) -> DynamicsMechanismSpec:
        """Look up the dynamics mechanism owned by this model using its mechanism identity."""
        return self._entities.mechanism(identity)

    def iter_indicators(self) -> Iterator[tuple[ConstructSpec, IndicatorSpec]]:
        """Yield each indicator paired with its owning construct in authored order."""
        return self._entities.iter_indicators()

    def iter_likelihoods(self) -> Iterator[tuple[IndicatorSpec, LikelihoodSpec]]:
        """Yield indicators with declared likelihoods, paired with their likelihood definitions."""
        return self._entities.iter_likelihoods()

    def iter_mechanisms(
        self,
    ) -> Iterator[tuple[ConstructSpec | CausalEdgeSpec, DynamicsMechanismSpec]]:
        """Yield construct dynamics followed by edge mechanisms, each paired with its owner."""
        return self._entities.iter_mechanisms()

    def parameters_for(
        self, identity: ConstructId | EdgeId | IndicatorId | MechanismId
    ) -> tuple[ParameterSpec, ...]:
        """Select parameters whose recorded ownership includes the named scientific entity."""
        return self._entities.parameters_for(identity)

    def require_measurements(self) -> None:
        """Require both a measurement clock and indicators before preparing or binding observations."""
        self._entities.require_measurements()

    def materialized(self, *, context: object = None) -> DynamicalModelSpec:
        """Parse and validate the assembled document, filling scientific defaults once."""
        return type(self)._from_entities(self._resolve(context))

    def changes_from(self, before: DynamicalModelSpec) -> DynamicalModelSpec:
        """Diff saved materialized specs using model_document.diff_fields semantics.

        The returned partial DynamicalModelSpec uses the existing edit language and
        reconstructs this spec when merged into before. ModelDiffOutput supplies
        the action boundary; this operation does not compile or evaluate findings.
        """
        return self.model_validate(
            diff_fields(before.model_dump(mode="json"), self.model_dump(mode="json"))
        )


class ModelEditResult(Value):
    """Scientific entities pruned while materializing an accepted model edit."""

    constructs: tuple[ConstructId, ...] = ()
    edges: tuple[EdgeId, ...] = ()
    parameters: tuple[ParameterId, ...] = ()
    distributions: tuple[DistributionId, ...] = ()


class MaterializedModel(Value):
    """The complete model and the pruning evidence owned by its editing boundary."""

    dynamical_model_spec: DynamicalModelSpec
    pruning: ModelEditResult


class _PruningGraph(Value):
    """Complete references required to scope an edit before parsing its scientific definitions."""

    edges: Mapping[EdgeId, tuple[ConstructId, ConstructId]]
    memberships: Mapping[ConstructId | ParameterId, DistributionId]


def _pruning_graph(document: DynamicalModelSpec) -> _PruningGraph:
    return _PruningGraph.model_validate(
        {
            "edges": {
                identity: (value.get("cause"), value.get("effect"))
                for identity, value in document.edge_definitions.items()
                if value is not None
            },
            "memberships": {
                identity: value["distribution"]
                for definitions in (document.construct_definitions, document.parameter_definitions)
                for identity, value in definitions.items()
                if value is not None and value.get("distribution") is not None
            },
        }
    )


def apply_model_edit(
    parent: DynamicalModelSpec,
    supplied: DynamicalModelSpec,
    outcome: ConstructId,
    *,
    context: object = None,
) -> MaterializedModel:
    """Merge a document, prune outside the outcome's ancestors, and parse the result."""
    document = DynamicalModelSpec.model_validate(
        merge_fields(parent.model_dump(mode="json"), supplied.model_dump(mode="json"))
    )
    graph = _pruning_graph(document)
    edges = graph.edges
    retained = frozenset({outcome})
    while (
        expanded := retained | {cause for cause, effect in edges.values() if effect in retained}
    ) != retained:
        retained = expanded
    constructs = {
        identity: value
        for identity, value in document.construct_definitions.items()
        if identity in retained
    }
    retained_edges = {
        identity: document.edge_definitions[identity]
        for identity, (cause, effect) in edges.items()
        if cause in retained and effect in retained
    }
    previous_uses = parameter_references(
        tuple(parent.construct_definitions.values())
    ) | parameter_references(tuple(parent.edge_definitions.values()))
    merged_uses = parameter_references(
        tuple(document.construct_definitions.values())
    ) | parameter_references(tuple(document.edge_definitions.values()))
    retained_uses = parameter_references(tuple(constructs.values())) | parameter_references(
        tuple(retained_edges.values())
    )
    removed_parameters = (previous_uses | merged_uses) - retained_uses
    parameters = {
        identity: value
        for identity, value in document.parameter_definitions.items()
        if identity not in removed_parameters
    }
    prior_memberships = frozenset(_pruning_graph(parent).memberships.values()) | frozenset(
        graph.memberships.values()
    )
    kept_memberships = {
        distribution
        for identity, distribution in graph.memberships.items()
        if identity in constructs or identity in parameters
    }
    removed_distributions = prior_memberships - kept_memberships
    complete = DynamicalModelSpec(
        constructs=constructs,
        edges=retained_edges,
        parameters=parameters,
        distributions={
            identity: value
            for identity, value in document.distribution_definitions.items()
            if identity not in removed_distributions
        },
        law_layouts={
            identity: value
            for identity, value in document.layout_definitions.items()
            if identity not in removed_distributions
        },
        measurement_clock=document.measurement_clock,
    ).materialized(context=context)
    return MaterializedModel(
        dynamical_model_spec=complete,
        pruning=ModelEditResult(
            constructs=tuple(
                identity
                for identity in document.construct_definitions
                if identity not in constructs
            ),
            edges=tuple(
                identity for identity in document.edge_definitions if identity not in retained_edges
            ),
            parameters=tuple(
                identity
                for identity in document.parameter_definitions
                if identity not in parameters
            ),
            distributions=tuple(
                identity
                for identity in document.distribution_definitions
                if identity in removed_distributions
            ),
        ),
    )
