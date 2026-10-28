"""Derive execution selections and projection findings from canonical model entities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import TYPE_CHECKING, Literal

from grassmoth.artifacts.model.construct import CausalEdgeSpec, TemporalStatus
from grassmoth.compilation_errors import AggregatedCompileError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from grassmoth.artifacts.identity import ConstructId, IndicatorId
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth.artifacts.model.indicator import IndicatorSpec
    from grassmoth.artifacts.model.question import QuestionSpec
    from grassmoth.artifacts.observations.duration import Duration
    from grassmoth.artifacts.reports.identification import IdentificationReport

type DependencyKind = Literal["innovation_correlation", "initial_state_correlation"]
type DependencyKey = tuple[ConstructId, ConstructId, DependencyKind]


class StructuralCompilationError(AggregatedCompileError):
    """Unsupported scientific structure encountered when preparing execution."""

    header = "Structural compilation failed"


class StructuralSelectionError(ValueError):
    """The model breaks a check its outcome scope decides: parameter anchors."""


@dataclass(frozen=True, eq=False)
class StructuralSelection:
    """One model's executable structure, scoped by the question's outcome.

    The outcome keeps its connected component as states; without one, every
    measured construct is a state. The rest of the DAG stays for identification.
    Construction checks what the scope decides: the retained constructs' anchors
    only.
    """

    dynamical_model_spec: DynamicalModelSpec
    outcome: ConstructId | None

    def __post_init__(self) -> None:
        """Require a model-owned outcome and valid parameter anchors for the structural selection."""
        if self.outcome is not None and self.outcome not in self.dynamical_model_spec._constructs:
            raise ValueError("A selection's outcome must be a construct of its model")
        validate_parameter_anchors(self)

    @classmethod
    def for_question(
        cls, dynamical_model_spec: DynamicalModelSpec, question: QuestionSpec
    ) -> StructuralSelection:
        """The question's outcome scopes the model once the model defines it."""
        defined = question.outcome in dynamical_model_spec._constructs
        return cls(dynamical_model_spec, question.outcome if defined else None)

    @cached_property
    def identification(self) -> IdentificationReport:
        """Causal identification findings for this model and selected outcome."""
        from grassmoth.models.causal.analysis import identify_model

        return identify_model(self)

    @cached_property
    def marginalized_construct_ids(self) -> frozenset[ConstructId]:
        """Construct identities removed from the execution state by marginalization."""
        return marginalized_construct_ids(self)

    @cached_property
    def retained_construct_ids(self) -> frozenset[ConstructId]:
        """Construct identities retained in the execution representation."""
        return retained_construct_ids(self)

    @cached_property
    def induced_dependencies(self) -> Mapping[DependencyKey, tuple[ConstructId, ...]]:
        """Dependencies induced by marginalization, paired with the constructs that induce them."""
        return induced_dependencies(self)


def marginalized_construct_ids(selection: StructuralSelection) -> frozenset[ConstructId]:
    """Identify eligible unmeasured roots without changing the scientific DAG."""
    dynamical_model_spec = selection.dynamical_model_spec
    observed = {
        construct.id
        for construct in dynamical_model_spec.constructs
        if construct.indicators or construct.role == "exogenous"
    }
    blocked = {
        confounder
        for treatment, finding in selection.identification.non_identifiable.items()
        if treatment in observed
        for confounder in finding.confounders
    }
    children = {edge.effect.id for edge in dynamical_model_spec.edges}
    return frozenset(
        construct.id
        for construct in dynamical_model_spec.constructs
        if construct.id not in observed | blocked and construct.id not in children
    )


def selected_state_ids(selection: StructuralSelection) -> tuple[ConstructId, ...]:
    """A structural selection is meaningful even while numerical choices are drafts."""
    selected = selection.retained_construct_ids
    return tuple(
        item.id
        for static in (False, True)
        for item in selection.dynamical_model_spec.constructs
        if item.id in selected and (item.temporal_status == TemporalStatus.TIME_INVARIANT) == static
    )


def selected_edges(selection: StructuralSelection) -> tuple[CausalEdgeSpec, ...]:
    """Select authored edges whose cause and effect both belong to the retained state set."""
    states = set(selected_state_ids(selection))
    return tuple(
        edge
        for edge in selection.dynamical_model_spec.edges
        if edge.cause.id in states and edge.effect.id in states
    )


def selected_indicators(selection: StructuralSelection) -> tuple[IndicatorSpec, ...]:
    """Select indicators owned by retained states, preserving authored model order."""
    states = set(selected_state_ids(selection))
    return tuple(
        indicator
        for owner, indicator in selection.dynamical_model_spec.iter_indicators()
        if owner.id in states
    )


def reference_indicators(selection: StructuralSelection) -> Mapping[ConstructId, IndicatorId]:
    """Choose the measurement anchor for each selected state and return its observation ID."""
    from types import MappingProxyType

    from grassmoth.models.causal.graphs import choose_reference_indicator

    return MappingProxyType(
        {
            identity: choose_reference_indicator(
                selection.dynamical_model_spec.get_construct(identity).indicators
            ).observation.id
            for identity in selected_state_ids(selection)
            if selection.dynamical_model_spec.get_construct(identity).indicators
        }
    )


def induced_dependencies(
    selection: StructuralSelection,
) -> dict[DependencyKey, tuple[ConstructId, ...]]:
    """Pairs of retained states sharing projected roots, with their scientific sources."""
    from grassmoth.models.causal.identification import dag_to_admg, get_observed_constructs

    dynamical_model_spec = selection.dynamical_model_spec
    observed = get_observed_constructs(dynamical_model_spec.constructs)
    _, confounders = dag_to_admg(
        dynamical_model_spec.constructs, dynamical_model_spec.edges, observed
    )
    retained = set(selected_state_ids(selection))
    sources: dict[DependencyKey, list[ConstructId]] = defaultdict(list)
    for identity in sorted(
        selection.marginalized_construct_ids,
        key=lambda cid: dynamical_model_spec.get_construct(cid).name,
    ):
        construct = dynamical_model_spec.get_construct(identity)
        if construct.name not in confounders:
            continue
        children = sorted(
            {
                edge.effect.id
                for edge in dynamical_model_spec.edges
                if edge.cause.id == identity and edge.effect.id in retained
            },
            key=lambda cid: dynamical_model_spec.get_construct(cid).name,
        )
        kind: DependencyKind = (
            "initial_state_correlation"
            if construct.temporal_status == TemporalStatus.TIME_INVARIANT
            else "innovation_correlation"
        )
        for first, second in combinations(children, 2):
            sources[first, second, kind].append(identity)
    return {
        key: tuple(sorted(identities, key=lambda cid: dynamical_model_spec.get_construct(cid).name))
        for key, identities in sorted(
            sources.items(),
            key=lambda item: (
                dynamical_model_spec.get_construct(item[0][0]).name,
                dynamical_model_spec.get_construct(item[0][1]).name,
                item[0][2],
            ),
        )
    }


def retained_construct_ids(selection: StructuralSelection) -> frozenset[ConstructId]:
    """Keep the outcome's component after projection, including statistical dependencies."""
    import networkx as nx

    dynamical_model_spec = selection.dynamical_model_spec
    measured = {
        construct.id
        for construct in dynamical_model_spec.constructs
        if construct.indicators or construct.role == "exogenous"
    }
    if selection.outcome is None:
        # A model-wide operation has no outcome against which to discard a component.
        return frozenset(measured)

    graph = nx.Graph()
    graph.add_nodes_from((*measured, selection.outcome))
    marginalized = selection.marginalized_construct_ids
    graph.add_edges_from(
        (edge.cause.id, edge.effect.id)
        for edge in dynamical_model_spec.edges
        if edge.effect.id in measured
        and (
            edge.cause.id in marginalized
            or (
                edge.cause.id in measured
                and edge.effect.temporal_status != TemporalStatus.TIME_INVARIANT
            )
        )
    )
    for construct in dynamical_model_spec.constructs:
        if construct.id not in measured:
            continue
        dependencies = {
            identity for operand in construct.coefficients for identity in operand.construct_ids
        }
        for indicator in construct.indicators:
            if indicator.likelihood is not None:
                dependencies.update(indicator.likelihood.parsed.loadings)
        graph.add_edges_from((construct.id, identity) for identity in dependencies & measured)

    # Shared parameters and joint laws can connect components without a direct causal edge.
    law_members: dict[str, set[ConstructId]] = defaultdict(set)
    for construct in dynamical_model_spec.constructs:
        if construct.distribution is not None and construct.id in measured:
            law_members[construct.distribution].add(construct.id)
    for parameter in dynamical_model_spec.parameters:
        owners = {
            owner.id
            for owner in dynamical_model_spec.parameter_context(parameter.id).owners
            if owner.kind == "construct" and owner.id in measured
        }
        nx.add_path(graph, sorted(owners))
        if parameter.distribution is not None:
            law_members[parameter.distribution].update(owners)
    for members in law_members.values():
        nx.add_path(graph, sorted(members))
    component = nx.node_connected_component(graph, selection.outcome)
    return frozenset(identity for identity in measured if identity in component)


def unsupported_construct_ids(selection: StructuralSelection) -> frozenset[ConstructId]:
    """Required parents outside the selection without an executable marginalization rule."""
    states = set(selected_state_ids(selection))
    selected = selection.retained_construct_ids
    return frozenset(
        {
            edge.cause.id
            for edge in selection.dynamical_model_spec.edges
            if edge.effect.id in states
            and edge.cause.id not in states
            and (not edge.cause.indicators or edge.cause.id not in selected)
            and edge.cause.id not in selection.marginalized_construct_ids
        }
    )


def validate_execution_structure(selection: StructuralSelection) -> None:
    """Check executable capabilities after the model's intrinsic/reference validation."""
    selection.dynamical_model_spec.require_measurements()
    states = set(selected_state_ids(selection))
    errors = []
    if selection.outcome is not None and selection.outcome not in states:
        errors.append("The question's outcome requires retained measurement indicators.")
    for edge in selected_edges(selection):
        if edge.effect.temporal_status == TemporalStatus.TIME_INVARIANT:
            errors.append(
                f"Unsupported retained static-target edge {edge.id} "
                f"({edge.cause.name!r} -> {edge.effect.name!r}). "
                "The executable SSM has no baseline structural-equation semantics. "
                "Supply supported baseline structural semantics before executing this relation."
            )
    unsupported = unsupported_construct_ids(selection)
    if unsupported:
        errors.append(
            "Required constructs outside the retained states cannot be projected: "
            f"{sorted(unsupported)}. Supply measurements or supported marginalization semantics."
        )
    if errors:
        raise StructuralCompilationError(errors)


def validate_parameter_anchors(selection: StructuralSelection) -> None:
    """Reject unanchored retained constructs once their anchor inputs are authored."""
    from grassmoth.artifacts.model.construct import CausalEdgeSpec
    from grassmoth.artifacts.model.expressions import StateExpression, restoring_coefficients
    from grassmoth.artifacts.probability.families import DistributionFamily

    dynamical_model_spec = selection.dynamical_model_spec
    if dynamical_model_spec.measurement_clock is None or not dynamical_model_spec.indicators:
        return

    for identity in selected_state_ids(selection):
        construct = dynamical_model_spec.get_construct(identity)
        initial_mean = construct.coefficient("initial_mean")
        channels = [
            (indicator, indicator.likelihood)
            for indicator in construct.indicators
            if indicator.likelihood is not None
        ]
        if initial_mean is None or len(channels) != len(construct.indicators):
            continue
        if any(
            operand.value is None
            for _, likelihood in channels
            for operand in likelihood.parsed.loadings.values()
        ):
            continue
        centers = [
            operand.value
            for owner, mechanism in dynamical_model_spec.iter_mechanisms()
            if (owner.effect.id if isinstance(owner, CausalEdgeSpec) else owner.id) == identity
            for operand in restoring_coefficients(
                mechanism.expression, identity, kind=mechanism.kind
            )
            if operand.role == "center"
        ]
        if any(center is None for center in centers):
            continue

        channel_location = any(
            likelihood.standardized
            or (
                likelihood.law.distribution == "Delta"
                and isinstance(likelihood.law.v, StateExpression)
            )
            for indicator, likelihood in channels
        )
        fixed_location = (
            not isinstance(initial_mean, str)
            if construct.temporal_status == "time_invariant"
            else any(not isinstance(center, str) for center in centers)
        )
        if not channel_location and not fixed_location:
            raise StructuralSelectionError(f"Construct {construct.name!r} has no location anchor")

        fixed_scale = False
        all_categorical = True
        for indicator, likelihood in channels:
            categorical = likelihood.law.family == DistributionFamily.CATEGORICAL
            loading = likelihood.parsed.loadings[identity].value
            value = None if isinstance(loading, str) else loading
            if categorical and value is None:
                raise StructuralSelectionError(
                    f"Construct {construct.name!r} has a free categorical loading on "
                    f"indicator {indicator.observation.name!r}"
                )
            all_categorical &= categorical
            fixed_scale |= not categorical and value is not None and value != 0.0
        if not fixed_scale and not all_categorical:
            reference = dynamical_model_spec.indicator(reference_indicators(selection)[identity])
            raise StructuralSelectionError(
                f"Construct {construct.name!r} has no scale anchor "
                f"(reference indicator {reference.observation.name!r})"
            )


def get_reference_indicator_lookup(selection: StructuralSelection) -> dict[str, str]:
    """Return retained construct name to its planned reference indicator name."""
    dynamical_model_spec = selection.dynamical_model_spec
    return {
        dynamical_model_spec._constructs[construct_id].name: (
            dynamical_model_spec.indicator(indicator_id).observation.name
        )
        for construct_id, indicator_id in reference_indicators(selection).items()
    }


def get_model_clock(dynamical_model_spec: DynamicalModelSpec) -> Duration:
    """Require a measurement-ready model and return its authored clock duration."""
    dynamical_model_spec.require_measurements()
    assert dynamical_model_spec.measurement_clock is not None
    return dynamical_model_spec.measurement_clock
