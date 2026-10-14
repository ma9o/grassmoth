"""Public pure-compilation entry points for executable SSM inputs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import ConstructId
from nof1_causal_lab.compilation_errors import AggregatedCompileError, IncompleteModelError
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.model_parameters import execution_parameters
from nof1_causal_lab.models.model_structure import selected_edges
from nof1_causal_lab.models.ssm.compile.prior_compilation import (
    bind_parameters,
    compile_priors,
)
from nof1_causal_lab.models.ssm.compile.prior_indexing import (
    build_site_bindings,
)
from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
from nof1_causal_lab.models.ssm.parameterization import (
    PriorRuntimeBundle,
    build_prior_runtime_bundle,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import jax
    import numpyro.distributions as dist

    from nof1_causal_lab.artifacts.identity import (
        EdgeId,
        IndicatorId,
    )
    from nof1_causal_lab.artifacts.likelihood import Law
    from nof1_causal_lab.artifacts.observations import ObservationWindow, ResolvedObservationSpec
    from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
    from nof1_causal_lab.artifacts.prior import PriorValidationResult
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.models.ssm.compile.bindings import CompiledParameterBinding
    from nof1_causal_lab.models.ssm.compile.state_space import StateSpace
    from nof1_causal_lab.models.ssm.dynamics.expression import BoundExpression
    from nof1_causal_lab.models.ssm.dynamics.spec import CompiledDynamics
    from nof1_causal_lab.models.ssm.structure import (
        DiffusionBlockSpec,
        ManifestCholBlockSpec,
        SparseBlockSpec,
        T0CholBlockSpec,
    )
    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor
    from nof1_causal_lab.utils.observation_semantics import IndicatorObservationSemantics


@dataclass(frozen=True)
class CompiledState:
    """One state coordinate and its execution semantics."""

    id: ConstructId
    name: str
    is_input: bool
    time_invariant: bool
    innovation_family: DistributionFamily
    incoming_edges: tuple[tuple[int, EdgeId], ...]


@dataclass(frozen=True)
class CompiledObservation:
    """One emission coordinate, with its state binding and support semantics."""

    observation: ResolvedObservationSpec
    state_index: int
    law: Law[BoundExpression]
    standardized: bool
    categorical_anchor: bool

    @property
    def id(self) -> IndicatorId:
        """Stable observation identity retained from the authored measurement definition."""
        return self.observation.id

    @property
    def name(self) -> str:
        """Human-readable observation label retained from the measurement definition."""
        return self.observation.name

    @property
    def levels(self) -> tuple[str, ...]:
        """Declared levels for categorical or ordinal likelihoods; empty for other families."""
        return (
            self.observation.definition.levels
            if self.law.family
            in {DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL}
            else ()
        )

    @property
    def support(self) -> IndicatorObservationSemantics:
        """Measurement support semantics derived from the resolved observation definition."""
        return self.observation._observation_semantics()

    @property
    def observation_window(self) -> ObservationWindow:
        """Resolved fixed or calendar window summarized by this indicator."""
        return self.observation.observation_window


@dataclass(frozen=True)
class CompiledLaw:
    """One native law and the scientific coordinates sampled together from it."""

    sample_index: int
    distribution: dist.Distribution
    layout: JointLawLayout


class CompiledInputTrajectory(Value):
    """One deterministic construct trajectory on its model-owned coordinates."""

    index: int
    construct_id: ConstructId
    layout: JointLawLayout
    values: tuple[float, ...]


@dataclass(frozen=True, eq=False)
class CompiledDynamicalModel:
    """Ordered execution records; authored entities remain at the compiler boundary."""

    clock_days: float
    states: tuple[CompiledState, ...]
    observations: tuple[CompiledObservation, ...]
    static_factors: tuple[CompiledState, ...]
    diffusion_block: DiffusionBlockSpec
    loading_block: SparseBlockSpec[tuple[int, int]]
    observation_mean_block: SparseBlockSpec[int]
    observation_noise_block: ManifestCholBlockSpec
    initial_mean_block: SparseBlockSpec[int]
    initial_covariance_block: T0CholBlockSpec
    static_scale_block: SparseBlockSpec[int]
    static_factor_loadings: jax.Array
    dynamics: CompiledDynamics
    site_registry: tuple[SiteDescriptor, ...]
    bindings: tuple[CompiledParameterBinding, ...]
    auxiliary_coordinates: tuple[ParameterCoordinate, ...]
    laws: tuple[CompiledLaw, ...]
    input_trajectories: tuple[CompiledInputTrajectory, ...]
    state_space: StateSpace

    @property
    def state_index(self) -> Mapping[ConstructId, int]:
        """Construct IDs mapped to their positions on the compiled state axis."""
        return MappingProxyType({state.id: index for index, state in enumerate(self.states)})

    @property
    def indicator_index(self) -> Mapping[IndicatorId, int]:
        """Observation IDs mapped to their positions on the compiled indicator axis."""
        return MappingProxyType(
            {observation.id: index for index, observation in enumerate(self.observations)}
        )


@dataclass(frozen=True)
class CompiledFitInputs:
    """Fit-specific prior capability for one compiled numerical model."""

    compiled_dynamical_model: CompiledDynamicalModel
    prior_runtime_bundle: PriorRuntimeBundle
    diagnostics: tuple[PriorValidationResult, ...]


@dataclass(frozen=True)
class IncompleteModel:
    """The authored model still needs choices before fitting."""

    message: str


@dataclass(frozen=True)
class UnsupportedFit:
    """The current fitting engine cannot compile these scientific choices."""

    errors: tuple[str, ...]

    @property
    def message(self) -> str:
        """Compilation findings joined into a readable explanation of why fitting is unsupported."""
        return "\n".join(self.errors)


type CompilationFailure = IncompleteModel | UnsupportedFit
type ModelCompilationResult = CompiledDynamicalModel | CompilationFailure
type FitCompilationResult = CompiledFitInputs | CompilationFailure


def _attach_compile_binding_provenance(
    diagnostics: list[PriorValidationResult],
    bindings: tuple[CompiledParameterBinding, ...],
) -> list[PriorValidationResult]:
    """Attach direct-writer parameter provenance to compile diagnostics when possible."""
    binding_index: dict[tuple[str, int], list[str]] = {}
    for binding in bindings:
        binding_index.setdefault((binding.site.name, binding.flat_index), []).append(
            binding.parameter_id
        )

    resolved: list[PriorValidationResult] = []
    for diagnostic in diagnostics:
        if diagnostic.compiled_site_name is None or diagnostic.compiled_flat_index is None:
            resolved.append(diagnostic)
            continue
        related_parameters = binding_index.get(
            (diagnostic.compiled_site_name, diagnostic.compiled_flat_index)
        )
        if related_parameters:
            diagnostic = diagnostic.with_parameter_provenance(tuple(related_parameters))
        resolved.append(diagnostic)

    return resolved


def compile_model(
    selection: StructuralSelection,
) -> ModelCompilationResult:
    """Compile shared execution facts without imposing fitting's prior-engine limits."""
    from nof1_causal_lab.models.model_parameters import require_priors
    from nof1_causal_lab.models.ssm.compile import support as numeric
    from nof1_causal_lab.models.ssm.compile.state_space import compile_state_space
    from nof1_causal_lab.models.ssm.dynamics.spec import compile_dynamics

    dynamical_model_spec = selection.dynamical_model_spec
    try:
        numeric._require_execution_choices(selection)
        require_priors(selection)
        state_ids = tuple(numeric.state_ids(selection))
        indicators = numeric.observed_indicators(selection)
        blocks = numeric.parameter_blocks(selection)
        dynamics_spec = numeric.dynamics_components(selection)
        static_names = tuple(numeric.static_factor_names(selection))
        sites = tuple(
            sorted(
                (
                    *(
                        site
                        for index, component in enumerate(dynamics_spec.components)
                        for site in component.iter_sites(f"vf_{index}", n_latent=len(state_ids))
                    ),
                    *(site for block in blocks for site in block.iter_sites()),
                    *numeric.likelihood_sites(selection),
                ),
                key=lambda site: site.name,
            )
        )
        state_index = {identity: index for index, identity in enumerate(state_ids)}
        states = tuple(
            CompiledState(
                identity,
                dynamical_model_spec.get_construct(identity).name,
                dynamical_model_spec.get_construct(identity).role == "exogenous",
                dynamical_model_spec.get_construct(identity).temporal_status == "time_invariant",
                family,
                tuple(
                    (state_index[edge.cause.id], edge.id)
                    for edge in selected_edges(selection)
                    if edge.effect.id == identity
                ),
            )
            for identity, family in zip(
                state_ids, numeric.diffusion_families(selection), strict=True
            )
        )
        state_index = {state.id: index for index, state in enumerate(states)}
        anchors = numeric.categorical_anchors(selection)
        clock_days = numeric.get_construct_dt_days(dynamical_model_spec)
        assert dynamical_model_spec.measurement_clock is not None

        parameters = execution_parameters(selection)
        bindings, auxiliary = bind_parameters(
            build_site_bindings(selection, sites, dynamics_spec.components, parameters),
            selection,
            parameters,
            sites,
        )
        from nof1_causal_lab.models.ssm.compile.observations import bind_observation_law

        binding_index = {binding.parameter_id: binding for binding in bindings}
        max_levels = max(numeric.observation_level_counts(selection), default=0)
        observations = tuple(
            CompiledObservation(
                indicator.observation.resolved(
                    indicator.observation.observation_window
                    or dynamical_model_spec.measurement_clock
                ),
                state_index[dynamical_model_spec.indicator_owner(indicator.observation.id).id],
                bind_observation_law(
                    indicator.likelihood.law,
                    indicator.likelihood.parsed,
                    binding_index,
                    channel=channel,
                    category_count=len(
                        indicator.observation.ordinal_levels
                        or indicator.observation.categorical_levels
                        or ()
                    ),
                    sampling_count=max_levels,
                    categorical_anchor=anchor,
                ),
                indicator.likelihood.standardized,
                anchor,
            )
            for channel, (indicator, anchor) in enumerate(zip(indicators, anchors, strict=True))
            if indicator.likelihood is not None
        )
        return CompiledDynamicalModel(
            clock_days=clock_days,
            states=states,
            observations=observations,
            static_factors=tuple(
                CompiledState(identity, name, False, True, DistributionFamily.GAUSSIAN, ())
                for identity, name in zip(
                    numeric.static_factor_ids(selection), static_names, strict=True
                )
            ),
            diffusion_block=blocks[0],
            loading_block=blocks[1],
            observation_mean_block=blocks[2],
            observation_noise_block=blocks[3],
            initial_mean_block=blocks[4],
            initial_covariance_block=blocks[5],
            static_scale_block=blocks[6],
            static_factor_loadings=numeric.static_factor_loadings(selection),
            dynamics=compile_dynamics(dynamics_spec),
            site_registry=sites,
            bindings=bindings,
            auxiliary_coordinates=auxiliary,
            laws=_compile_laws(selection, bindings, states),
            input_trajectories=_compile_input_trajectories(selection, states),
            state_space=compile_state_space(
                blocks[0],
                blocks[5],
                numeric.static_factor_loadings(selection),
                tuple(index for index, state in enumerate(states) if state.is_input),
                tuple(
                    index
                    for index, state in enumerate(states)
                    if state.time_invariant and not state.is_input
                ),
                dynamics_spec,
            ),
        )
    except IncompleteModelError as exc:
        return IncompleteModel(str(exc))
    except AggregatedCompileError as exc:
        return UnsupportedFit(tuple(dict.fromkeys(exc.errors)))


def _compile_input_trajectories(
    selection: StructuralSelection, states: tuple[CompiledState, ...]
) -> tuple[CompiledInputTrajectory, ...]:
    import numpy as np

    from nof1_causal_lab.numpyro_json import materialize_distribution

    dynamical_model_spec = selection.dynamical_model_spec
    result = []
    for index, state in enumerate(states):
        if not state.is_input:
            continue
        identity = dynamical_model_spec.get_construct(state.id).distribution
        assert identity is not None  # Execution readiness owns the required membership.
        layout = dynamical_model_spec.law_layouts[identity]
        law = materialize_distribution(dynamical_model_spec.distributions[identity])
        result.append(
            CompiledInputTrajectory(
                index=index,
                construct_id=state.id,
                layout=layout,
                values=tuple(
                    float(value)
                    for value in np.asarray(law.mean)[layout.trajectory_slices[state.id]]
                ),
            )
        )
    return tuple(result)


def _compile_laws(
    selection: StructuralSelection,
    bindings: tuple[CompiledParameterBinding, ...],
    states: tuple[CompiledState, ...],
) -> tuple[CompiledLaw, ...]:
    from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform
    from nof1_causal_lab.models.ssm.compile.bindings import joint_law_layout
    from nof1_causal_lab.models.ssm.compile.prior_compilation import quantity_parameter_law
    from nof1_causal_lab.numpyro_json import distribution_shape

    dynamical_model_spec = selection.dynamical_model_spec
    parameters = execution_parameters(selection)
    endogenous = tuple(state.id for state in states if not state.is_input)
    active = {
        member.distribution
        for member in (
            *parameters,
            *(dynamical_model_spec.get_construct(identity) for identity in endogenous),
        )
        if member.distribution is not None
    }
    result = []
    retained = set()
    for index, (identity, law) in enumerate(sorted(dynamical_model_spec.distributions.items())):
        if identity not in active:
            continue
        members = tuple(parameter for parameter in parameters if parameter.distribution == identity)
        trajectories = tuple(
            state
            for state in endogenous
            if dynamical_model_spec.get_construct(state).distribution == identity
        )
        retained.update(trajectories)
        layout = joint_law_layout(
            bindings,
            parameters=tuple(parameter.id for parameter in members),
            constructs=trajectories,
            time_points=dynamical_model_spec.time_points if trajectories else (),
        )
        batch_shape, event_shape = distribution_shape(law)
        if not batch_shape and not event_shape:
            parameter = members[0]
            law, _ = quantity_parameter_law(dynamical_model_spec, parameter)
        elif any(
            parameter.transform.kind != PriorAuthoringTransform.IDENTITY for parameter in members
        ):
            raise AggregatedCompileError(
                ["Joint probability laws must use native scientific coordinates"]
            )
        if event_shape:
            retained_layout = dynamical_model_spec.law_layouts[identity]
            if (
                layout.parameters != retained_layout.parameters
                or layout.constructs != retained_layout.constructs
            ):
                raise AggregatedCompileError(
                    ["Current execution coordinates do not support the retained joint law"]
                )
            layout = retained_layout
        result.append(CompiledLaw(index, law, layout))
    if retained and retained != set(endogenous):
        raise AggregatedCompileError(
            ["Conditional simulation requires a joint draw for every state"]
        )
    return tuple(result)


def compile_executable_model(selection: StructuralSelection) -> CompiledDynamicalModel:
    """Parse the authored execution boundary for callers whose contract requires success.

    Fit readiness consumes compile_model's alternatives directly. At an
    executing shell/owner boundary these specific semantic errors preserve the
    same failure distinction while returning the resolved input to the caller.
    """
    from typing import assert_never

    outcome = compile_model(selection)
    match outcome:
        case CompiledDynamicalModel():
            return outcome
        case IncompleteModel():
            raise IncompleteModelError(outcome.message)
        case UnsupportedFit():
            raise AggregatedCompileError(list(outcome.errors))
    return assert_never(outcome)


def compile_ssm_inputs_from_model(
    selection: StructuralSelection,
) -> FitCompilationResult:
    """Resolve fitting once; incomplete/unsupported choices remain editable.

    Only the compiler's expected diagnostic exceptions become variants. Broken
    internal assumptions (including other ValueErrors) still propagate.
    """
    return compile_fit_inputs(compile_model(selection), selection)


def compile_fit_inputs(
    compiled: ModelCompilationResult, selection: StructuralSelection
) -> FitCompilationResult:
    """Resolve fitting laws once against the already compiled shared execution facts."""
    if not isinstance(compiled, CompiledDynamicalModel):
        return compiled
    try:
        prior_registry, _, diagnostics = compile_priors(compiled, selection)
        diagnostics = _attach_compile_binding_provenance(diagnostics, compiled.bindings)
        prior_runtime_bundle = build_prior_runtime_bundle(compiled, prior_registry)
    except IncompleteModelError as exc:
        return IncompleteModel(str(exc))
    except AggregatedCompileError as exc:
        return UnsupportedFit(tuple(dict.fromkeys(exc.errors)))
    return CompiledFitInputs(
        compiled_dynamical_model=compiled,
        prior_runtime_bundle=prior_runtime_bundle,
        diagnostics=tuple(diagnostics),
    )
