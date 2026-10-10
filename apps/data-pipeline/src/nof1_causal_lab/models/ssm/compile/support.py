"""Pure derivations of numerical support from scientific structure and policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.expressions import COEFFICIENT_MEANINGS
from nof1_causal_lab.artifacts.likelihood import OBSERVATION_FAMILY_SPECS, CategoricalLawSpec
from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.models.model_parameters import coefficient_value, execution_coefficient_uses
from nof1_causal_lab.models.model_structure import selected_indicators, selected_state_ids
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseBlockSpec,
    T0CholBlockSpec,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from nof1_causal_lab.artifacts.dynamical_model_spec import DynamicalModelSpec
    from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId, ParameterId
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec
    from nof1_causal_lab.distributions import DistributionFamily
    from nof1_causal_lab.models.model_parameters import CoefficientUse
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.models.ssm.dynamics.expression import ExpressionComponentSpec

from nof1_causal_lab.artifacts.likelihood import DistributionFamily
from nof1_causal_lab.compilation_errors import AggregatedCompileError
from nof1_causal_lab.models.model_semantics import (
    indicator_requires_observation_intercept,
)
from nof1_causal_lab.models.ssm.structure.assembly import (
    dense_vector_positions,
    rect_matrix_positions,
)
from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor
from nof1_causal_lab.utils.model_structure import (
    get_model_clock,
    get_reference_indicator_lookup,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.dynamical_model_spec import DynamicalModelSpec


class NumericalSupportError(AggregatedCompileError):
    """Aggregate failures to derive numerical support from DynamicalModelSpec."""

    header = "Numerical derivation failed"


def get_construct_dt_days(
    dynamical_model_spec: DynamicalModelSpec,
) -> float:
    """Get the model clock interval in fractional days."""
    return get_model_clock(dynamical_model_spec).days


def categorical_anchors(selection: StructuralSelection) -> tuple[bool, ...]:
    """Pin the reference contrast only for states with exclusively nominal emissions."""
    dynamical_model_spec = selection.dynamical_model_spec
    references = get_reference_indicator_lookup(selection)
    categorical_states = {
        identity
        for identity in selected_state_ids(selection)
        if all(
            indicator.likelihood is not None
            and indicator.likelihood.law.family == DistributionFamily.CATEGORICAL
            for indicator in dynamical_model_spec.get_construct(identity).indicators
        )
    }
    return tuple(
        dynamical_model_spec.indicator_owner(indicator.observation.id).id in categorical_states
        and indicator.observation.name
        == references[dynamical_model_spec.indicator_owner(indicator.observation.id).name]
        for indicator in observed_indicators(selection)
    )


def _build_static_factor_structure(
    selection: StructuralSelection, latent_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Derive baseline-factor incidence and fixed loadings from explicit common causes."""
    from nof1_causal_lab.artifacts.expressions import linear_coefficient
    from nof1_causal_lab.models.model_parameters import baseline_factor_groups, coefficient_value

    dynamical_model_spec = selection.dynamical_model_spec
    groups = baseline_factor_groups(selection)
    factors = [group[0] for group in groups]
    state_index = {identity: index for index, identity in enumerate(selected_state_ids(selection))}
    loadings = np.zeros((len(latent_names), len(factors)))
    for index, group in enumerate(groups):
        sources = {construct.id for construct in group}
        for source in sources:
            construct = dynamical_model_spec.get_construct(source)
            initial_mean = construct.coefficient("initial_mean")
            assert initial_mean is not None
            if coefficient_value(initial_mean) != 0.0:
                raise NumericalSupportError(
                    ["Marginalized baseline factors require a fixed zero initial mean"]
                )
            if (
                construct.indicators
                or construct.temporal_status != "time_invariant"
                or any(edge.effect.id == source for edge in dynamical_model_spec.edges)
            ):
                raise NumericalSupportError(
                    ["A marginalized baseline factor must be an unmeasured time-invariant root"]
                )
            for edge in dynamical_model_spec.edges:
                if edge.cause.id != source or edge.effect.id not in state_index:
                    continue
                weight = 1.0
                if edge.mechanisms:
                    if len(edge.mechanisms) != 1:
                        raise NumericalSupportError(
                            ["A marginalized factor loading requires one fixed linear coefficient"]
                        )
                    coefficient = coefficient_value(
                        linear_coefficient(edge.mechanisms[0].expression, source)
                    )
                    if coefficient is None:
                        raise NumericalSupportError(
                            ["A marginalized factor loading requires one fixed linear coefficient"]
                        )
                    weight = coefficient
                loadings[state_index[edge.effect.id], index] = weight
        if not np.any(loadings[:, index]):
            raise NumericalSupportError(["A baseline factor must affect a retained state"])
    return (
        np.ones(len(factors), dtype=bool),
        np.zeros(len(factors)),
        np.asarray(loadings),
        tuple(factor.name for factor in factors),
    )


def state_ids(selection: StructuralSelection) -> tuple[ConstructId, ...]:
    """Return retained construct IDs in the selection's execution-state order."""
    return selected_state_ids(selection)


def state_names(selection: StructuralSelection) -> tuple[str, ...]:
    """Return state display labels in the same order as the compiled state axis."""
    dynamical_model_spec = selection.dynamical_model_spec
    return tuple(
        dynamical_model_spec.get_construct(identity).name for identity in state_ids(selection)
    )


def n_states(selection: StructuralSelection) -> int:
    """Count the construct coordinates retained on the compiled state axis."""
    return len(state_ids(selection))


def observed_indicators(selection: StructuralSelection) -> tuple[IndicatorSpec, ...]:
    """Return model-owned indicators in the selection's compiled observation order."""
    dynamical_model_spec = selection.dynamical_model_spec
    return tuple(
        dynamical_model_spec.indicator(identity)
        for identity in tuple(
            indicator.observation.id for indicator in selected_indicators(selection)
        )
    )


def observation_ids(selection: StructuralSelection) -> tuple[IndicatorId, ...]:
    """Return stable observation IDs in compiled indicator order."""
    return tuple(indicator.observation.id for indicator in observed_indicators(selection))


def observation_names(selection: StructuralSelection) -> tuple[str, ...]:
    """Return observation display labels in compiled indicator order."""
    return tuple(indicator.observation.name for indicator in observed_indicators(selection))


def n_observations(selection: StructuralSelection) -> int:
    """Count observation channels in the compiled selection, not recorded rows."""
    return len(observed_indicators(selection))


def _likelihoods(selection: StructuralSelection) -> Iterator[LikelihoodSpec]:
    for indicator in observed_indicators(selection):
        if indicator.likelihood is None:
            raise IncompleteModelError(
                f"Retained indicator {indicator.observation.name!r} requires a likelihood"
            )
        yield indicator.likelihood


def observation_families(selection: StructuralSelection) -> tuple[DistributionFamily, ...]:
    """Return each selected observation's likelihood family in compiled indicator order."""
    return tuple(likelihood.law.family for likelihood in _likelihoods(selection))


def observation_level_counts(selection: StructuralSelection) -> tuple[int, ...]:
    """Count declared levels for ordinal and categorical observations, using zero otherwise."""
    return tuple(
        len(indicator.observation.ordinal_levels or indicator.observation.categorical_levels or ())
        if indicator.likelihood is not None
        and indicator.likelihood.law.family
        in {DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL}
        else 0
        for indicator in observed_indicators(selection)
    )


def quantity_position(selection: StructuralSelection, parameter: CoefficientUse) -> tuple[int, ...]:
    """Locate a scientific scalar by its owners in the derived execution axes."""
    state = {key: i for i, key in enumerate(state_ids(selection))}
    observation = {key: i for i, key in enumerate(observation_ids(selection))}
    owners = {owner.id for owner in parameter.owners}

    def one(axis: Mapping[ConstructId, int] | Mapping[IndicatorId, int]) -> int:
        matches = [index for key, index in axis.items() if key in owners]
        if len(matches) != 1:
            raise NumericalSupportError(
                [f"Quantity {parameter.slot!r} needs one owner on this axis"]
            )
        return matches[0]

    kind = parameter.quantity
    if kind == SiteKind.LOADING:
        return (one(observation), one(state))
    if kind in {SiteKind.DIFFUSION_LOWER, SiteKind.T0_VAR_LOWER}:
        pair = sorted(index for key, index in state.items() if key in owners)
        if len(pair) != 2:
            raise NumericalSupportError(
                [f"Covariance quantity {parameter.slot!r} requires two distinct state owners"]
            )
        return (pair[1], pair[0])
    if kind in {SiteKind.MANIFEST_MEANS, SiteKind.MANIFEST_VAR_DIAG}:
        return (one(observation),)
    if kind == SiteKind.STATIC_STATE_SD:
        from nof1_causal_lab.models.model_parameters import baseline_factor_groups

        matches = [
            index
            for index, group in enumerate(baseline_factor_groups(selection))
            if owners & {construct.id for construct in group}
        ]
        if len(matches) != 1:
            raise NumericalSupportError(["A baseline scale must belong to one identifiable factor"])
        return (matches[0],)
    return (one(state),)


def _quantity_values(
    selection: StructuralSelection,
    kind: SiteKind,
    template: np.ndarray,
    support: np.ndarray,
    *,
    diagonal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply explicit scalar definitions to the scientific default policy."""
    values = np.array(template, dtype=float, copy=True)
    free = np.array(support, dtype=bool, copy=True)
    occupied = {}
    for parameter in execution_coefficient_uses(selection):
        if parameter.quantity != kind:
            continue
        if kind == SiteKind.T0_MEANS and not any(
            owner.id in state_ids(selection) for owner in parameter.owners
        ):
            continue
        position = quantity_position(selection, parameter)
        value = coefficient_value(parameter.value)
        if (
            kind in {SiteKind.DIFFUSION_DIAG, SiteKind.DIFFUSION_LOWER}
            and any(time_invariant_mask(selection)[index] for index in position)
            and value != 0.0
        ):
            raise NumericalSupportError(["Time-invariant constructs cannot have innovations"])
        if kind in {SiteKind.DIFFUSION_LOWER, SiteKind.T0_VAR_LOWER}:
            expected_kind = (
                "innovation_correlation"
                if kind == SiteKind.DIFFUSION_LOWER
                else "initial_state_correlation"
            )
            pair = {state_ids(selection)[index] for index in position}
            if not any(
                kind == expected_kind and {first, second} == pair
                for first, second, kind in selection.induced_dependencies
            ):
                raise NumericalSupportError(
                    [
                        "Correlated quantities require an explicit latent confounder in the scientific DAG"
                    ]
                )
        if (
            value is not None
            and kind
            in {
                SiteKind.DIFFUSION_DIAG,
                SiteKind.MANIFEST_VAR_DIAG,
                SiteKind.T0_VAR_DIAG,
                SiteKind.STATIC_STATE_SD,
            }
            and value < 0
        ):
            raise NumericalSupportError(["A fixed standard deviation cannot be negative"])
        if kind == SiteKind.STATIC_STATE_SD and occupied.get(position) == parameter.value:
            continue
        if position in occupied:
            raise NumericalSupportError(
                [f"Multiple scientific definitions for {kind.value} at {position}"]
            )
        occupied[position] = parameter.value
        index = (position[0], position[0]) if diagonal else position
        support_index = index if free.ndim == len(index) else position
        free[support_index] = value is None
        if value is not None:
            values[index] = value
    return values, free


def loading_block(selection: StructuralSelection) -> SparseBlockSpec[tuple[int, int]]:
    """Compile the observation-by-state loading template and its free coefficient positions."""
    shape = (n_observations(selection), n_states(selection))
    template, support = _quantity_values(
        selection, SiteKind.LOADING, np.zeros(shape), np.zeros(shape, dtype=bool)
    )
    return SparseBlockSpec[tuple[int, int]](
        free_support=support,
        template=jnp.asarray(template),
        free_positions=tuple(rect_matrix_positions(support, *shape)),
        free_site_name="lambda_free",
        support=SupportClass.REAL,
        site_kind=SiteKind.LOADING,
        assembly_group="lambda",
        prior_field="lambda_free",
    )


def observation_mean_block(selection: StructuralSelection) -> SparseBlockSpec[int]:
    """Compile active observation intercepts, rejecting free intercepts unused by their laws."""
    inactive = [
        indicator.observation.name
        for indicator in observed_indicators(selection)
        if indicator.likelihood is not None
        and isinstance(indicator.likelihood.parsed.intercept.value, str)
        and not indicator_requires_observation_intercept(
            indicator.likelihood.law.family,
            indicator.likelihood.parsed.link,
            standardized=indicator.likelihood.standardized,
        )
    ]
    if inactive:
        raise NumericalSupportError(
            [
                f"Observation intercept 'manifest_mean_{name}' is inactive for the locked likelihood "
                "semantics; fix the channel location through its compiler-owned anchor instead."
                for name in inactive
            ]
        )
    template, support = _quantity_values(
        selection,
        SiteKind.MANIFEST_MEANS,
        np.zeros(n_observations(selection)),
        np.zeros(n_observations(selection), dtype=bool),
    )
    return SparseBlockSpec[int](
        free_support=support,
        template=jnp.asarray(template),
        free_positions=tuple(dense_vector_positions(support, n_observations(selection))),
        free_site_name="manifest_means_free",
        support=SupportClass.REAL,
        site_kind=SiteKind.MANIFEST_MEANS,
        assembly_group="manifest",
        prior_field="manifest_means",
    )


def observation_noise_block(selection: StructuralSelection) -> ManifestCholBlockSpec:
    """Compile the observation-noise template and free diagonal scale positions."""
    n = n_observations(selection)
    template, support = _quantity_values(
        selection,
        SiteKind.MANIFEST_VAR_DIAG,
        np.zeros((n, n)),
        np.zeros(n, dtype=bool),
        diagonal=True,
    )
    return ManifestCholBlockSpec(
        n_manifest=n_observations(selection), diag_support=support, template=jnp.asarray(template)
    )


def time_invariant_mask(selection: StructuralSelection) -> np.ndarray:
    """Mark compiled state coordinates whose constructs are time-invariant."""
    dynamical_model_spec = selection.dynamical_model_spec
    return np.asarray(
        [
            dynamical_model_spec.get_construct(identity).temporal_status == "time_invariant"
            for identity in state_ids(selection)
        ],
        dtype=bool,
    )


def input_mask(selection: StructuralSelection) -> np.ndarray:
    """Coordinates read from the panel instead of generated under a state law."""
    dynamical_model_spec = selection.dynamical_model_spec
    return np.asarray(
        [
            dynamical_model_spec.get_construct(identity).role == "exogenous"
            for identity in state_ids(selection)
        ],
        dtype=bool,
    )


def diffusion_families(selection: StructuralSelection) -> tuple[DistributionFamily, ...]:
    """Resolve innovation families, requiring diffusion scales for dynamic endogenous states."""
    from nof1_causal_lab.distributions import DistributionFamily

    dynamical_model_spec = selection.dynamical_model_spec
    result = []
    for identity in state_ids(selection):
        construct = dynamical_model_spec.get_construct(identity)
        if construct.role == "exogenous" or construct.temporal_status == "time_invariant":
            result.append(DistributionFamily.GAUSSIAN)
        elif construct.coefficient("diffusion_scale") is None:
            raise IncompleteModelError(f"Construct {construct.name!r} requires a diffusion scale")
        else:
            result.append(construct.innovation_family)
    return tuple(result)


def diffusion_block(selection: StructuralSelection) -> DiffusionBlockSpec:
    """Compile diffusion Cholesky support, including induced correlations and zero static rows."""
    count = n_states(selection)
    support = np.eye(count, dtype=bool)
    axis = {identity: index for index, identity in enumerate(state_ids(selection))}
    for first_id, second_id, kind in selection.induced_dependencies:
        if kind == "innovation_correlation":
            first, second = axis[first_id], axis[second_id]
            support[max(first, second), min(first, second)] = True
    static = time_invariant_mask(selection) | input_mask(selection)
    support[static, :] = False
    support[:, static] = False
    template, support = _quantity_values(
        selection, SiteKind.DIFFUSION_DIAG, np.eye(count), np.zeros_like(support), diagonal=True
    )
    template, support = _quantity_values(selection, SiteKind.DIFFUSION_LOWER, template, support)
    template[static, :] = 0.0
    template[:, static] = 0.0
    support[static, :] = False
    support[:, static] = False
    return DiffusionBlockSpec(
        n_latent=count,
        diffusion_chol_support=support,
        diffusion_chol_template=jnp.asarray(template),
    )


def initial_mean_block(selection: StructuralSelection) -> SparseBlockSpec[int]:
    """Compile fixed and free initial-state mean coordinates in state-axis order."""
    support = np.zeros(n_states(selection), dtype=bool)
    template, support = _quantity_values(
        selection, SiteKind.T0_MEANS, np.zeros(n_states(selection)), support
    )
    return SparseBlockSpec[int](
        free_support=support,
        template=jnp.asarray(template),
        free_positions=tuple(dense_vector_positions(support, n_states(selection))),
        free_site_name="t0_means_free",
        support=SupportClass.REAL,
        site_kind=SiteKind.T0_MEANS,
        assembly_group="t0",
        prior_field="t0_means",
    )


def initial_covariance_block(selection: StructuralSelection) -> T0CholBlockSpec:
    """Compile initial-state scales and correlations into a Cholesky template with fixed inputs."""
    count = n_states(selection)
    support = np.zeros(count, dtype=bool)
    std, support = _quantity_values(selection, SiteKind.T0_VAR_DIAG, np.ones(count), support)
    std[input_mask(selection)] = 0.0
    correlations, correlation_support = _quantity_values(
        selection, SiteKind.T0_VAR_LOWER, np.eye(count), np.zeros((count, count), dtype=bool)
    )
    lower = np.tril(np.asarray(correlations), -1)
    corr = np.eye(count) + lower + lower.T
    template = jnp.asarray(np.asarray(std)[:, None] * np.linalg.cholesky(corr))
    return T0CholBlockSpec(
        n_latent=count,
        diag_support=support,
        correlation_support=correlation_support,
        template=jnp.asarray(template),
    )


def static_factor_ids(selection: StructuralSelection) -> tuple[ConstructId, ...]:
    """Return the representative construct identity for each compiled baseline-factor group."""
    from nof1_causal_lab.models.model_parameters import baseline_factor_groups

    return tuple(group[0].id for group in baseline_factor_groups(selection))


def static_factor_names(selection: StructuralSelection) -> tuple[str, ...]:
    """Label baseline factors from their initial-scale parameters or owning constructs."""
    dynamical_model_spec = selection.dynamical_model_spec
    names = []
    for identity in static_factor_ids(selection):
        construct = dynamical_model_spec.get_construct(identity)
        coefficient = construct.coefficient("initial_scale")
        assert coefficient is not None
        names.append(
            dynamical_model_spec.parameter(coefficient).name
            if isinstance(coefficient, str)
            else construct.name
        )
    return tuple(names)


def static_factor_loadings(selection: StructuralSelection) -> jnp.ndarray:
    """Build the state-by-factor loadings used to represent marginalized static effects."""
    return jnp.asarray(_build_static_factor_structure(selection, state_names(selection))[2])


def static_scale_block(selection: StructuralSelection) -> SparseBlockSpec[int]:
    """Compile positive baseline-factor scales and their free parameter positions."""
    n = len(static_factor_ids(selection))
    values, support = _quantity_values(
        selection, SiteKind.STATIC_STATE_SD, np.zeros(n), np.ones(n, dtype=bool)
    )
    return SparseBlockSpec[int](
        free_support=support,
        template=jnp.asarray(values),
        free_positions=tuple(dense_vector_positions(support, len(values))),
        free_site_name="static_state_sd_free",
        support=SupportClass.POSITIVE,
        site_kind=SiteKind.STATIC_STATE_SD,
        assembly_group="t0",
        prior_field="static_state_sd",
    )


def dynamics_expressions(selection: StructuralSelection) -> tuple[ExpressionComponentSpec, ...]:
    """Lower authored dynamics mechanisms into executable symbolic component specifications."""
    from nof1_causal_lab.models.ssm.compile.mechanisms import lower_mechanisms

    return lower_mechanisms(selection)


def dynamics_components(selection: StructuralSelection) -> DynamicsSpec:
    """Pair lowered dynamics components with their shared compiled state dimension."""
    return DynamicsSpec(n_latent=n_states(selection), components=dynamics_expressions(selection))


def parameter_blocks(
    selection: StructuralSelection,
) -> tuple[
    DiffusionBlockSpec,
    SparseBlockSpec[tuple[int, int]],
    SparseBlockSpec[int],
    ManifestCholBlockSpec,
    SparseBlockSpec[int],
    T0CholBlockSpec,
    SparseBlockSpec[int],
]:
    """Compile all parameter blocks in the order expected by SSM assembly.

    Args:
        selection: Scientific model and outcome determining the execution structure.

    Returns:
        Diffusion, loadings, observation means, observation noise, initial means,
        initial covariance, and static-factor scale blocks, in that order.
    """
    return (
        diffusion_block(selection),
        loading_block(selection),
        observation_mean_block(selection),
        observation_noise_block(selection),
        initial_mean_block(selection),
        initial_covariance_block(selection),
        static_scale_block(selection),
    )


def _require_execution_choices(selection: StructuralSelection) -> None:
    """Check numerical execution requirements."""
    from nof1_causal_lab.distributions import DistributionFamily
    from nof1_causal_lab.models.model_structure import validate_execution_structure

    validate_execution_structure(selection)
    dynamical_model_spec = selection.dynamical_model_spec

    def require_hyperparameter(coefficient: float | ParameterId | None, label: str) -> None:
        if not isinstance(coefficient, str):
            raise IncompleteModelError(f"{label} requires a prior parameter")

    for identity in state_ids(selection):
        construct = dynamical_model_spec.get_construct(identity)
        if construct.role == "exogenous":
            if construct.distribution is None:
                raise IncompleteModelError(
                    f"Exogenous construct {construct.name!r} requires a deterministic trajectory law"
                )
            continue
        if any(construct.coefficient(role) is None for role in ("initial_mean", "initial_scale")):
            raise IncompleteModelError(
                f"Construct {construct.name!r} requires initial-state coefficients"
            )
        if (
            construct.temporal_status == "time_varying"
            and construct.coefficient("diffusion_scale") is None
        ):
            raise IncompleteModelError(
                f"Construct {construct.name!r} requires an innovation distribution"
            )
        if construct.innovation_family == DistributionFamily.STUDENT_T:
            require_hyperparameter(
                construct.coefficient("process_degrees_of_freedom"),
                f"{construct.name}.process_degrees_of_freedom",
            )
    for indicator in observed_indicators(selection):
        likelihood = indicator.likelihood
        if likelihood is None:
            raise IncompleteModelError(
                f"Retained indicator {indicator.observation.name!r} requires a likelihood"
            )
        terms = likelihood.parsed
        missing = [operand.role for operand in terms.operands if operand.value is None]
        if missing:
            raise IncompleteModelError(
                f"Indicator {indicator.observation.name!r} requires explicit measurement coefficients: {missing}"
            )
        for operand in terms.auxiliary:
            if operand.role != "observation_scale":
                require_hyperparameter(
                    operand.value, f"{indicator.observation.name}.likelihood.{operand.role}"
                )


def likelihood_sites(selection: StructuralSelection) -> tuple[SiteDescriptor, ...]:
    """Declare emission/process hyperparameters in their native sampling order."""
    sites = []
    families = set(observation_families(selection))
    for law in OBSERVATION_FAMILY_SPECS:
        if len(law.parameter_roles) != 1:
            continue
        meaning = COEFFICIENT_MEANINGS[law.parameter_roles[0]]
        family, kind = law.family, meaning.quantity
        name = kind.value
        if family in families:
            sites.append(
                SiteDescriptor(
                    name=name,
                    shape=(),
                    support=meaning.support,
                    assembly_group="likelihood",
                    site_kind=kind,
                    prior_field=name,
                )
            )
    n = n_observations(selection)
    cutpoints = max(max(observation_level_counts(selection), default=0) - 1, 0)
    if DistributionFamily.ORDERED_LOGISTIC in families and cutpoints:
        base = COEFFICIENT_MEANINGS["cutpoint_base"]
        sites.append(
            SiteDescriptor(
                name=base.quantity.value,
                shape=(n,),
                support=base.support,
                assembly_group="likelihood",
                site_kind=base.quantity,
                prior_field=base.quantity.value,
            )
        )
        if cutpoints > 1:
            gaps = COEFFICIENT_MEANINGS["cutpoint_gaps"]
            sites.append(
                SiteDescriptor(
                    name=gaps.quantity.value,
                    shape=(n, cutpoints - 1),
                    support=gaps.support,
                    assembly_group="likelihood",
                    site_kind=gaps.quantity,
                    prior_field=gaps.quantity.value,
                )
            )
    if DistributionFamily.CATEGORICAL in families and cutpoints:
        for role in CategoricalLawSpec.parameter_roles:
            meaning = COEFFICIENT_MEANINGS[role]
            kind = meaning.quantity
            name = kind.value
            sites.append(
                SiteDescriptor(
                    name=name,
                    shape=(n, cutpoints),
                    support=meaning.support,
                    assembly_group="likelihood",
                    site_kind=kind,
                    prior_field=name,
                )
            )
    if DistributionFamily.STUDENT_T in diffusion_families(selection):
        meaning = COEFFICIENT_MEANINGS["process_degrees_of_freedom"]
        sites.append(
            SiteDescriptor(
                name=meaning.quantity.value,
                shape=(),
                support=meaning.support,
                assembly_group="process",
                site_kind=meaning.quantity,
                prior_field=meaning.quantity.value,
            )
        )
    return tuple(sites)
