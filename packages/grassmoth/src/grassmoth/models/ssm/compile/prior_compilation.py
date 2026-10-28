"""Pure prior-compilation and binding stages for SSM compilation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import numpyro.distributions as dist
import scipy.linalg

from grassmoth.artifacts.model.parameter import (
    ParameterCoordinate,
    PriorAuthoringTransform,
    SiteKind,
)
from grassmoth.artifacts.model.parameter_spec import (
    InitialCorrelationTransformSpec,
    IntervalEffectTransformSpec,
    PersistenceTransformSpec,
)
from grassmoth.artifacts.probability.native import (
    batch_prior_distributions,
    interval_effect_to_rate,
    persistence_to_decay,
    prior_reference_value,
)
from grassmoth.artifacts.reports.prior import PriorValidationResult
from grassmoth.compilation_errors import AggregatedCompileError
from grassmoth.models.ssm import compiled_access as numeric
from grassmoth.models.ssm.compile.bindings import (
    CompiledEffectInterval,
    CompiledParameterBinding,
    resolve_site_selection,
)
from grassmoth.models.ssm.execution.contracts import NUMERICAL_EPSILON
from grassmoth.models.ssm.parameterization import build_site_registry
from grassmoth.models.ssm.priors import (
    default_prior_for_descriptor,
    site_constraint,
    validate_site_prior,
)
from grassmoth.models.ssm.structure.sites import (
    CompiledEdgeTarget,
    CompiledNodeTarget,
    site_size,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth.artifacts.model.parameter_spec import ParameterSpec
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
    from grassmoth.models.ssm.compile.prior_indexing import CompiledBindingRegistry
    from grassmoth.models.ssm.structure.sites import SiteDescriptor
    from grassmoth.models.structure import StructuralSelection

CompileDiagnostic = PriorValidationResult
_LOGM_IMAG_TOL = 1e-8
_LOGM_RELATIVE_DEVIATION_WARNING_THRESHOLD = 0.2

_DEGENERATE_PRIOR_PREAMBLE = (
    "model-spec priors must have strictly positive variance. Represent a fixed value "
    "with a literal coefficient in its component slot."
)


class PriorCompilationError(AggregatedCompileError):
    """Aggregate independent prior-compilation failures into one exception."""

    header = "Prior compilation failed"


def _decay_bindings(
    compiled_dynamical_model: CompiledDynamicalModel,
) -> tuple[tuple[CompiledParameterBinding, CompiledNodeTarget | CompiledEdgeTarget], ...]:
    return tuple(
        (binding, binding.target)
        for binding in compiled_dynamical_model.bindings
        if isinstance(binding.target, (CompiledNodeTarget, CompiledEdgeTarget))
        and binding.site.site_kind == SiteKind.DYNAMICS_DECAY
        and binding.transform == PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY
    )


def _linear_effect_bindings(
    compiled_dynamical_model: CompiledDynamicalModel,
) -> tuple[tuple[CompiledParameterBinding, CompiledEdgeTarget], ...]:
    """Transformed linear effects carry their resolved edge coordinates."""
    return tuple(
        (binding, binding.target)
        for binding in compiled_dynamical_model.bindings
        if isinstance(binding.target, CompiledEdgeTarget)
        and binding.site.site_kind == SiteKind.DYNAMICS_WEIGHT
        and binding.transform == PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE
    )


def _resolve_transform_interval_days(
    transform: PersistenceTransformSpec | IntervalEffectTransformSpec,
    dynamical_model_spec: DynamicalModelSpec,
) -> float:
    """Resolve the explicit interval reference once at the law's compiler boundary."""
    from grassmoth.models.ssm.compile.layout.axes import get_construct_dt_days

    return (
        get_construct_dt_days(dynamical_model_spec)
        if transform.interval_days == "model_clock"
        else transform.interval_days
    )


def _format_interval_days(days: float) -> str:
    """Render a positive day interval for diagnostics."""
    return f"{float(days):.1f}d"


def _compile_warning(
    binding: CompiledParameterBinding, *, issue: str, suggested_adjustment: str
) -> CompileDiagnostic:
    """Build a typed non-fatal compile diagnostic for one compiled prior coordinate."""
    parameter = binding.site.prior_field or binding.site.name
    return CompileDiagnostic(
        parameter=parameter,
        issue=issue,
        suggested_adjustment=suggested_adjustment,
        related_parameters=(parameter,),
        compiled_site_name=binding.site.name,
        compiled_flat_index=binding.flat_index,
    )


def collect_compile_diagnostics(
    compiled_dynamical_model: CompiledDynamicalModel,
    *,
    prior_registry: dict[str, dist.Distribution] | None = None,
    offdiag_interval_days: dict[tuple[int, int], float] | None = None,
) -> list[CompileDiagnostic]:
    """Collect structured compiler diagnostics for downstream consumers."""
    diagnostics: list[CompileDiagnostic] = []

    if prior_registry is not None:
        diagnostics.extend(
            collect_first_order_approximation_warnings(
                prior_registry,
                compiled_dynamical_model=compiled_dynamical_model,
                offdiag_interval_days=offdiag_interval_days,
            )
        )
    return diagnostics


def collect_first_order_approximation_warnings(
    prior_registry: dict[str, dist.Distribution],
    *,
    compiled_dynamical_model: CompiledDynamicalModel,
    offdiag_interval_days: dict[tuple[int, int], float] | None = None,
) -> list[CompileDiagnostic]:
    """Return warnings when exact matrix-log DT->CT diagnostics diverge from beta/dt."""
    taylor_drift = _assemble_reference_drift_from_component_priors(
        prior_registry, compiled_dynamical_model
    )
    if taylor_drift is None:
        return []

    diag_abs = np.abs(np.diag(taylor_drift))
    eligible_mask = diag_abs >= NUMERICAL_EPSILON
    if not np.any(eligible_mask):
        return []
    eligible_indices = np.where(eligible_mask)[0]
    min_diag = float(np.min(diag_abs[eligible_indices]))
    if min_diag < NUMERICAL_EPSILON:
        return []
    min_diag_latent_idx = int(eligible_indices[int(np.argmin(diag_abs[eligible_indices]))])
    min_diag_name = next(
        (
            binding.parameter_name
            for binding, target in _decay_bindings(compiled_dynamical_model)
            if target.target_index == min_diag_latent_idx
        ),
        None,
    )
    min_diag_label = f"{min_diag_name}" if min_diag_name else f"latent[{min_diag_latent_idx}]"

    warnings: list[CompileDiagnostic] = []
    latent_names = numeric.state_names(compiled_dynamical_model)
    for binding, target in _linear_effect_bindings(compiled_dynamical_model):
        prior = prior_registry.get(binding.site.name)
        if prior is None:
            continue
        offdiag_mu = np.asarray(prior_reference_value(prior)).reshape(-1)
        if offdiag_mu.size == 0:
            continue
        offdiag_value = _value_at(offdiag_mu, binding.flat_index, default=0.0)
        interval_days = _resolve_offdiag_interval_days(
            effect_idx=target.target_index,
            cause_idx=target.source_index,
            offdiag_interval_days=offdiag_interval_days,
        )
        if interval_days is None:
            continue

        cause_name = latent_names[target.source_index]
        effect_name = latent_names[target.target_index]
        offdiag_label = f"{binding.parameter_name} ({cause_name} -> {effect_name})"

        try:
            exact_drift = matrix_log_diagnostic_drift(
                taylor_drift,
                interval_days=interval_days,
            )
        except PriorCompilationError as exc:
            warnings.append(
                _compile_warning(
                    binding,
                    issue=f"{offdiag_label}: exact matrix-log CT diagnostic failed: {exc}",
                    suggested_adjustment=(
                        "Shrink the DT beta prior or elicit the prior directly on a real, stable "
                        "CT drift scale."
                    ),
                )
            )
            continue

        exact_value = float(exact_drift[target.target_index, target.source_index])
        deviation = abs(exact_value - float(offdiag_value)) / max(
            abs(exact_value), NUMERICAL_EPSILON
        )
        ratio = abs(exact_value) / min_diag
        if deviation <= _LOGM_RELATIVE_DEVIATION_WARNING_THRESHOLD and ratio <= 0.2:
            continue
        warnings.append(
            _compile_warning(
                binding,
                issue=(
                    f"{offdiag_label}: matrix-log mismatch; exact CT coupling at "
                    f"{_format_interval_days(interval_days)} is {abs(exact_value):.3f} 1/day "
                    f"versus the elementwise beta/dt value {abs(float(offdiag_value)):.3f} "
                    f"1/day; logm deviation is {deviation * 100:.0f}% and the exact coupling "
                    f"is {ratio * 100:.0f}% of the smallest realised CT diagonal damping "
                    f"({min_diag:.3f} 1/day, {min_diag_label})."
                ),
                suggested_adjustment=(
                    "Use the exact matrix-log CT scale when revising this edge: shorten the "
                    "reference interval, shrink the DT beta prior, or elicit the prior directly "
                    "on the CT rate."
                ),
            )
        )
    return warnings


def _value_at(values: np.ndarray, flat_index: int, *, default: float) -> float:
    if values.size == 0:
        return float(default)
    if values.size == 1:
        return float(values[0])
    if flat_index < values.size:
        return float(values[flat_index])
    return float(default)


def _assemble_reference_drift_from_component_priors(
    prior_registry: dict[str, dist.Distribution],
    compiled_dynamical_model: CompiledDynamicalModel,
) -> np.ndarray | None:
    drift = np.zeros(
        (numeric.n_states(compiled_dynamical_model), numeric.n_states(compiled_dynamical_model)),
        dtype=float,
    )
    populated = False

    for binding, target in _decay_bindings(compiled_dynamical_model):
        prior = prior_registry.get(binding.site.name)
        if prior is None:
            continue
        decay_mu = np.asarray(prior_reference_value(prior)).reshape(-1)
        if decay_mu.size == 0:
            continue
        drift[target.target_index, target.target_index] = -_value_at(
            decay_mu,
            binding.flat_index,
            default=0.0,
        )
        populated = True

    for binding, target in _linear_effect_bindings(compiled_dynamical_model):
        prior = prior_registry.get(binding.site.name)
        if prior is None:
            continue
        weight_mu = np.asarray(prior_reference_value(prior)).reshape(-1)
        if weight_mu.size == 0:
            continue
        drift[target.target_index, target.source_index] = _value_at(
            weight_mu,
            binding.flat_index,
            default=0.0,
        )
        populated = True

    return drift if populated else None


def _resolve_offdiag_interval_days(
    *,
    effect_idx: int,
    cause_idx: int,
    offdiag_interval_days: dict[tuple[int, int], float] | None,
) -> float | None:
    interval = (offdiag_interval_days or {}).get((effect_idx, cause_idx))
    if interval is None:
        return None
    interval = float(interval)
    if interval <= 0:
        return None
    return interval


def _transition_from_elementwise_dt_terms(
    drift: np.ndarray,
    interval_days: float,
) -> np.ndarray:
    transition = np.eye(drift.shape[0], dtype=float)
    for idx in range(drift.shape[0]):
        transition[idx, idx] = math.exp(float(drift[idx, idx]) * interval_days)

    offdiag_support = ~np.eye(drift.shape[0], dtype=bool)
    transition[offdiag_support] = drift[offdiag_support] * interval_days
    return transition


def matrix_log_diagnostic_drift(
    drift: np.ndarray,
    *,
    interval_days: float,
) -> np.ndarray:
    """Compute the full matrix-log CT drift used by dynamics diagnostics."""
    if interval_days <= 0:
        raise PriorCompilationError(
            ["matrix-log CT dynamics diagnostics require a positive interval."]
        )

    transition = _transition_from_elementwise_dt_terms(drift, interval_days)
    log_transition = scipy.linalg.logm(transition)
    imaginary_scale = float(np.max(np.abs(np.imag(log_transition))))
    if imaginary_scale > _LOGM_IMAG_TOL:
        raise PriorCompilationError(
            [
                "Matrix-log CT dynamics diagnostics require an embeddable real transition matrix; "
                f"max imaginary logm component is {imaginary_scale:.3g}."
            ]
        )
    return np.real(log_transition) / interval_days


def _correlation_prior(prior: dist.Distribution) -> dist.Distribution:
    """Apply the declared correlation domain to Normal and bounded priors."""
    if isinstance(prior, dist.Normal):
        return dist.TruncatedNormal(
            np.asarray(prior.loc), np.asarray(prior.scale), low=-1.0, high=1.0
        )
    if isinstance(prior, dist.TwoSidedTruncatedDistribution):
        low = np.maximum(np.asarray(prior.low), -1.0)
        high = np.minimum(np.asarray(prior.high), 1.0)
        if np.any(low >= high):
            raise PriorCompilationError(
                ["Initial-state correlation prior has no support within [-1, 1]"]
            )
        return dist.TruncatedNormal(
            np.asarray(prior.base_dist.loc), np.asarray(prior.base_dist.scale), low=low, high=high
        )
    if isinstance(prior, dist.Uniform):
        low = np.maximum(np.asarray(prior.low), -1.0)
        high = np.minimum(np.asarray(prior.high), 1.0)
        if np.any(low >= high):
            raise PriorCompilationError(
                ["Initial-state correlation prior has no support within [-1, 1]"]
            )
        return dist.Uniform(low, high)
    return prior


def compile_parameter_law(
    dynamical_model_spec: DynamicalModelSpec,
    parameter: ParameterSpec,
    binding: CompiledParameterBinding,
) -> tuple[dist.Distribution, CompiledEffectInterval | None]:
    """Translate one scalar scientific law into its native numerical coordinates."""
    prior, interval = quantity_parameter_law(dynamical_model_spec, parameter)
    if interval is None:
        return prior, None
    if not isinstance(binding.target, CompiledEdgeTarget):
        raise PriorCompilationError(
            [f"Dynamics effect prior {parameter.id!r} is missing effect/cause metadata"]
        )
    return prior, CompiledEffectInterval(target=binding.target, days=interval)


def quantity_parameter_law(
    dynamical_model_spec: DynamicalModelSpec, parameter: ParameterSpec
) -> tuple[dist.Distribution, float | None]:
    """Resolve a scalar quantity's scale without compiling unrelated model components."""
    prior = dynamical_model_spec.distribution_for(parameter.id)
    if prior is None:
        raise PriorCompilationError(
            [f"Parameter {parameter.id!r} requires an explicit probability law"]
        )
    if prior.batch_shape or prior.event_shape:
        raise PriorCompilationError(
            [
                "Fitting requires independent scalar input laws; shared laws remain intact in "
                "DynamicalModelSpec. Select an input revision supported by the fitting engine."
            ]
        )
    if isinstance(prior, dist.Delta):
        raise PriorCompilationError([f"Prior {parameter.id!r}: {_DEGENERATE_PRIOR_PREAMBLE}"])
    transform = parameter.transform
    if isinstance(transform, PersistenceTransformSpec):
        dt = _resolve_transform_interval_days(transform, dynamical_model_spec)
        return persistence_to_decay(prior, dt), None
    if isinstance(transform, IntervalEffectTransformSpec):
        dt = _resolve_transform_interval_days(transform, dynamical_model_spec)
        return interval_effect_to_rate(prior, dt), dt
    if isinstance(transform, InitialCorrelationTransformSpec):
        prior = _correlation_prior(prior)
    return prior, None


def compile_priors(
    compiled_dynamical_model: CompiledDynamicalModel,
    selection: StructuralSelection,
) -> tuple[
    dict[str, dist.Distribution], tuple[CompiledParameterBinding, ...], list[CompileDiagnostic]
]:
    """Bind the model's native distributions to their declared execution coordinates."""
    from grassmoth.models.parameters import execution_parameters

    authored = selection.dynamical_model_spec
    parameters = {parameter.id: parameter for parameter in execution_parameters(selection)}
    missing = [parameter.id for parameter in parameters.values() if parameter.distribution is None]
    if missing:
        raise PriorCompilationError(
            [f"DynamicalModelSpec parameters require explicit prior distributions: {missing}"]
        )

    active_sites = build_site_registry(compiled_dynamical_model)
    prior_entries: dict[str, dist.Distribution] = {
        site.name: default_prior_for_descriptor(site) for site in active_sites
    }
    site_by_name = {site.name: site for site in active_sites}
    per_site: dict[str, dict[int, dist.Distribution]] = {}

    bindings = compiled_dynamical_model.bindings
    binding_by_parameter = {binding.parameter_id: binding for binding in bindings}
    errors: list[str] = []
    offdiag_interval_days: dict[tuple[int, int], float] = {}

    for param_name, parameter in parameters.items():
        try:
            binding = binding_by_parameter.get(param_name)
            if binding is None:
                errors.append(
                    f"Prior {param_name!r} for {authored.parameter_context(parameter.id).quantity.value!r} could not be structurally bound to the compiled SSM."
                )
                continue

            prior, effect_interval = compile_parameter_law(authored, parameter, binding)
            if effect_interval is not None:
                target = effect_interval.target
                offdiag_interval_days[(target.target_index, target.source_index)] = (
                    effect_interval.days
                )

            site = site_by_name[binding.site.name]
            validate_site_prior(site, prior)
            per_site.setdefault(site.name, {}).update(
                {coordinate.flat_index: prior for coordinate in binding.native_coordinates}
            )
        except AggregatedCompileError as exc:
            errors.extend(exc.errors)
            continue

    if errors:
        raise PriorCompilationError(errors)

    for site_name, entries in per_site.items():
        site = site_by_name[site_name]
        coordinates = [prior_entries[site.name]] * site_size(site.shape)
        for index, prior in entries.items():
            coordinates[index] = prior
        prior_entries[site.name] = batch_prior_distributions(
            coordinates, site.shape, support=site_constraint(site)
        )

    prior_registry = prior_entries

    diagnostics = collect_compile_diagnostics(
        compiled_dynamical_model,
        prior_registry=prior_registry,
        offdiag_interval_days=offdiag_interval_days,
    )

    return prior_registry, bindings, diagnostics


def bind_parameters(
    bindings: CompiledBindingRegistry,
    structure: StructuralSelection,
    parameters: Sequence[ParameterSpec],
    registry: tuple[SiteDescriptor, ...],
) -> tuple[tuple[CompiledParameterBinding, ...], tuple[ParameterCoordinate, ...]]:
    """Compile scientific definitions into explicit scalar execution bindings."""
    from grassmoth.models.ssm.compile.parameter_identity import component_identity

    sites = {site.name: site for site in registry}
    definitions = {parameter.id: parameter for parameter in parameters}
    all_bindings = dict(bindings.by_parameter)

    result = []
    bound_coordinates = set()
    auxiliary = []
    for parameter_id, binding in sorted(all_bindings.items()):
        definition = definitions[parameter_id]
        site = binding.site
        native_coordinates = resolve_site_selection(site, binding.selection)
        elements, coordinates = {}, {}
        for native in native_coordinates:
            coordinate = native.coordinate
            if coordinate in bound_coordinates:
                raise PriorCompilationError(
                    [f"Runtime coordinate {coordinate.label} has multiple scientific owners"]
                )
            bound_coordinates.add(coordinate)
            component = component_identity(definition, native, binding.selection, site, structure)
            if component is None:
                auxiliary.append(coordinate)
                continue
            element_id, label = component
            if element_id in elements:
                raise PriorCompilationError(
                    [f"Parameter {definition.name!r} has duplicate logical components"]
                )
            elements[element_id] = label
            coordinates[element_id] = coordinate
        if not coordinates:
            continue
        result.append(
            CompiledParameterBinding(
                parameter_id=definition.id,
                parameter_name=definition.name,
                coordinates=coordinates,
                elements=elements,
                native_coordinates=native_coordinates,
                site=site,
                target=binding.target,
                transform=definition.transform.kind,
            )
        )
    # Rectangular likelihood tensors contain padded/unused indicator rows. Their
    # explicit execution-only status keeps them out of reported scientific findings.
    for site in sites.values():
        for index in np.ndindex(site.shape):
            coordinate = ParameterCoordinate(site_name=site.name, indices=index)
            if coordinate in bound_coordinates:
                continue
            if site.site_kind not in {SiteKind.OBS_ORDERED_BASE, SiteKind.OBS_ORDERED_GAPS}:
                raise PriorCompilationError(
                    [
                        f"Runtime coordinate {coordinate.label} has no scientific parameter definition"
                    ]
                )
            auxiliary.append(coordinate)
    return tuple(result), tuple(auxiliary)
