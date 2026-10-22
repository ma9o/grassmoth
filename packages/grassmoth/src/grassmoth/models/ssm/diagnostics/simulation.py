"""Reusable reductions of nonlinear simulation arrays, independent of authoring recipes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from grassmoth.artifacts.immutability import freeze_fields
from grassmoth.artifacts.model.expressions import (
    LiteralExpression,
    StateExpression,
    expression_states,
    hill_applications,
    restoring_coefficients,
)
from grassmoth.artifacts.reports.checks import NumericCriterionEvidence
from grassmoth.models.ssm import compiled_access as numeric
from grassmoth.models.ssm.compile.expressions import compile_expression
from grassmoth.models.ssm.diagnostics.reachability import (
    C1B_GROWTH_RATIO,
    C1B_MAX_EXPLOSIVE_FRAC,
    CheckResult,
    check_confinement,
    check_coverage,
    check_edge_share,
    check_resolvability,
    check_saturation,
    check_scale,
    check_transmission,
)
from grassmoth.models.ssm.dynamics.spec import DynamicsSpec
from grassmoth.models.ssm.predictive.registry_runtime import predictive_keys
from grassmoth.models.ssm.predictive.statistics import observation_signal_and_variance

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import jax.numpy as jnp

    from grassmoth.artifacts.model.expressions import CoefficientExpression
    from grassmoth.compute.budget import ComputeBudget
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel, CompiledState
    from grassmoth.models.ssm.dynamics.expression import ExpressionComponentSpec
    from grassmoth.models.ssm.predictive.types import PredictiveDraws
    from grassmoth.models.ssm.preparation.observations import ObservationSupportRuntime


@dataclass(frozen=True)
class ConstructSimulationTarget:
    """One construct and its incoming mechanisms measured in an existing simulation."""

    construct: CompiledState
    edge_parents: tuple[str, ...] = ()
    hill_parents: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """Display label of the construct whose simulated behavior is being assessed."""
        return self.construct.name


def _edge_components(compiled_dynamical_model: CompiledDynamicalModel, source: int, target: int):
    """Every additive native term belonging to one causal edge."""
    return [
        (i, component)
        for i, component in enumerate(compiled_dynamical_model.dynamics.spec.components)
        if component.edge_owned and source in component.sources and component.target == target
    ]


@dataclass(frozen=True)
class DesignInfo:
    """Sampling design + observed data needed by the checks, bound once per build.

    Real longitudinal data is irregular and per-indicator, so observations are
    indexed per indicator: ``obs_index_by_indicator`` maps each indicator to the
    ``t_grid`` indices where it was actually observed, and ``values_by_indicator``
    holds the aligned observed values. ``observation_support`` is passed to the
    exact prior-predictive sampler and edge-off re-simulation when present;
    synthetic tests use a shared index across indicators.

    ``c1b_growth_ratio`` / ``c1b_max_explosive_frac`` calibrate C1b confinement: a
    draw explodes when its late-window amplitude exceeds ``c1b_growth_ratio`` times
    its own early amplitude, and the check reds when at least
    ``c1b_max_explosive_frac`` of draws do. The defaults encode the model class's
    confinement commitment (every self-dynamics component is a restoring force);
    they are design calibration, not part of the statistic — an intrinsically
    trending domain raises them here instead of accepting a standing soft-fail.
    """

    t_grid: jnp.ndarray
    manifest_ids: tuple[str, ...]
    obs_index_by_indicator: Mapping[str, np.ndarray]
    values_by_indicator: Mapping[str, np.ndarray]
    n_draws: int = 200
    seed: int = 0
    c1b_growth_ratio: float = C1B_GROWTH_RATIO
    c1b_max_explosive_frac: float = C1B_MAX_EXPLOSIVE_FRAC
    observation_support: ObservationSupportRuntime | None = None

    def __post_init__(self) -> None:
        """Own immutable collection fields describing the simulation-check design."""
        freeze_fields(self)

    @property
    def pooled_obs_index(self) -> np.ndarray:
        """Sorted union of every indicator's observation indices.

        Used by the latent-at-observation checks (edge overwhelm, saturation) that
        are indicator-agnostic — they read the latent path where any data exists.
        """
        if not self.obs_index_by_indicator:
            return np.arange(int(np.asarray(self.t_grid).shape[0]))
        return np.unique(
            np.concatenate([np.asarray(v) for v in self.obs_index_by_indicator.values()])
        )

    def observation_indices_for(self, indicators: tuple[str, ...]) -> np.ndarray:
        """Sorted union of actual observation indices for the requested indicators."""
        indices = [
            np.asarray(self.obs_index_by_indicator[name])
            for name in indicators
            if name in self.obs_index_by_indicator
            and np.asarray(self.obs_index_by_indicator[name]).size > 0
        ]
        if not indices:
            return np.asarray([], dtype=int)
        return np.unique(np.concatenate(indices))


@dataclass(frozen=True)
class _EdgeOffTarget:
    """One compiled edge coordinate to disable under the same predictive draws."""

    components: tuple[int, ...] = ()


def _coefficient_draws(
    operand: CoefficientExpression,
    component: ExpressionComponentSpec,
    pred: PredictiveDraws,
    prefix: str,
) -> np.ndarray:
    value = operand.value
    if value is None:
        raise ValueError("Unassigned coefficients do not have predictive draws")
    if isinstance(value, (int, float)):
        return np.full(pred.n_draws, value)
    sites = dict(component.parameter_sites(prefix))
    return np.asarray(pred.parameters[sites[value].name])


def measure_construct_simulation(
    compiled_dynamical_model: CompiledDynamicalModel,
    pred: PredictiveDraws,
    design: DesignInfo,
    target: ConstructSimulationTarget,
    *,
    dynamics: bool = True,
    measurement: bool = True,
    edge_contrasts: ComputeBudget | None,
) -> list[CheckResult]:
    """Execute selected check groups over a shared predictive batch."""
    results: list[CheckResult] = []
    if dynamics:
        results.extend(
            measure_construct_dynamics(
                compiled_dynamical_model,
                pred,
                design,
                target,
                edge_contrasts=edge_contrasts,
            )
        )
    if measurement:
        results.extend(
            measure_construct_measurement(compiled_dynamical_model, pred, design, target)
        )
    return results


def measure_construct_dynamics(
    compiled_dynamical_model: CompiledDynamicalModel,
    pred: PredictiveDraws,
    design: DesignInfo,
    target: ConstructSimulationTarget,
    *,
    edge_contrasts: ComputeBudget | None,
) -> list[CheckResult]:
    """Measure confinement, scale, resolvability, edge contrasts and saturation."""
    latent_names = numeric.state_names(compiled_dynamical_model)
    d = latent_names.index(target.name)
    x = np.asarray(pred.trajectory.latents[:, :, d])
    times = np.asarray(design.t_grid, dtype=float)
    indicator_names = tuple(
        observation.id
        for observation in compiled_dynamical_model.observations
        if observation.state_index == d
    )
    target_obs = design.observation_indices_for(indicator_names)
    structural_indices = target_obs if target_obs.size else np.arange(times.size)

    results: list[CheckResult] = []

    phase_results = (
        list(
            check_confinement(
                target.name,
                x,
                times,
                growth_ratio=design.c1b_growth_ratio,
                max_explosive_frac=design.c1b_max_explosive_frac,
            )
        )
        if times.size >= 4
        else [
            CheckResult.measured(
                "C1a finiteness",
                target.name,
                f"nonfinite {float(np.mean(~np.isfinite(x))):.1%}",
                "0%",
                "Finite-value scan; confinement requires at least four times.",
                outcome="passed" if bool(np.isfinite(x).all()) else "failed",
                measurements=(
                    NumericCriterionEvidence(
                        criterion="nonfinite_fraction",
                        value=float(np.mean(~np.isfinite(x))),
                        upper=0.0,
                    ),
                ),
            )
        ]
    )
    results.extend(phase_results)
    if times.size < 4:
        results.append(
            CheckResult.unevaluated(
                "C1b confinement",
                target.name,
                "INSUFFICIENT_TIMES",
                "Confinement requires at least four design times.",
            )
        )
    if not np.isfinite(x).all():
        results.extend(
            CheckResult.unevaluated(
                check,
                target.name,
                "NONFINITE_PATHS",
                "The latent paths contain non-finite values.",
            )
            for check in (
                "C2 latent scale",
                "C3 resolvability",
                "C4b edge overwhelm",
                "C4c saturation",
            )
        )
        return results

    results.append(check_scale(target.name, x))

    # C3 resolvability uses the declared stiffness, whether fixed or estimated.
    potentials = [
        (i, comp, operand)
        for i, comp in enumerate(compiled_dynamical_model.dynamics.spec.components)
        if comp.target == d and not comp.edge_owned
        for operand in restoring_coefficients(comp.expression, comp.state_ids[d], kind=comp.kind)
        if operand.role == "decay"
    ]
    if potentials:
        stiffness = sum(
            _coefficient_draws(operand, comp, pred, f"vf_{i}") for i, comp, operand in potentials
        )
        tau = 1.0 / np.asarray(stiffness)
        results.append(check_resolvability(target.name, tau, times[target_obs]))
    else:
        results.append(
            CheckResult.unevaluated(
                "C3 resolvability",
                target.name,
                "NO_RELAXATION_TERM",
                "No declared decay coefficient supplies the timescale screen.",
            )
        )

    # C4b edge overwhelm (edge-off re-simulation holds all else fixed).
    if not edge_contrasts:
        results.extend(
            CheckResult.unevaluated(
                "C4b edge overwhelm",
                f"{parent}->{target.name}",
                "EDGE_CONTRASTS_EXPLICIT",
                "Edge-share analysis requires a separate paired edge-knockout experiment.",
            )
            for parent in target.edge_parents
        )
    contrasts = (
        tuple((parent, edge_contrasts) for parent in target.edge_parents)
        if edge_contrasts is not None
        else ()
    )
    for parent, compute_budget in contrasts:
        edge_target = _incoming_edge_off_target(
            compiled_dynamical_model, replace(target, edge_parents=(parent,)), latent_names, d
        )
        x_off = _resimulate_edge_off(
            compiled_dynamical_model,
            pred,
            design.t_grid,
            edge_target,
            design.seed,
            compute_budget=compute_budget,
        )[:, :, d]
        results.extend(
            check_edge_share(
                f"{parent}->{target.name}",
                x[:, structural_indices],
                np.asarray(x_off)[:, structural_indices],
            )
        )

    # C4c Hill saturation (per saturating parent).
    for parent in target.hill_parents:
        p_idx = latent_names.index(parent)
        applications = [
            (i, comp, source, ec50, exponent)
            for i, comp in _edge_components(compiled_dynamical_model, p_idx, d)
            for source, ec50, exponent in hill_applications(comp.expression)
            if comp.state_ids[p_idx] in expression_states(source)
        ]
        if not applications:
            raise ValueError(f"No Hill expression for {parent!r} -> {target.name!r}")
        for comp_idx, comp, source, ec50, exponent in applications:
            operands, numerical = compile_expression(source)
            parent_vals = numerical(
                0.0,
                *(
                    np.asarray(
                        pred.trajectory.latents[
                            :, structural_indices, comp.state_ids.index(operand.construct_id)
                        ]
                    )
                    if isinstance(operand, StateExpression)
                    else np.asarray(
                        _coefficient_draws(operand, comp, pred, f"vf_{comp_idx}")
                    ).reshape(-1, 1)
                    for operand in operands
                ),
            )
            results.append(
                check_saturation(
                    f"{parent}->{target.name}",
                    _coefficient_draws(ec50, comp, pred, f"vf_{comp_idx}"),
                    _coefficient_draws(exponent, comp, pred, f"vf_{comp_idx}"),
                    np.asarray(parent_vals),
                )
            )

    return results


def measure_construct_measurement(
    compiled_dynamical_model: CompiledDynamicalModel,
    pred: PredictiveDraws,
    design: DesignInfo,
    target: ConstructSimulationTarget,
) -> list[CheckResult]:
    """Measure observation coverage and exact-law temporal transmission."""
    d = numeric.state_names(compiled_dynamical_model).index(target.name)
    results: list[CheckResult] = []
    target_is_time_invariant = compiled_dynamical_model.states[d].time_invariant

    # C5a/C5b coverage for every indicator; C5c transmission only for dynamic constructs.
    for indicator in compiled_dynamical_model.observations:
        if indicator.state_index != d:
            continue
        lik = indicator.law
        var = indicator.id
        observed = np.asarray(design.values_by_indicator[var])
        m = design.manifest_ids.index(var)
        oi = np.asarray(design.obs_index_by_indicator[var])
        if not oi.size:
            results.extend(
                CheckResult.unevaluated(
                    check,
                    var,
                    "NO_OBSERVATION_SUPPORT",
                    "No supported observation times exist for this indicator.",
                )
                for check in ("C5a location reach", "C5b width", "C5c transmission")
            )
            continue
        pp_y = np.asarray(pred.trajectory.observations[:, oi, m])
        if not np.isfinite(pred.trajectory.latents).all() or not np.isfinite(pp_y).all():
            if not np.isfinite(pp_y).all():
                results.append(
                    CheckResult.measured(
                        "C1a finiteness",
                        var,
                        f"nonfinite {float(np.mean(~np.isfinite(pp_y))):.1%}",
                        "0%",
                        "The sampled emission contains non-finite values.",
                        outcome="failed",
                        measurements=(
                            NumericCriterionEvidence(
                                criterion="nonfinite_fraction",
                                value=float(np.mean(~np.isfinite(pp_y))),
                                upper=0.0,
                            ),
                        ),
                    )
                )
            results.extend(
                CheckResult.unevaluated(
                    check,
                    var,
                    "NONFINITE_PATHS",
                    "Dependent predictive paths contain non-finite values.",
                )
                for check in ("C5a location reach", "C5b width", "C5c transmission")
            )
            continue
        level_count = numeric.observation_level_counts(compiled_dynamical_model)[m]
        phase_results = (
            [
                replace(result, target=var)
                for result in check_coverage(
                    indicator.name,
                    pp_y,
                    observed,
                    distribution=lik.family,
                    level_count=level_count,
                )
            ]
            if observed.size
            else [
                CheckResult.unevaluated(
                    check,
                    var,
                    "NO_OBSERVATIONS",
                    "No comparison observations exist for this indicator.",
                )
                for check in ("C5a location reach", "C5b width")
            ]
        )
        if not target_is_time_invariant:
            signal, conditional_variance = observation_signal_and_variance(
                compiled_dynamical_model,
                pred,
                m,
                oi,
                observation_support=design.observation_support,
            )
            phase_results.append(
                replace(
                    check_transmission(indicator.name, signal, conditional_variance), target=var
                )
            )
        else:
            phase_results.append(
                CheckResult.unevaluated(
                    "C5c transmission",
                    var,
                    "STATIC_CONSTRUCT",
                    "Temporal transmission applies only to time-varying constructs.",
                )
            )
        results.extend(phase_results)

    return results


def _incoming_edge_off_target(
    compiled_dynamical_model: CompiledDynamicalModel,
    contribution: ConstructSimulationTarget,
    latent_names: Sequence[str],
    target: int,
) -> _EdgeOffTarget:
    """Compiled vector-field components for incoming edges."""
    components: set[int] = set()
    for parent in contribution.edge_parents:
        if parent in latent_names:
            p_idx = latent_names.index(parent)
            components.update(
                index for index, _ in _edge_components(compiled_dynamical_model, p_idx, target)
            )
    if contribution.edge_parents and not components:
        raise ValueError(
            "Could not resolve an edge-off coordinate for incoming parents "
            f"{list(contribution.edge_parents)!r}"
        )
    return _EdgeOffTarget(tuple(sorted(components)))


def _resimulate_edge_off(
    compiled_dynamical_model: CompiledDynamicalModel,
    pred: PredictiveDraws,
    t_grid: jnp.ndarray,
    edge_target: _EdgeOffTarget,
    seed: int,
    *,
    compute_budget: ComputeBudget,
) -> jnp.ndarray:
    """Re-simulate the latents with the given edge contributions zeroed, all else fixed.

    Reuses the exact param draws (and Brownian path via ``seed``) from ``pred`` so
    the only difference from the edge-on trajectory is the zeroed edge — the
    same-noise contrast :func:`reachability.check_edge_share` expects.
    """
    from grassmoth.models.ssm.predictive.registry_runtime import (
        _simulate_vector_field_predictive_latents,
    )

    samples = pred.parameters
    natural = compiled_dynamical_model.dynamics.spec.components
    intervention_dynamics = DynamicsSpec(
        n_latent=numeric.n_states(compiled_dynamical_model),
        components=tuple(
            replace(component, expression=LiteralExpression(value=0))
            if index in edge_target.components
            else component
            for index, component in enumerate(natural)
        ),
    )
    latents, _linear_predictors = _simulate_vector_field_predictive_latents(
        compiled_dynamical_model,
        samples,
        t_grid,
        rng_key=predictive_keys(seed).latents,
        dynamics=intervention_dynamics,
        compute_budget=compute_budget,
    )
    return latents
