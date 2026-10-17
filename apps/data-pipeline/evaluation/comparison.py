"""Paired controls and MCSE-aware comparisons of saved production fit evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import arviz_base as az
import numpy as np
from arviz_stats.sampling_diagnostics import mcse

from nof1_causal_lab.artifacts.probability.serialization import empirical_atoms

if TYPE_CHECKING:
    from collections.abc import Mapping

    from evaluation.declarations import Reference, Tolerances
    from nof1_causal_lab.actions.contracts.bodies import FitOutput
    from nof1_causal_lab.artifacts.reports.inference import FitSegment
    from nof1_causal_lab.json_types import JsonValue


def flattened(value: Mapping[str, JsonValue], prefix: str = "") -> dict[str, JsonValue]:
    """Produce leaf control paths, preserving discrete choices and collection order."""
    result: dict[str, JsonValue] = {}
    for key, item in value.items():
        path = prefix + key
        if isinstance(item, dict):
            result.update(flattened(item, path + "."))
        else:
            result[path] = item
    return result


def differing_controls(
    left: Mapping[str, JsonValue], right: Mapping[str, JsonValue], factors: tuple[str, ...]
) -> tuple[str, ...]:
    """Assigned controls must match outside the treatment; adapted values are not inputs."""
    a, b = flattened(left), flattened(right)
    return tuple(
        sorted(
            key
            for key in a.keys() | b.keys()
            if a.get(key) != b.get(key)
            and not any(key == factor or key.startswith(factor + ".") for factor in factors)
        )
    )


def _posterior_draws(result: FitOutput, segment: FitSegment | None = None):
    """Read the core's aligned empirical law and chain axes without reconstructing a model."""
    metadata = result.inference.core.inference_metadata
    model = result.dynamical_model_spec
    if model is None or metadata.distribution is None:
        raise ValueError("Posterior comparisons require retained draws")
    layout = model.law_layouts[metadata.distribution]
    atoms = empirical_atoms(model.distributions[metadata.distribution])
    if segment is not None:
        atoms = atoms.reshape(metadata.num_chains, -1, atoms.shape[-1])[
            :, segment.start : segment.stop
        ].reshape(-1, atoms.shape[-1])
    return layout, atoms


def _empirical_moments(atoms: np.ndarray):
    """A repeated literal has zero variance regardless of floating-point summation error."""
    variable = np.ptp(atoms, axis=0) > 0
    mean = np.where(variable, atoms.mean(axis=0, dtype=np.float64), atoms[0])
    scale = np.where(variable, atoms.std(axis=0, ddof=1, dtype=np.float64), 0.0)
    return variable, mean, scale


def posterior_arrays(result: FitOutput, segment: FitSegment | None = None):
    """Summarize saved joint coordinates and their Monte Carlo uncertainty."""
    layout, atoms = _posterior_draws(result, segment)
    metadata = result.inference.core.inference_metadata
    chains = atoms.reshape(metadata.num_chains, -1, atoms.shape[-1])
    variable, mean, scale = _empirical_moments(atoms)
    errors = np.zeros(atoms.shape[-1])
    if np.any(variable):
        data = az.from_dict({"posterior": {"joint": chains[:, :, variable]}})
        errors[variable] = np.asarray(mcse(data, method="mean")["joint"].values)
    return layout, mean, scale, errors


def qualify_reference(result: FitOutput, rule: Reference) -> None:
    """Require production convergence and adequate reference precision on all saved coordinates."""
    core = result.inference.core
    if core.convergence.status != "passed" or core.inference_diagnostics is None:
        raise ValueError("Reference fit did not pass the production convergence gate")
    diagnostics = core.inference_diagnostics
    for row in (*diagnostics.per_parameter, *(diagnostics.per_latent or ())):
        if row.r_hat is None or row.r_hat >= rule.max_r_hat:
            raise ValueError("Reference fit exceeds the declared reference R-hat limit")
    _, _, scale, error = posterior_arrays(result)
    free = scale > 0
    if not np.all(np.isfinite(error)) or np.any(error[free] / scale[free] > rule.max_mcse_sd):
        raise ValueError("Reference Monte Carlo precision is insufficient")


def posterior_equivalence(
    result: FitOutput,
    reference: FitOutput,
    tolerance: Tolerances,
    segment: FitSegment | None = None,
) -> dict[str, JsonValue]:
    """Compare declared marginal means and SDs; this is not full distribution equivalence."""
    layout, mean, scale, error = posterior_arrays(result, segment)
    other_layout, other_mean, other_scale, other_error = posterior_arrays(reference)
    if layout != other_layout:
        raise ValueError("Reference and arm have different scientific coordinates")
    free = other_scale > 0
    fixed_equal = bool(np.all(mean[~free] == other_mean[~free]) and np.all(scale[~free] == 0))
    uncertainty = tolerance.mcse_multiplier * np.hypot(error[free], other_error[free])
    mean_bound = (np.abs(mean[free] - other_mean[free]) + uncertainty) / other_scale[free]
    sd_error = np.abs(scale[free] / other_scale[free] - 1)
    finite = bool(np.all(np.isfinite(mean_bound)) and np.all(np.isfinite(sd_error)))
    equivalent = (
        finite
        and fixed_equal
        and bool(
            np.all(mean_bound <= tolerance.mean_sd) and np.all(sd_error <= tolerance.sd_relative)
        )
    )
    discrepant = not fixed_equal or bool(
        np.any(
            (np.abs(mean[free] - other_mean[free]) - uncertainty) / other_scale[free]
            > tolerance.mean_sd
        )
    )
    return {
        "summary_equivalent": equivalent,
        "comparison_status": "equivalent"
        if equivalent
        else "discrepant"
        if discrepant
        else "inconclusive",
        "max_mean_bound_sd": float(np.max(mean_bound)) if finite and mean_bound.size else None,
        "max_sd_relative_error": float(np.max(sd_error)) if finite and sd_error.size else None,
        "fixed_coordinates_equal": fixed_equal,
    }


def _start_scores(starts: np.ndarray, reference: np.ndarray) -> dict[str, JsonValue]:
    """Compare every delivered start and its ensemble coverage to reference marginals."""
    if starts.shape[1:] != reference.shape[1:]:
        raise ValueError("Initializer and reference have different scientific coordinates")
    if not np.isfinite(starts).all():
        return {"status": "not scored", "reason": "Non-finite delivered starts"}
    free, mean, scale = _empirical_moments(reference)
    distances = (starts[:, free] - mean[free]) / scale[free]
    spread = np.std(starts[:, free], axis=0) / scale[free]
    coverage = np.mean(
        (reference >= starts.min(axis=0)) & (reference <= starts.max(axis=0)), axis=0
    )
    return {
        "start_rms_reference_sd": np.sqrt(np.mean(distances**2, axis=1)).tolist()
        if np.any(free)
        else [None] * len(starts),
        "start_max_abs_reference_sd": np.max(np.abs(distances), axis=1).tolist()
        if np.any(free)
        else [None] * len(starts),
        "start_fixed_coordinates_equal": np.all(
            starts[:, ~free] == reference[0, ~free], axis=1
        ).tolist(),
        "spread_reference_sd": [
            float(value) if varying else None
            for value, varying in zip(
                np.divide(np.std(starts, axis=0), scale, out=np.zeros_like(scale), where=free),
                free,
                strict=True,
            )
        ],
        "min_spread_reference_sd": float(spread.min()) if spread.size else None,
        "reference_mass_covered": coverage.tolist(),
    }


def initializer_scores(result: FitOutput, reference: FitOutput | None) -> dict[str, JsonValue]:
    """Score delivered public starts against the caller's qualified reference, plus cost."""
    intervals = sorted(
        (phase.start_seconds, phase.end_seconds)
        for phase in result.inference.core.inference_metadata.segments[-1].phases
        if phase.stage in {"preparing_initialization", "initializing_chains"}
    )
    stop, elapsed = 0.0, 0.0
    for left, right in intervals:
        elapsed += max(0.0, right - max(left, stop))
        stop = max(stop, right)
    cost: dict[str, JsonValue] = {
        "initializer_preparation_seconds": elapsed if intervals else None,
    }
    fields = result.inference.evidence.phase_extra_fields
    initial = (
        fields.get("continued_start")
        if "continued_start" in fields
        else fields.get("initialization")
    )
    if initial is None:
        return {**cost, "initializer": "not scored", "initializer_reason": "No delivered starts"}
    if reference is None:
        return {
            **cost,
            "initializer": "not scored",
            "initializer_reason": "No qualified reference",
        }
    layout, atoms = _posterior_draws(reference)
    if layout.constructs and not np.array_equal(initial["time_points"].values, layout.time_points):
        raise ValueError("Initializer and reference have different time grids")
    return {
        **cost,
        "initializer": "scored",
        "initializer_parameters": {
            str(identity): _start_scores(
                initial[str(identity)].values[:, None], atoms[:, column, None]
            )
            for identity, column in layout.parameter_columns.items()
        },
        "initializer_paths": {
            str(identity): _start_scores(initial[str(identity)].values, atoms[:, span])
            for identity, span in layout.trajectory_slices.items()
        },
    }
