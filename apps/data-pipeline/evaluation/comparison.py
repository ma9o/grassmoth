"""Paired controls and MCSE-aware comparisons of saved production fit evidence."""

from __future__ import annotations

from collections.abc import Mapping

import arviz_base as az
import numpy as np
from arviz_stats.sampling_diagnostics import mcse

from evaluation.declarations import Reference, Tolerances
from nof1_causal_lab.actions.contracts.bodies import FitOutput
from nof1_causal_lab.artifacts.probability.serialization import empirical_atoms
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


def differing_controls(left: Mapping[str, JsonValue], right: Mapping[str, JsonValue], factors: tuple[str, ...]) -> tuple[str, ...]:
    """Assigned controls must match outside the treatment; adapted values are not inputs."""
    a, b = flattened(left), flattened(right)
    return tuple(sorted(key for key in a.keys() | b.keys()
                        if a.get(key) != b.get(key)
                        and not any(key == factor or key.startswith(factor + ".") for factor in factors)))


def posterior_arrays(result: FitOutput):
    """Read the core's aligned empirical law and chain axes without reconstructing a model."""
    metadata = result.inference.core.inference_metadata
    model = result.dynamical_model_spec
    if model is None or metadata.distribution is None:
        raise ValueError("Posterior comparisons require retained draws")
    layout = model.law_layouts[metadata.distribution]
    atoms = empirical_atoms(model.distributions[metadata.distribution])
    chains = atoms.reshape(metadata.num_chains, -1, atoms.shape[-1])
    variable = np.ptp(atoms, axis=0) > 0
    errors = np.zeros(atoms.shape[-1])
    if np.any(variable):
        data = az.from_dict({"posterior": {"joint": chains[:, :, variable]}})
        errors[variable] = np.asarray(mcse(data, method="mean")["joint"].values)
    return layout, atoms.mean(axis=0), atoms.std(axis=0, ddof=1), errors


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


def posterior_equivalence(result: FitOutput, reference: FitOutput, tolerance: Tolerances) -> dict[str, JsonValue]:
    """Compare declared marginal means and SDs; this is not full distribution equivalence."""
    layout, mean, scale, error = posterior_arrays(result)
    other_layout, other_mean, other_scale, other_error = posterior_arrays(reference)
    if layout != other_layout:
        raise ValueError("Reference and arm have different scientific coordinates")
    free = other_scale > 0
    fixed_equal = bool(np.all(mean[~free] == other_mean[~free]) and np.all(scale[~free] == 0))
    uncertainty = tolerance.mcse_multiplier * np.hypot(error[free], other_error[free])
    mean_bound = (np.abs(mean[free] - other_mean[free]) + uncertainty) / other_scale[free]
    sd_error = np.abs(scale[free] / other_scale[free] - 1)
    finite = bool(np.all(np.isfinite(mean_bound)) and np.all(np.isfinite(sd_error)))
    return {
        "summary_equivalent": finite and fixed_equal and bool(np.all(mean_bound <= tolerance.mean_sd) and np.all(sd_error <= tolerance.sd_relative)),
        "max_mean_bound_sd": float(np.max(mean_bound)) if finite and mean_bound.size else None,
        "max_sd_relative_error": float(np.max(sd_error)) if finite and sd_error.size else None,
        "fixed_coordinates_equal": fixed_equal,
    }


def initializer_scores(result: FitOutput) -> dict[str, JsonValue]:
    """Score the initializer's own retained state, separately from posterior performance."""
    initial = result.inference.evidence.phase_extra_fields.get("initialization")
    if initial is None:
        return {"initializer": "not_run"}
    values = np.asarray(initial["log_joint"].values)
    return {
        "initializer_finite_fraction": float(np.isfinite(values).mean()),
        "initializer_mean_log_joint": float(values.mean()) if np.isfinite(values).all() else None,
        "initializer_parameter_spread": float(np.mean(np.std(initial["positions"].values, axis=0))) if initial["positions"].values.shape[-1] else 0.,
        "initializer_state_spread": float(np.mean(np.std(initial["paths"].values, axis=0))),
    }
