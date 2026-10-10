"""posterior: Bayesian inference and diagnostics."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from typing_extensions import TypedDict

from nof1_causal_lab.compute.resources import read_compute_budget
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.observation_windows import compile_observation_scan
from nof1_causal_lab.models.ssm.inference import ParticleMCMCPosterior
from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure
from nof1_causal_lab.models.ssm.runtime import BoundPanel, PreparedFit, prepare_fit
from nof1_causal_lab.sampler_config import SamplerSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from jax.stages import Compiled

    from nof1_causal_lab.artifacts.observation_data import ObservationDataset
    from nof1_causal_lab.artifacts.posterior import FitSettingsSpec
    from nof1_causal_lab.models.model_structure import StructuralSelection

logger = logging.getLogger(__name__)


class FittedModelResult(TypedDict, closed=True):
    """Successful production inference, including its aligned runtime inputs."""

    fitted: Literal[True]
    duration_seconds: float
    result: ParticleMCMCPosterior
    panel: BoundPanel


class ModelFitFailure(TypedDict, closed=True):
    """Unavailable inference with no posterior to persist."""

    fitted: Literal[False]
    error: str
    duration_seconds: float


def resolve_sampler_spec(settings: FitSettingsSpec) -> SamplerSpec:
    """Resolve public overrides once, before numerical preparation or execution."""
    from nof1_causal_lab.utils.config import get_config

    configured = get_config().inference.sampler
    return SamplerSpec(
        num_warmup=configured.num_warmup if settings.num_warmup is None else settings.num_warmup,
        num_samples_per_chain=configured.num_samples_per_chain
        if settings.num_samples_per_chain is None
        else settings.num_samples_per_chain,
        num_chains=configured.num_chains if settings.num_chains is None else settings.num_chains,
        seed=configured.seed if settings.seed is None else settings.seed,
        num_particles=configured.num_particles
        if settings.num_particles is None
        else settings.num_particles,
        retain_latent_paths=True,
        marginal_particle_gibbs=configured.marginal_particle_gibbs,
    )


def fit_prepared_model(
    prepared: PreparedFit,
    *,
    sampler: SamplerSpec,
) -> ParticleMCMCPosterior | ObservationPreflightFailure | InitializationFailure:
    """Run on resolved sampler choices and the sampler's owned initialization."""
    from nof1_causal_lab.models.ssm.inference import fit

    profile_dir = resolve_profile_dir(None)
    start_trace(profile_dir, label="fit")
    try:
        result = fit(
            prepared.inputs.prior_runtime_bundle,
            prepared.panel,
            compute_budget=read_compute_budget(),
            sampler=sampler,
            clock=time.monotonic,
        )
    finally:
        stop_trace(profile_dir)
    if isinstance(result, (ObservationPreflightFailure, InitializationFailure)):
        return result
    if profile_dir is not None:
        compiled_step = result.diagnostics.compiled_step
        if compiled_step is None:
            raise RuntimeError("Profiling requested but inference did not retain its compiled step")
        dump_compiled_analysis(compiled_step, profile_dir=profile_dir, label="run_batched_step")
    return result


def _fit_elapsed_seconds(start: float) -> float:
    return time.monotonic() - start


def _format_name_preview(names: Sequence[str], limit: int = 4) -> str:
    if not names:
        return "none"
    preview = ", ".join(names[:limit])
    if len(names) > limit:
        preview = f"{preview}, ..."
    return preview


def _observed_cell_counts(observations: jnp.ndarray) -> tuple[int, int]:
    total_cells = int(observations.size)
    observed_cells = int(jnp.sum(~jnp.isnan(observations)).item())
    return observed_cells, total_cells


def _time_span_days(times: jnp.ndarray) -> float:
    if times.size <= 1:
        return 0.0
    return float((times[-1] - times[0]).item())


def _support_summary(panel: BoundPanel) -> str:
    support = panel.observation_support
    if not support.requires_interval_summary_handling:
        return "point-only"
    return (
        f"interval({len(support.interval_summary_manifest_names)}: "
        f"{_format_name_preview(support.interval_summary_manifest_names)}) "
        f"max_active_windows={compile_observation_scan(support.window_plan).max_active_windows}"
    )


def fit_model(
    selection: StructuralSelection,
    data_for_model: ObservationDataset,
    *,
    time_origin: datetime,
    sampler: SamplerSpec,
) -> FittedModelResult | ModelFitFailure:
    """Fit the selected SSM to one observation history with the production particle sampler.

    Args:
        selection: Pinned scientific model scoped to the study question's outcome.
        data_for_model: Recorded observations and their measurement definitions.
        time_origin: Calendar instant of model day zero, or ``None`` for a
            calendar-free history.
        sampler: Fully resolved initialization, sampling, and diagnostic controls.

    Returns:
        Fitted model and retained inference evidence on success, or a fit-failure
        payload explaining an unsupported compilation or observation preflight.
    """
    logger.info(
        "Fitting model: rows=%d indicators=%d sampler=%s",
        sum(len(series.rows) for series in data_for_model.series.values()),
        len(data_for_model.series),
        "marginal_particle_gibbs",
    )
    t0 = time.monotonic()

    from nof1_causal_lab.models.ssm.compile.inputs import compile_ssm_inputs_from_model

    inputs = compile_ssm_inputs_from_model(selection)
    prep_t0 = time.monotonic()
    prepared = prepare_fit(inputs, data_for_model, time_origin=time_origin)
    if not isinstance(prepared, PreparedFit):
        return {
            "fitted": False,
            "error": prepared.message,
            "duration_seconds": _fit_elapsed_seconds(t0),
        }
    panel = prepared.panel
    observed_cells, total_cells = _observed_cell_counts(panel.observations)
    logger.info(
        "Prepared runtime in %.1fs: wide_rows=%d timepoints=%d manifest_vars=%d "
        "observed_cells=%d/%d time_span_days=%.2f support=%s",
        _fit_elapsed_seconds(prep_t0),
        len(panel.times),
        len(panel.times),
        len(numeric.observation_names(panel.compiled_dynamical_model)),
        observed_cells,
        total_cells,
        _time_span_days(panel.times),
        _support_summary(panel),
    )
    logger.info(
        "Manifest order: %s",
        _format_name_preview(numeric.observation_names(panel.compiled_dynamical_model), limit=6),
    )

    # Fit the model — returns a production particle posterior.
    logger.info("Starting inference kernel...")
    fit_t0 = time.monotonic()
    result = fit_prepared_model(prepared, sampler=sampler)
    if isinstance(result, (ObservationPreflightFailure, InitializationFailure)):
        return {
            "fitted": False,
            "error": result.message,
            "duration_seconds": _fit_elapsed_seconds(t0),
        }
    logger.info(
        "Particle inference complete in %.1fs: wide_rows=%d manifest_vars=%d",
        _fit_elapsed_seconds(fit_t0),
        len(panel.times),
        len(numeric.observation_names(panel.compiled_dynamical_model)),
    )

    logger.info(
        "Retaining native posterior and sampler telemetry: n_samples=%d",
        result.draws.describe().n_draws,
    )
    return {
        "fitted": True,
        "duration_seconds": _fit_elapsed_seconds(t0),
        "result": result,
        "panel": panel,
    }


_PROFILE_DIR_ENV = "NOF1_PROFILE_DIR"

# HLO ops whose counts characterize the device access pattern:
#   data movement      — transpose / copy / bitcast / reshape / broadcast /
#                         concatenate / slice
#   indexed access     — gather / scatter / dynamic-slice / dynamic-update-slice
#   tiny linear algebra — dot / triangular-solve / cholesky (the D=2-3 offenders)
#   fused compute      — fusion
#   serialization      — while (the lax.scan forward/backward filters)
_ACCESS_PATTERN_OPS = (
    "fusion",
    "dot",
    "triangular-solve",
    "cholesky",
    "transpose",
    "copy",
    "bitcast",
    "reshape",
    "broadcast",
    "concatenate",
    "slice",
    "gather",
    "scatter",
    "dynamic-slice",
    "dynamic-update-slice",
    "reduce",
    "while",
    "conditional",
    "sort",
    "custom-call",
)


def resolve_profile_dir(profile_dir: str | os.PathLike[str] | None) -> Path | None:
    """Resolve the effective profile directory, or ``None`` when profiling is off.

    ``profile_dir`` takes precedence; otherwise ``NOF1_PROFILE_DIR`` is consulted.
    The directory is created if it does not exist.
    """
    raw = profile_dir if profile_dir is not None else os.environ.get(_PROFILE_DIR_ENV)
    if raw is None or str(raw) == "":
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def start_trace(profile_dir: Path | None, *, label: str) -> None:
    """Begin a ``jax.profiler`` trace into ``profile_dir / label`` (no-op if None)."""
    if profile_dir is None:
        return
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 0
    options.python_tracer_level = 0
    options.include_dataset_ops = False
    options.enable_hlo_proto = False
    jax.profiler.start_trace(str(profile_dir / label), profiler_options=options)


def stop_trace(profile_dir: Path | None) -> None:
    """Stop the active ``jax.profiler`` trace (no-op if None)."""
    if profile_dir is None:
        return
    jax.profiler.stop_trace()


def _summarize_access_patterns(hlo_text: str) -> dict[str, int]:
    """Count producer instructions per op in the optimized HLO.

    Matches ``<op>(`` at a call site (operand references like ``%transpose.5``
    carry no paren, so they are not counted). A heuristic histogram, not a parse.
    """
    return {op: len(re.findall(rf"\b{re.escape(op)}\(", hlo_text)) for op in _ACCESS_PATTERN_OPS}


def dump_compiled_analysis(
    compiled: Compiled,
    profile_dir: Path | None,
    label: str,
) -> None:
    """Persist native producer compilation telemetry at the profiling shell boundary."""
    if profile_dir is None:
        return
    hlo_text = compiled.as_text()
    if hlo_text is None:
        raise RuntimeError("The JAX backend did not expose compiled HLO text")
    (profile_dir / f"{label}.hlo.txt").write_text(hlo_text)
    (profile_dir / f"{label}.cost.json").write_text(
        json.dumps(compiled.cost_analysis(), indent=2, sort_keys=True)
    )
    (profile_dir / f"{label}.access_patterns.json").write_text(
        json.dumps(_summarize_access_patterns(hlo_text), indent=2, sort_keys=True)
    )
