"""posterior: Bayesian inference and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from grassmoth.artifacts.base import Value
from grassmoth.models.ssm import compiled_access as numeric
from grassmoth.models.ssm.compile.observation_windows import compile_observation_scan
from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure
from grassmoth.models.ssm.preparation.panel import PreparedFit, prepare_fit
from grassmoth.models.ssm.preparation.preflight import ObservationPreflightFailure

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable, Sequence
    from datetime import datetime

    from grassmoth.artifacts.continuation import SamplerCheckpoint
    from grassmoth.artifacts.observations.dataset import ObservationDataset
    from grassmoth.artifacts.sampling import SamplerSpec
    from grassmoth.compute.budget import ComputeBudget
    from grassmoth.compute.programs import ProgramCompiler, ProgramPreparer
    from grassmoth.models.ssm.inference.types import MCMCPosterior, UnretainedRun
    from grassmoth.models.ssm.preparation.binding import BoundPanel
    from grassmoth.models.structure import StructuralSelection


@dataclass(frozen=True)
class FittedModelResult:
    """Successful production inference, including its aligned runtime inputs."""

    duration_seconds: float
    result: MCMCPosterior | UnretainedRun
    panel: BoundPanel

    @classmethod
    def from_run(
        cls, duration_seconds: float, result: MCMCPosterior | UnretainedRun, panel: BoundPanel
    ) -> FittedModelResult:
        """Bind a successful producer's retained buffers and observations."""
        return cls(duration_seconds, result, panel)


class ModelFitFailure(Value):
    """Unavailable inference with no posterior to persist."""

    error: str
    duration_seconds: float


def fit_prepared_model(
    prepared: PreparedFit,
    *,
    sampler: SamplerSpec,
    budget: ComputeBudget,
    clock: Callable[[], float],
    wall_clock: Callable[[], datetime],
    logger: logging.Logger,
    compiler: ProgramCompiler,
    prepare_program: ProgramPreparer,
    continuation: SamplerCheckpoint | None = None,
) -> MCMCPosterior | UnretainedRun | ObservationPreflightFailure | InitializationFailure:
    """Run on resolved sampler choices and the sampler's owned initialization."""
    from grassmoth.models.ssm.inference.fit import fit

    return fit(
        prepared.inputs.prior_runtime_bundle,
        prepared.panel,
        compute_budget=budget,
        sampler=sampler,
        continuation=continuation,
        clock=clock,
        logger=logger,
        wall_clock=wall_clock,
        compiler=compiler,
        prepare_program=prepare_program,
    )


def _fit_elapsed_seconds(start: float, clock: Callable[[], float]) -> float:
    return clock() - start


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
    budget: ComputeBudget,
    clock: Callable[[], float],
    wall_clock: Callable[[], datetime],
    logger: logging.Logger,
    compiler: ProgramCompiler,
    prepare_program: ProgramPreparer,
    continuation: SamplerCheckpoint | None = None,
) -> FittedModelResult | ModelFitFailure:
    """Fit the selected SSM using supplied computation, timing and logging dependencies."""
    logger.info(
        "Fitting model: rows=%d indicators=%d sampler=%s",
        sum(len(series.rows) for series in data_for_model.series.values()),
        len(data_for_model.series),
        sampler.algorithm.method,
    )
    t0 = clock()

    from grassmoth.models.ssm.compile.inputs import compile_ssm_inputs_from_model

    inputs = compile_ssm_inputs_from_model(selection)
    prep_t0 = clock()
    prepared = prepare_fit(inputs, data_for_model, time_origin=time_origin)
    if not isinstance(prepared, PreparedFit):
        return ModelFitFailure(
            error=prepared.message, duration_seconds=_fit_elapsed_seconds(t0, clock)
        )
    panel = prepared.panel
    observed_cells, total_cells = _observed_cell_counts(panel.observations)
    logger.info(
        "Prepared runtime in %.1fs: wide_rows=%d timepoints=%d manifest_vars=%d "
        "observed_cells=%d/%d time_span_days=%.2f support=%s",
        _fit_elapsed_seconds(prep_t0, clock),
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

    # Fit the model and retain the exact joint posterior.
    logger.info("Starting inference kernel...")
    fit_t0 = clock()
    result = fit_prepared_model(
        prepared,
        sampler=sampler,
        continuation=continuation,
        budget=budget,
        clock=clock,
        wall_clock=wall_clock,
        logger=logger,
        compiler=compiler,
        prepare_program=prepare_program,
    )
    if isinstance(result, (ObservationPreflightFailure, InitializationFailure)):
        return ModelFitFailure(
            error=result.message, duration_seconds=_fit_elapsed_seconds(t0, clock)
        )
    logger.info(
        "Inference complete in %.1fs: wide_rows=%d manifest_vars=%d",
        _fit_elapsed_seconds(fit_t0, clock),
        len(panel.times),
        len(numeric.observation_names(panel.compiled_dynamical_model)),
    )

    logger.info(
        "Retaining native posterior and sampler telemetry: n_samples=%d",
        sampler.num_chains * sampler.num_samples_per_chain,
    )
    return FittedModelResult(
        duration_seconds=_fit_elapsed_seconds(t0, clock), result=result, panel=panel
    )
