"""Supply runtime services to the value-based fit action."""

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime

import jax
from jax.stages import Compiled

from nof1_causal_lab.actions.contracts.bodies import FitOutput
from nof1_causal_lab.actions.errors import ModelFitError
from nof1_causal_lab.actions.fit.action import fit
from nof1_causal_lab.actions.fit.execution import ModelFitFailure
from nof1_causal_lab.artifacts.continuation import SamplerCheckpoint
from nof1_causal_lab.artifacts.model.question import QuestionSpec
from nof1_causal_lab.artifacts.observations.dataset import ObservationDataset
from nof1_causal_lab.artifacts.reports.checks import SpecificationAssessment
from nof1_causal_lab.artifacts.sampling import SamplerSpec
from nof1_causal_lab.compute.budget import MemoryDemand
from nof1_causal_lab.compute.programs import PreparedProgram
from nof1_causal_lab.models.structure import StructuralSelection
from nof1_causal_service.actions.fit.execution import (
    dump_compiled_analysis,
    resolve_profile_dir,
    start_trace,
    stop_trace,
)
from nof1_causal_service.actions.fit.runtime import fit_runtime
from nof1_causal_service.bootstrap import initialize_numerical_runtime
from nof1_causal_service.compute.compilation_cache import compile_staged, prepare_program
from nof1_causal_service.compute.resources import read_compute_budget


def run_fit(
    *,
    question: QuestionSpec,
    selection: StructuralSelection,
    data_for_model: ObservationDataset,
    time_origin: datetime,
    sampler: SamplerSpec,
    preflight: tuple[SpecificationAssessment, ...],
    compute_loo_diagnostics: bool,
    continuation: SamplerCheckpoint | None = None,
    previous: FitOutput | None = None,
) -> FitOutput:
    """Bootstrap local compute and retain tracing outside the scientific calculation."""
    initialize_numerical_runtime()
    profile_dir = resolve_profile_dir(None)

    def compile_profiled[Result](
        function: Callable[..., Result], *args: object, **kwargs: object
    ) -> PreparedProgram[Result]:
        program = compile_staged(function, *args, **kwargs)
        dump_compiled_analysis(program.executable, profile_dir, "run_batched_step")
        return program

    def prepare_profiled[Result](
        function: Callable[..., Result],
        *args: object,
        **kwargs: object,
    ) -> tuple[Callable[[], Compiled], tuple[jax.Array, ...], Result, MemoryDemand]:
        compile_program, operands, result_shape, demand = prepare_program(function, *args, **kwargs)

        def compile_with_profile() -> Compiled:
            executable = compile_program()
            dump_compiled_analysis(executable, profile_dir, "run_batched_step")
            return executable

        return compile_with_profile, operands, result_shape, demand

    start_trace(profile_dir, label="fit")
    try:
        result = fit(
            question=question,
            selection=selection,
            data_for_model=data_for_model,
            time_origin=time_origin,
            sampler=sampler,
            budget=read_compute_budget(),
            runtime=fit_runtime(sampler),
            clock=time.monotonic,
            wall_clock=lambda: datetime.now(UTC),
            logger=logging.getLogger(__name__),
            compiler=compile_profiled,
            prepare_program=prepare_profiled,
            preflight=preflight,
            compute_loo_diagnostics=compute_loo_diagnostics,
            continuation=continuation,
            previous=previous,
        )
    finally:
        stop_trace(profile_dir)
    if isinstance(result, ModelFitFailure):
        raise ModelFitError(result.error, diagnostics={"duration_seconds": result.duration_seconds})
    return result
