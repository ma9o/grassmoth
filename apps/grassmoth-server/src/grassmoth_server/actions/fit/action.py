"""Supply runtime services to the value-based fit action."""

import logging
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import jax
from jax.stages import Compiled

from grassmoth.public import (
    ContinuationRefusal,
    FitInputs,
    FitOutput,
    MemoryDemand,
    ModelExecutionContext,
    ModelFitFailure,
    PreparedProgram,
    fit,
)
from grassmoth_server.actions.errors import ModelFitError
from grassmoth_server.actions.fit.execution import dump_compiled_analysis
from grassmoth_server.actions.fit.runtime import fit_runtime
from grassmoth_server.bootstrap import initialize_numerical_runtime
from grassmoth_server.compute.compilation_cache import compile_program, prepare_program
from grassmoth_server.compute.resources import read_compute_budget

if TYPE_CHECKING:
    from types import FunctionType

type CacheStatus = Literal["disabled", "not_consulted", "miss", "hit"]


class CompiledArtifact(NamedTuple):
    """The executable and cache observation for one library-named request."""

    executable: Compiled
    persistent_cache: CacheStatus


def run_fit(inputs: FitInputs, *, profile_dir: Path | None = None) -> FitOutput:
    """Supply compilation services, then write profiling artifacts outside fit timing.

    A profile directory turns profiling on.
    """
    initialize_numerical_runtime()
    profile = profile_dir is not None
    if profile_dir is not None:
        profile_dir.mkdir(parents=True)
    captured: dict[str, CompiledArtifact] = {}
    active_events: ContextVar[list[str] | None] = ContextVar("fit_compile_events", default=None)

    def cache_event(event: str, **_metadata: str | int) -> None:
        events = active_events.get()
        if events is not None and event in {
            "/jax/compilation_cache/compile_requests_use_cache",
            "/jax/compilation_cache/cache_hits",
        }:
            events.append(event)

    @contextmanager
    def observe_cache() -> Iterator[list[str]]:
        events: list[str] = []
        token = active_events.set(events)
        try:
            yield events
        finally:
            active_events.reset(token)

    def cache_status(events: Sequence[str]) -> CacheStatus:
        # The final request/hit pair belongs to the requested executable, after
        # any incidental JAX compilation during tracing. No event is not a hit.
        if events:
            return "hit" if events[-1].endswith("/cache_hits") else "miss"
        if (
            not jax.config.jax_enable_compilation_cache
            or jax.config.jax_compilation_cache_dir is None
        ):
            return "disabled"
        return "not_consulted"

    def compile_profiled[Result](
        function: Callable[..., Result], *args: object, **kwargs: object
    ) -> PreparedProgram[Result]:
        with observe_cache() as events:
            program = compile_program(function, *args, **kwargs)
        captured[cast("FunctionType", function).__name__] = CompiledArtifact(
            program.executable, cache_status(events)
        )
        return program

    def prepare_profiled[Result](
        function: Callable[..., Result], *args: object, **kwargs: object
    ) -> tuple[Callable[[], Compiled], tuple[jax.Array, ...], Result, MemoryDemand]:
        name = cast("FunctionType", function).__name__
        compile_function, operands, result_shape, demand = prepare_program(
            function, *args, **kwargs
        )

        def compile_with_profile() -> Compiled:
            with observe_cache() as events:
                executable = compile_function()
            captured[name] = CompiledArtifact(executable, cache_status(events))
            return executable

        return compile_with_profile, operands, result_shape, demand

    logger = logging.getLogger(__name__)
    if profile:
        jax.monitoring.register_event_listener(cache_event)
    try:
        result = fit(
            inputs,
            execution_context=ModelExecutionContext(
                compiler=compile_profiled if profile else compile_program,
                prepare_program=prepare_profiled if profile else prepare_program,
                budget=read_compute_budget(),
                clock=time.monotonic,
                wall_clock=lambda: datetime.now(UTC),
                logger=logger,
                profile=profile,
            ),
            runtime=fit_runtime(inputs.sampler),
        )
    finally:
        if profile:
            jax.monitoring.unregister_event_listener(cache_event)
    if isinstance(result, ContinuationRefusal):
        raise ModelFitError(result.message, diagnostics={"code": result.code})
    if isinstance(result, ModelFitFailure):
        raise ModelFitError(result.error, diagnostics={"duration_seconds": result.duration_seconds})
    if result.profile is not None:
        report = result.profile.revised(
            compilation=tuple(
                entry.revised(persistent_cache=captured[entry.name].persistent_cache)
                for entry in result.profile.compilation
            ),
            profiling_compilation=tuple(
                entry.revised(persistent_cache=captured[entry.name].persistent_cache)
                for entry in result.profile.profiling_compilation
            ),
        )
        result = result.revised(profile=report)
        for entry in (*report.compilation, *report.profiling_compilation):
            logger.info("Profile cache %s: %s", entry.name, entry.persistent_cache)
            dump_compiled_analysis(captured[entry.name].executable, profile_dir, entry.name)
        if profile_dir is not None:
            (profile_dir / "summary.json").write_text(report.model_dump_json(indent=2))
            profile_dir.touch()
    return result
