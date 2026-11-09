"""Local JAX compilation and resource budgets for numerical library tests."""

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from types import MappingProxyType
from typing import cast

import jax
from jax.stages import Compiled

from grassmoth.compute.batching import compiled_memory
from grassmoth.compute.budget import ComputeBudget, MemoryDemand, SharedMemoryBudget
from grassmoth.compute.context import ModelExecutionContext
from grassmoth.compute.programs import PreparedProgram, ProgramShapes

COMPUTE_BUDGET = ComputeBudget(
    memory=SharedMemoryBudget(available_bytes=256 * 1024**2),
)


def compile_program[**P, Result](
    function: Callable[P, Result],
    *args: P.args,
    **kwargs: P.kwargs,
) -> PreparedProgram[P, Result]:
    """Supply a direct JAX compiler explicitly to numerical library tests."""
    executable = jax.jit(function).lower(*args, **kwargs).compile()
    return PreparedProgram(executable, cast("Callable[P, Result]", executable))


def prepare_program(
    function: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> tuple[Callable[[], Compiled], tuple[jax.Array, ...], ProgramShapes, MemoryDemand]:
    """Supply the sampler with native shapes and actual compiled buffer estimates."""

    def bound(_operands: tuple[jax.Array, ...], *inputs: object, **options: object) -> object:
        return function(*inputs, **options)

    executable = jax.jit(bound).lower((), *args, **kwargs).compile()
    shapes = MappingProxyType(
        dict(jax.tree_util.tree_leaves_with_path(jax.eval_shape(function, *args, **kwargs)))
    )
    return lambda: executable, (), shapes, compiled_memory(executable)


EXECUTION_CONTEXT = ModelExecutionContext(
    compiler=compile_program,
    prepare_program=prepare_program,
    budget=COMPUTE_BUDGET,
    clock=time.monotonic,
    wall_clock=lambda: datetime.now(UTC),
    logger=logging.getLogger("grassmoth.tests"),
    profile=False,
)
