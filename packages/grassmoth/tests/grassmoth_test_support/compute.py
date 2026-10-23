"""Local JAX compilation and resource budgets for numerical library tests."""

from collections.abc import Callable
from typing import cast

import jax
from jax.stages import Compiled

from grassmoth.compute.batching import compiled_memory
from grassmoth.compute.budget import ComputeBudget, MemoryDemand, SharedMemoryBudget
from grassmoth.compute.programs import PreparedProgram, compile_jit

COMPUTE_BUDGET = ComputeBudget(
    memory=SharedMemoryBudget(available_bytes=256 * 1024**2),
)


def compile_program[Result](
    function: Callable[..., Result],
    *args: object,
    **kwargs: object,
) -> PreparedProgram[Result]:
    """Compile directly using the library's local JAX compiler."""
    return compile_jit(function, *args, **kwargs)


def prepare_program[Result](
    function: Callable[..., Result],
    *args: object,
    **kwargs: object,
) -> tuple[Callable[[], Compiled], tuple[jax.Array, ...], Result, MemoryDemand]:
    """Supply the sampler with native shapes and actual compiled buffer estimates."""

    def bound(_operands: tuple[jax.Array, ...], *inputs: object, **options: object) -> Result:
        return function(*inputs, **options)

    executable = jax.jit(bound).lower((), *args, **kwargs).compile()
    shapes = cast("Result", jax.eval_shape(function, *args, **kwargs))
    return lambda: executable, (), shapes, compiled_memory(executable)
