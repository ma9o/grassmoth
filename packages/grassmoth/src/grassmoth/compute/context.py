"""Execution context the shell supplies to pure numerical code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable
    from datetime import datetime

    from grassmoth.compute.budget import ComputeBudget
    from grassmoth.compute.programs import ProgramCompiler, ProgramPreparer


@dataclass(frozen=True)
class ModelExecutionContext:
    """Compilation, memory budget, clocks, logging and profiling for one execution."""

    compiler: ProgramCompiler
    prepare_program: ProgramPreparer
    budget: ComputeBudget
    clock: Callable[[], float]
    wall_clock: Callable[[], datetime]
    logger: logging.Logger
    profile: bool
