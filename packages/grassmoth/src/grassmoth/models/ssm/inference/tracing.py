"""Typed, immutable observations of inference transitions, returned only when requested."""

from dataclasses import dataclass
from typing import final

import jax


@jax.tree_util.register_dataclass
@final
@dataclass(frozen=True)
class TracedResult[Result, Trace]:
    """An ordinary numerical result paired with the trace of the same execution."""

    result: Result
    trace: Trace


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CategoricalTraceRecord:
    """Normalized log weights (candidate,) and the scalar index actually selected."""

    log_weights: jax.Array  # noqa: F841, RUF100 - Notebook consumers are outside Vulture's source-only pass.
    selected: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SmootherTraceRecord:
    """One DSMC transition, with axes time, particle and state coordinate.

    Auxiliary points and centers have shape (time, state); particles have
    (time, particle, state). Root selection weights flatten the
    candidates at the root; selected particle indices have shape (time,).
    """

    auxiliary: jax.Array
    proposal_centers: jax.Array  # noqa: F841, RUF100 - Notebook consumers are outside Vulture's source-only pass.
    particles: jax.Array
    root_selection: CategoricalTraceRecord  # noqa: F841, RUF100 - Notebook consumers are outside Vulture's source-only pass.
    selected_particles: jax.Array  # noqa: F841, RUF100 - Notebook consumers are outside Vulture's source-only pass.


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ParameterProposalTraceRecord:
    """Parameter candidates (label, parameter), auxiliary (parameter,), and label weights."""

    candidates: jax.Array
    auxiliary: jax.Array
    label_log_prior: jax.Array  # noqa: F841, RUF100 - Notebook consumers are outside Vulture's source-only pass.


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MarginalParticleTraceRecord:
    """A joint transition composed of its parameter proposal, smoother and final label draw."""

    parameters: ParameterProposalTraceRecord
    smoother: SmootherTraceRecord
    label_selection: CategoricalTraceRecord  # noqa: F841, RUF100 - Notebook consumers are outside Vulture's source-only pass.
