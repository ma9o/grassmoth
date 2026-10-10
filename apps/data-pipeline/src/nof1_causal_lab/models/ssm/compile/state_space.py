"""Compile the independent coordinates of a possibly degenerate state process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax

    from nof1_causal_lab.models.ssm.structure.blocks import DiffusionBlockSpec, T0CholBlockSpec


@dataclass(frozen=True)
class StateSpace:
    """Gaussian density coordinates; the remaining coordinates are exact recurrences."""

    size: int
    inputs: tuple[int, ...]
    initial: tuple[int, ...]
    transition: tuple[int, ...]

    @property
    def requires_replay(self) -> bool:
        """Whether endogenous coordinates must be reconstructed along a whole path."""
        endogenous = self.size - len(self.inputs)
        return len(self.initial) != endogenous or len(self.transition) != endogenous


def compile_state_space(
    diffusion: DiffusionBlockSpec,
    initial: T0CholBlockSpec,
    static_loadings: jax.Array,
    inputs: tuple[int, ...],
) -> StateSpace:
    """Resolve support from declared fixed values and free coefficient positions once."""
    initial_random = np.any(np.asarray(initial.template) != 0, axis=1) | np.asarray(
        initial.diag_support
    )
    initial_random |= np.any(np.asarray(static_loadings) != 0, axis=1)
    transition_random = np.any(
        (np.asarray(diffusion.diffusion_chol_template) != 0)
        | np.asarray(diffusion.diffusion_chol_support),
        axis=1,
    )
    return StateSpace(
        size=diffusion.n_latent,
        inputs=inputs,
        initial=tuple(int(i) for i in np.flatnonzero(initial_random) if i not in inputs),
        transition=tuple(int(i) for i in np.flatnonzero(transition_random) if i not in inputs),
    )
