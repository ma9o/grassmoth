"""Aligned numerical batches produced by exact predictive simulation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import jax

from grassmoth.artifacts.immutability import freeze_fields

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PredictiveTrajectory:
    """One aligned draw/time grid with state and observation channel axes."""

    latents: jax.Array
    linear_predictors: jax.Array
    observations: jax.Array
    observations_mask: jax.Array
    expected_observations: jax.Array

    def __post_init__(self) -> None:
        """Require aligned draw/time axes and a boolean observation mask matching the generated values."""
        if self.latents.ndim != 3 or self.observations.ndim != 3:
            raise ValueError("Predictive trajectories require draw, time, and state/channel axes")
        if self.latents.shape[:2] != self.observations.shape[:2]:
            raise ValueError("Predictive states and observations must share draw and time axes")
        for name, values in (
            ("linear_predictors", self.linear_predictors),
            ("observations_mask", self.observations_mask),
            ("expected_observations", self.expected_observations),
        ):
            if values.shape != self.observations.shape:
                raise ValueError(f"{name} must match the predictive observation axes")
        if self.observations_mask.dtype != bool:
            raise ValueError("Predictive observation masks must be boolean")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PredictiveDraws:
    """Parameter draws and paired trajectories.

    Numerical site names stay inside ``parameters``. Emissions cannot accidentally become
    parameter draws at persistence or resimulation boundaries.
    """

    parameters: Mapping[str, jax.Array | NDArray]
    trajectory: PredictiveTrajectory
    reference: PredictiveTrajectory | None = None

    def __post_init__(self) -> None:
        """Freeze parameter draws and require reference trajectories to share the predictive axes."""
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        for name, values in self.parameters.items():
            if values.ndim < 1 or values.shape[0] != self.n_draws:
                raise ValueError(f"{name} must share the predictive draw axis")
        if self.reference is not None and (
            self.reference.latents.shape != self.trajectory.latents.shape
            or self.reference.observations.shape != self.trajectory.observations.shape
        ):
            raise ValueError("Reference trajectories must match the predictive axes")
        freeze_fields(self)

    @property
    def n_draws(self) -> int:
        """Number of generated trajectories on the leading predictive draw axis."""
        return self.trajectory.latents.shape[0]
