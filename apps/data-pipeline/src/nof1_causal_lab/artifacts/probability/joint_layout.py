"""Scientific column order shared by joint-law identity, storage and sampling."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import AwareDatetime, Field, FiniteFloat, model_validator

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import (
    ConstructId,
    DistributionId,
    ParameterElementId,
    ParameterId,
    scientific_id,
)

if TYPE_CHECKING:
    import jax


class JointLawLayout(Value):
    """An explicit law membership ordered independently of native tensor positions.

    Parameters sort by identity, then by scientific element identity. Trajectories
    follow in construct-identity order, retaining the supplied time-point order.
    Element identities include their categorical or covariance basis.
    """

    parameters: tuple[tuple[ParameterId, tuple[ParameterElementId, ...]], ...]
    constructs: tuple[ConstructId, ...]
    time_points: tuple[FiniteFloat, ...]
    time_origin: AwareDatetime | Literal["relative"] = Field(
        default="relative",
        description="Calendar instant of model day zero, or relative coordinates bound at execution. Fitting retains its calendar origin with the law.",
    )
    labels: Mapping[ParameterElementId, str]

    @model_validator(mode="after")
    def own_coordinates(self) -> JointLawLayout:
        """Require unique identity-ordered parameter coordinates and aligned trajectory labels and times."""
        parameters = tuple(identity for identity, _ in self.parameters)
        elements = tuple(element for _, members in self.parameters for element in members)
        if parameters != tuple(sorted(set(parameters))) or any(
            not members or members != tuple(sorted(set(members))) for _, members in self.parameters
        ):
            raise ValueError(
                "Joint parameters and their elements must be uniquely identity-ordered"
            )
        if len(elements) != len(set(elements)) or set(self.labels) != set(elements):
            raise ValueError(
                "Joint labels must name every scientific parameter element exactly once"
            )
        if self.constructs != tuple(sorted(set(self.constructs))):
            raise ValueError("Joint constructs must be uniquely identity-ordered")
        if bool(self.constructs) != bool(self.time_points) or any(
            right <= left
            for left, right in zip(self.time_points, self.time_points[1:], strict=False)
        ):
            raise ValueError("Trajectory coordinates require their strictly increasing time grid")
        if not self.constructs and self.time_origin != "relative":
            raise ValueError("Only trajectory coordinates have a calendar origin")
        return self

    @property
    def parameter_columns(self) -> Mapping[ParameterElementId, int]:
        """Scientific parameter-element IDs mapped to their columns in the flattened joint law."""
        return MappingProxyType(
            {
                identity: index
                for index, identity in enumerate(
                    element for _, elements in self.parameters for element in elements
                )
            }
        )

    @property
    def trajectory_slices(self) -> Mapping[ConstructId, slice]:
        """Contiguous time-series slices for each construct, following the parameter columns."""
        offset = len(self.parameter_columns)
        steps = len(self.time_points)
        return MappingProxyType(
            {
                identity: slice(offset + index * steps, offset + (index + 1) * steps)
                for index, identity in enumerate(self.constructs)
            }
        )

    @property
    def width(self) -> int:
        """Total number of scalar parameter and trajectory coordinates in the joint law."""
        return len(self.parameter_columns) + len(self.constructs) * len(self.time_points)

    @property
    def distribution_id(self) -> DistributionId:
        """Content-derived distribution identity determined by scientific coordinates and time points."""
        return scientific_id(
            "distribution",
            [
                [[identity, list(elements)] for identity, elements in self.parameters],
                [[identity, list(self.time_points)] for identity in self.constructs],
                self.time_origin
                if isinstance(self.time_origin, str)
                else self.time_origin.isoformat(),
            ],
        )

    def pack(
        self,
        parameters: Mapping[ParameterElementId, np.ndarray | jax.Array],
        trajectories: Mapping[ConstructId, np.ndarray | jax.Array],
    ) -> np.ndarray:
        """Pack aligned scientific draws into a host array for persistence."""
        columns = [np.asarray(parameters[identity])[:, None] for identity in self.parameter_columns]
        columns.extend(np.asarray(trajectories[identity]) for identity in self.trajectory_slices)
        return np.concatenate(columns, axis=1)

    def unpack(
        self, draws: jax.Array
    ) -> tuple[dict[ParameterElementId, jax.Array], dict[ConstructId, jax.Array]]:
        """Read aligned scientific quantities from a draw-by-event runtime array."""
        if draws.ndim != 2 or draws.shape[1] != self.width:
            raise ValueError(
                "Joint event coordinates do not match the model's scientific quantities"
            )
        return (
            {identity: draws[:, column] for identity, column in self.parameter_columns.items()},
            {identity: draws[:, span] for identity, span in self.trajectory_slices.items()},
        )
