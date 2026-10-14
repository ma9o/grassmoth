"""Scientific prior evidence and validation findings."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value

type PriorFailureStage = Literal[
    "compiled_parameters",
    "latent_dynamics",
    "observation_mean",
    "observation_sample",
    "support_violation",
    "model_build",
    "prior_sampling",
    "unknown",
]


class PriorValidationResult(Value):
    """Non-fatal compiler warning about prior laws."""

    parameter: str = Field(description="Name of the parameter that was validated")
    code: str = Field(default="unspecified")
    issue: str | None = None
    suggested_adjustment: str | None = None
    related_parameters: tuple[str, ...] = Field(default_factory=tuple)
    compiled_site_name: str | None = None
    compiled_flat_index: int | None = None
    failure_stage: PriorFailureStage | None = None

    def with_parameter_provenance(self, parameters: tuple[str, ...]) -> Self:
        """Attach resolved writer identities without mutating a diagnostic."""
        return self.revised(related_parameters=parameters)


__all__ = [
    "PriorFailureStage",
    "PriorValidationResult",
]
