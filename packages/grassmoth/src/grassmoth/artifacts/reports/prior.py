"""Scientific prior evidence and validation findings."""

from __future__ import annotations

from typing import ClassVar, Literal, Self

from pydantic import Field

from grassmoth.artifacts.base import Value


class PriorValidationResult(Value):
    """Non-fatal compiler warning about prior laws."""

    code: ClassVar[Literal["dt_ct_approximation_warning"]] = "dt_ct_approximation_warning"
    parameter: str = Field(description="Name of the parameter that was validated")
    issue: str | None = None
    suggested_adjustment: str | None = None
    related_parameters: tuple[str, ...] = Field(default_factory=tuple)
    compiled_site_name: str | None = None
    compiled_flat_index: int | None = None

    def with_parameter_provenance(self, parameters: tuple[str, ...]) -> Self:
        """Attach resolved writer identities without mutating a diagnostic."""
        return self.revised(related_parameters=parameters)


__all__ = [
    "PriorValidationResult",
]
