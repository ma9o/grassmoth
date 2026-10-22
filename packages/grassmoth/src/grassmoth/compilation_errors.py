"""Dependency-neutral aggregated errors for deterministic compilation fronts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class AggregatedCompileError(ValueError):
    """Base class for compilation-stage errors collected across independent checks."""

    header = "Compilation failed"

    def __init__(self, errors: Sequence[str]) -> None:
        """Deduplicate compilation errors in first-seen order and format them as one exception."""
        self.errors = list(dict.fromkeys(errors))
        if len(self.errors) == 1:
            message = self.errors[0]
        else:
            bullets = "\n".join(f"- {error}" for error in self.errors)
            message = f"{self.header}:\n{bullets}"
        super().__init__(message)


class IncompleteModelError(ValueError):
    """A valid partial model is missing choices required by the requested operation."""
