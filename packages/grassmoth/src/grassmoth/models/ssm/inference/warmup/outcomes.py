"""Expected scientific initialization failures, carried to the fitting shell."""

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class InitializationFailure:
    """An initializer could not produce a usable finite joint starting package."""

    message: str

    @classmethod
    def unavailable(cls, message: str) -> Self:
        """Own the failure value returned by numerical initialization."""
        return cls(message)
