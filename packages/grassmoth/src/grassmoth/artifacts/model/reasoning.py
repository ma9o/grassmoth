"""Authored scientific reasoning and the references its numbered citations select."""

import re
from typing import Self

from pydantic import Field, model_validator

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.text import NonBlankText, ReferenceUrl


class ReasoningSpec(Value):
    """A scientific explanation with a consistent list of numbered references."""

    text: NonBlankText = Field(
        description=(
            "Scientific reasoning. Cite reference_urls with 1-based [n] markers, for example [1]. "
            "Every marker must select a URL and every URL must be cited. "
            "Text without citations needs no URLs."
        )
    )
    reference_urls: tuple[ReferenceUrl, ...] = Field(
        default=(),
        description="URLs cited by [n] in text, ordered by their 1-based citation number.",
    )

    @model_validator(mode="after")
    def parse_citations(self) -> Self:
        """Own the correspondence between citation markers and the reference list."""
        cited = frozenset(int(marker) for marker in re.findall(r"\[([0-9]+)\]", self.text))
        references = frozenset(range(1, len(self.reference_urls) + 1))
        if cited - references:
            raise ValueError("Every [n] citation marker must match a reference URL")
        if references - cited:
            raise ValueError("Every reference URL must be cited by a [n] marker")
        return self
