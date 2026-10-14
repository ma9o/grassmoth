"""A model's generative observation definition and construct binding."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.model.likelihood import LikelihoodSpec
from nof1_causal_lab.artifacts.observations.dtypes import VALID_LIKELIHOODS_FOR_DTYPE
from nof1_causal_lab.artifacts.observations.spec import AuthoredObservationSpec


class IndicatorPolarity(StrEnum):
    """Whether the measurement increases or decreases with its owning construct."""

    POSITIVE = "positive"
    NEGATIVE = "negative"  # noqa: V107 - native enum value construction


class IndicatorSpec(Value):
    """Bind an observed-variable ID to a construct and an emission likelihood.

    Extraction instructions belong to DataPreparationSpec. The shared observation
    schema also permits generative models before any observations have been collected.
    """

    observation: AuthoredObservationSpec
    likelihood: LikelihoodSpec | None = None
    construct_polarity: IndicatorPolarity = Field(
        description="Whether higher values move with (positive) or against (negative) the construct."
    )

    @model_validator(mode="after")
    def validate_likelihood(self) -> IndicatorSpec:
        """Reject a likelihood family incompatible with the observation's declared measurement kind."""
        if (
            self.likelihood is not None
            and self.likelihood.law.family
            not in VALID_LIKELIHOODS_FOR_DTYPE[self.observation.measurement_dtype]
        ):
            raise ValueError(
                f"Likelihood {self.likelihood.law.family.value!r} is incompatible with {self.observation.measurement_dtype!r} indicator {self.observation.id!r}"
            )
        return self
