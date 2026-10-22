"""Conditioning provenance of current model laws."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, computed_field

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.observations.reference import DataRef


class AuthoredLawProvenance(Value):
    """The current laws have authored ancestry without retained fitting."""

    kind: Literal["authored"] = "authored"

    @computed_field
    @property
    def interpretation(self) -> Literal["prior_predictive"]:
        """Prior-predictive interpretation of draws made entirely from authored parameter laws."""
        return "prior_predictive"


class FittedLawProvenance(Value):
    """All current laws retain one committed fit's model and observation panel."""

    kind: Literal["fitted"] = "fitted"
    fitted_data: DataRef[GitOid, int]
    fitted_model_revision: GitOid
    interpretation: Literal[
        "in_sample_posterior_predictive",
        "posterior_predictive",
    ]


class MixedLawProvenance(Value):
    """Some laws retain a committed fit and others have different ancestry."""

    kind: Literal["mixed"] = "mixed"
    fitted_data: DataRef[GitOid, int]
    fitted_model_revision: GitOid

    @computed_field
    @property
    def interpretation(self) -> Literal["mixed"]:
        """Mixed interpretation of draws whose parameter laws combine different provenance."""
        return "mixed"


class UnknownLawProvenance(Value):
    """Imported laws do not establish a conditioning history."""

    kind: Literal["unknown"] = "unknown"

    @computed_field
    @property
    def interpretation(self) -> Literal["unknown"]:
        """Unknown predictive interpretation when parameter-law provenance is unavailable."""
        return "unknown"


type PredictiveLawProvenance = Annotated[
    AuthoredLawProvenance | FittedLawProvenance | MixedLawProvenance | UnknownLawProvenance,
    Field(discriminator="kind"),
]
