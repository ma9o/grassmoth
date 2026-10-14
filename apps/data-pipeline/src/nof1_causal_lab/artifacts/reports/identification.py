"""Identification findings about an explicit scientific model and query."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import ConstructId

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec


class IdentifiedTreatmentStatus(Value):
    """Details on how a treatment effect is identified."""

    status: Literal["identified"] = "identified"
    estimand: str = Field(description="Nonparametric estimand returned by do-calculus")
    marginalized_confounders: tuple[ConstructId, ...] = Field(
        default_factory=tuple,
        description="Unobserved confounders the estimand integrates out",
    )
    instruments: tuple[ConstructId, ...] = Field(
        default_factory=tuple,
        description="Instrument constructs appearing in the nonparametric identification argument",
    )


class NonIdentifiableTreatmentStatus(Value):
    """Context on why a treatment effect is not identifiable."""

    status: Literal["not_identified"] = "not_identified"
    confounders: tuple[ConstructId, ...] = Field(
        default_factory=tuple,
        description="Unobserved constructs blocking identification",
    )
    notes: str | None = Field(
        default=None,
        description="Optional explanation if confounders cannot be enumerated",
    )


class IdentificationReport(Value):
    """Positive and negative causal identification findings for the study question's outcome."""

    outcome: ConstructId | None
    treatments: Mapping[
        ConstructId,
        Annotated[
            IdentifiedTreatmentStatus | NonIdentifiableTreatmentStatus,
            Field(discriminator="status"),
        ],
    ] = Field(
        default_factory=dict,
        description="One tagged identification result per treatment, including its supporting evidence",
    )

    def validate_model(self, dynamical_model_spec: DynamicalModelSpec) -> None:
        """Reject identification findings that refer to constructs absent from the supplied model."""
        owners = set(self.treatments)
        if self.outcome is not None:
            owners.add(self.outcome)
        for finding in self.treatments.values():
            if finding.status == "identified":
                owners.update(finding.marginalized_confounders)
                owners.update(finding.instruments)
            else:
                owners.update(finding.confounders)
        unknown = owners - {construct.id for construct in dynamical_model_spec.constructs}
        if unknown:
            raise ValueError(f"Identification references unknown construct IDs: {sorted(unknown)}")

    @property
    def estimable_treatments(self) -> tuple[ConstructId, ...]:
        """Treatment construct IDs for which the report established identification."""
        return tuple(
            (
                identity
                for identity, finding in self.treatments.items()
                if finding.status == "identified"
            )
        )

    @property
    def non_identifiable(self) -> Mapping[ConstructId, NonIdentifiableTreatmentStatus]:
        """Unidentified treatment findings keyed by their treatment construct IDs."""
        return {
            identity: finding
            for identity, finding in self.treatments.items()
            if finding.status == "not_identified"
        }
