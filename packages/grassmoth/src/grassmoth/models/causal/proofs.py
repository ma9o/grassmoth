"""Nominal evidence required before emitting numeric causal results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


class CausalCertificationError(Exception):
    """Owned causal inputs do not supply the evidence required for a numeric claim."""


if TYPE_CHECKING:
    from grassmoth.artifacts.identity import GitRef
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth.artifacts.reports.identification import IdentificationReport
    from grassmoth.artifacts.reports.inference import InferenceReportCore


@dataclass(frozen=True, slots=True)
class IdentifiedEstimand:
    """Positive identification evidence for one treatment/outcome estimand."""

    dynamical_model_spec_ref: GitRef
    treatment: str
    outcome: str
    estimand: str


@dataclass(frozen=True, slots=True)
class CertifiedCausalAnalysis:
    """Identification and particle-posterior evidence joined by provenance."""

    dynamical_model_spec: DynamicalModelSpec
    dynamical_model_spec_ref: GitRef
    identification: IdentificationReport
    estimands: tuple[IdentifiedEstimand, ...]
    fitted_dynamical_model_spec: DynamicalModelSpec
    report: InferenceReportCore

    def __post_init__(self) -> None:
        """Certify the retained model, fit, and identification evidence for a common causal outcome."""
        if not self.estimands:
            raise CausalCertificationError("at least one identified estimand is required")
        certify_conditioned_model(
            self.dynamical_model_spec,
            self.fitted_dynamical_model_spec,
            self.report,
        )
        outcomes = {estimand.outcome for estimand in self.estimands}
        if len(outcomes) != 1:
            raise CausalCertificationError("all identified estimands must target the same outcome")
        treatments = [estimand.treatment for estimand in self.estimands]
        if len(treatments) != len(set(treatments)):
            raise CausalCertificationError(
                "identified estimands must not contain duplicate treatments"
            )
        for estimand in self.estimands:
            if estimand.dynamical_model_spec_ref != self.dynamical_model_spec_ref:
                raise CausalCertificationError(
                    "identification evidence and posterior provenance reference different "
                    "model revisions"
                )
            expected = certify_identified_estimand(
                self.dynamical_model_spec,
                self.identification,
                dynamical_model_spec_ref=self.dynamical_model_spec_ref,
                treatment=estimand.treatment,
                outcome=estimand.outcome,
            )
            if expected != estimand:
                raise CausalCertificationError(
                    "identification evidence does not match the model findings"
                )

    @property
    def treatments(self) -> list[str]:
        """Treatment identities in the order of the certified estimands."""
        return [estimand.treatment for estimand in self.estimands]

    @property
    def outcome(self) -> str:
        """The outcome shared by every certified estimand."""
        return self.estimands[0].outcome


def certify_identified_estimand(
    dynamical_model_spec: DynamicalModelSpec,
    identification: IdentificationReport,
    *,
    dynamical_model_spec_ref: GitRef,
    treatment: str,
    outcome: str,
) -> IdentifiedEstimand:
    """Validate and materialize identification evidence for one estimand."""
    identification.validate_model(dynamical_model_spec)
    covered = next(
        (
            construct.name
            for construct in dynamical_model_spec.constructs
            if identification.outcome is not None and construct.id == identification.outcome
        ),
        None,
    )
    if covered is None or outcome != covered:
        raise CausalCertificationError(
            f"{outcome!r} does not match the outcome covered by the model identification"
        )
    construct_ids = {construct.name: construct.id for construct in dynamical_model_spec.constructs}
    if treatment not in construct_ids:
        raise CausalCertificationError(f"{treatment!r} is not a construct in the model")
    details = identification.treatments.get(construct_ids[treatment])
    if details is None or details.status != "identified":
        raise CausalCertificationError(f"effect of {treatment!r} on {outcome!r} is not identified")
    return IdentifiedEstimand(
        dynamical_model_spec_ref=dynamical_model_spec_ref,
        treatment=treatment,
        outcome=outcome,
        estimand=details.estimand,
    )


def certify_conditioned_model(
    dynamical_model_spec: DynamicalModelSpec,
    fitted_dynamical_model_spec: DynamicalModelSpec,
    report: InferenceReportCore,
) -> None:
    """Check resolved fitted values against retained uncertainty and convergence evidence."""
    from grassmoth.models.inputs import input_fingerprints
    from grassmoth.models.ssm.inference.diagnostics.convergence import convergence_failures

    if (
        input_fingerprints(fitted_dynamical_model_spec)["belief"]
        != input_fingerprints(dynamical_model_spec)["belief"]
    ):
        raise CausalCertificationError(
            "The model value differs from the revision certified by the inference log"
        )
    if not dynamical_model_spec.distributions or not dynamical_model_spec.time_points:
        raise CausalCertificationError("The model has no retained joint uncertainty")
    if failures := convergence_failures(report.convergence):
        raise CausalCertificationError(
            "The fit did not pass its convergence checks: "
            + "; ".join(failures)
            + ". Revise the model before reporting causal effects."
        )
