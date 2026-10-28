"""Producer-specialized findings, independent of authoring progression."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import (
    ConstructId,
    ConstructRef,
    EntityRef,
    IndicatorRef,
    ParameterRef,
)

type PredictiveCheckReason = Literal[
    "MODEL_INCOMPLETE",
    "MODEL_NOT_EXECUTABLE",
    "NO_COMPATIBLE_PANEL",
    "INSUFFICIENT_OBSERVATION_TIMES",
    "SIMULATION_UNSUPPORTED",
    "ARCHIVED_MEASUREMENT_NOT_RETAINED",
]
type NotEvaluatedReason = (
    PredictiveCheckReason
    | Literal[
        "NONFINITE_EMISSION_MEAN",
        "INSUFFICIENT_TIMES",
        "NO_RELAXATION_TERM",
        "EDGE_CONTRASTS_EXPLICIT",
        "NO_OBSERVATION_SUPPORT",
        "NO_OBSERVATIONS",
        "STATIC_CONSTRUCT",
        "INSUFFICIENT_OBSERVATIONS",
        "ZERO_RESIDUAL_VARIANCE",
        "ZERO_OBSERVED_VARIANCE",
        "NONFINITE_PATHS",
        "NONFINITE_SIGNAL",
        "COMPARISON_INPUTS_MISSING",
        "MISSING_REPLICATE_VALUES",
        "CAUSAL_EVALUATION_FAILED",
        "INSUFFICIENT_CHAIN_SAMPLES",
        "NO_RETAINED_CHAINS",
        "NO_OUTCOME",
        "CONSTRUCT_UNDEFINED",
        "NO_PANEL",
        "STATE_NOT_RECORDED",
    ]
)

type CheckOutcome = Literal["passed", "failed"]
type CheckStatus = CheckOutcome | Literal["not_evaluated"]


class Evaluated[Subject, Evidence](Value):
    """One producer's measured outcome and the evidence supporting it."""

    kind: Literal["evaluated"] = "evaluated"
    code: str
    subject: Subject
    outcome: CheckOutcome
    evidence: Evidence


class NotEvaluated[Subject](Value):
    """An explicit absence of evaluation, with a closed producer reason."""

    kind: Literal["not_evaluated"] = "not_evaluated"
    code: str
    subject: Subject
    reason: NotEvaluatedReason
    detail: str


type Assessment[Subject, Evidence] = Annotated[
    Evaluated[Subject, Evidence] | NotEvaluated[Subject], Field(discriminator="kind")
]


class ConvergenceCriterion(StrEnum):
    """Convergence statistics assessed for each retained parameter coordinate."""

    R_HAT = "r_hat"
    ESS_BULK = "ess_bulk"
    ESS_TAIL = "ess_tail"


class NumericCriterionEvidence(Value):
    """A measured scalar and the producer's numerical acceptance region."""

    criterion: str
    value: float
    lower: float | None = None
    upper: float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    note: str = ""
    display_value: str = ""
    band_label: str = ""

    @property
    def accepted(self) -> bool:
        """Whether the value lies inside the acceptance region; NaN never does."""
        above = self.lower is None or (
            self.value >= self.lower if self.lower_inclusive else self.value > self.lower
        )
        below = self.upper is None or (
            self.value <= self.upper if self.upper_inclusive else self.value < self.upper
        )
        return above and below


class PredictiveSubject(Value):
    """One named check and its stable target in a construct's scientific context."""

    construct_id: ConstructId | None = None
    target: EntityRef | Literal["whole_model", "observations"]


class IndicatorCheckSubject(Value):
    """The indicator and criterion remain present when evaluation is unavailable."""

    target: IndicatorRef


class ConvergenceSubject(Value):
    """A convergence criterion on one stable scientific scalar."""

    parameter: ParameterRef
    label: str


class LatentConvergenceSubject(Value):
    """A convergence criterion over a construct's independent latent coordinates."""

    state: ConstructRef
    label: str


class OutcomeSubject(Value):
    """The outcome construct selected by the study question."""

    outcome: ConstructRef


class QueryTargetSubject(Value):
    """An intervention target in one named query."""

    query: str
    target: ConstructRef


class QueryWindowSubject(Value):
    """The observation window of one named query."""

    query: str


type QuestionSubject = OutcomeSubject | QueryTargetSubject | QueryWindowSubject
type FindingSubject = (
    str
    | EntityRef
    | PredictiveSubject
    | IndicatorCheckSubject
    | ConvergenceSubject
    | LatentConvergenceSubject
    | QuestionSubject
)


type SpecificationAssessment = Assessment[str, str]
type PredictiveAssessment = Assessment[PredictiveSubject, tuple[NumericCriterionEvidence, ...]]
