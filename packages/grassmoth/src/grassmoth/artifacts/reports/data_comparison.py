"""Scientific comparisons of saved histories; source observations retain their own owner."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, FiniteFloat

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.identity import GitOid, IndicatorId, IndicatorRef
from grassmoth.artifacts.observations.reference import DataRef
from grassmoth.artifacts.reports.checks import NotEvaluated
from grassmoth.artifacts.reports.data_quality import DataFinding
from grassmoth.artifacts.reports.posterior_diagnostics import PosteriorPredictiveChecks


class Added[PayloadT](Value):
    """The right payload exists at an identity absent on the left."""

    kind: Literal["added"] = "added"
    after: PayloadT


class Removed[PayloadT](Value):
    """The left payload exists at an identity absent on the right."""

    kind: Literal["removed"] = "removed"
    before: PayloadT


class Revised[PayloadT](Value):
    """Both payloads exist at the same identity and their stored values differ."""

    kind: Literal["revised"] = "revised"
    before: PayloadT
    after: PayloadT


type Change[PayloadT] = Annotated[
    Added[PayloadT] | Removed[PayloadT] | Revised[PayloadT], Field(discriminator="kind")
]


class DataPoint(Value):
    """One recorded anchor, support and value; a null value is a present but missing observation.

    Row absence, represented by Added or Removed, is distinct from a present point's null value.
    """

    anchor_time: AwareDatetime
    support_start: AwareDatetime | None
    support_end: AwareDatetime | None
    value: FiniteFloat | None


type DataStatistic = Literal[
    "observed_count", "missing_count", "mean", "sd", "min", "max", "proportion"
]


class ScalarStatisticComparison(Value):
    """One statistic per whole selected history, in the report's source order.

    Histories are never pooled or paired across sides. Null denotes an undefined
    statistic. Discrete codebooks use level proportions instead of numeric moments.
    These descriptive values include each history's own observed positions; the
    predictive checks separately use the reference history's observed-value mask.
    """

    statistic: Literal["observed_count", "missing_count", "mean", "sd", "min", "max"]
    left: tuple[FiniteFloat | None, ...]
    right: tuple[FiniteFloat | None, ...]


class ProportionComparison(Value):
    """The fraction of observed values at one declared level, separately for each history."""

    statistic: Literal["proportion"] = "proportion"
    level: str
    left: tuple[FiniteFloat | None, ...]
    right: tuple[FiniteFloat | None, ...]


type DataStatisticComparison = Annotated[
    ScalarStatisticComparison | ProportionComparison, Field(discriminator="statistic")
]


class PredictiveComparison(Value):
    """One reference history compared with the other side's replicated histories.

    The singleton side owns the reference role even when evaluation is unavailable.
    Swapping sides changes reference_side and preserves the evaluation. Matching
    measurement definitions, calendar binding, and support at every observed anchor
    are required. Missing reference values are masked; missing replicate values at
    observed anchors prevent evaluation. Replicates may have additional anchors.
    The source simulation's law provenance determines prior/posterior interpretation;
    the side names and number of selected histories do not establish that provenance.
    """

    reference_side: Literal["left", "right"]
    evaluation: PosteriorPredictiveChecks | NotEvaluated[IndicatorRef]


class DescriptiveIndicatorComparison(Value):
    """Computed evidence for one persistent indicator identity, without copying its histories.

    Point changes are directional, from left to right, and only computed for one
    history on each side. Match by calendar instant:
    a new anchor is Added, a lost anchor Removed, and an exact value or support change
    at a retained anchor Revised. Omit unchanged points and order changes by anchor.
    Renames and measurement-definition changes are not point revisions; definition
    mismatches are findings. No tolerance, interpolation, or imputation is
    used. Reversing sides exchanges additions/removals and before/after payloads.

    An empty changes tuple can mean identical points or an inapplicable point diff
    (replicated selections); it never asserts that whole
    datasets are equal. Missing variables and differing schedules remain explicit
    findings. Predictive evaluation owns its own applicability, so extra replicate
    anchors can produce a schedule finding without preventing checks at observed anchors.
    """

    indicator_id: IndicatorId
    changes: tuple[Change[DataPoint], ...]
    statistics: tuple[DataStatisticComparison, ...]
    findings: tuple[DataFinding, ...]


class PredictiveIndicatorComparison(Value):
    """Compatible reference and replicate histories with their evaluated predictive comparison."""

    indicator_id: IndicatorId
    changes: tuple[Change[DataPoint], ...]
    statistics: tuple[DataStatisticComparison, ...]
    findings: tuple[DataFinding, ...]
    predictive: PredictiveComparison


type IndicatorComparison = DescriptiveIndicatorComparison | PredictiveIndicatorComparison


class DataComparisonReport(Value):
    """Statistical comparison of two nonempty selections of immutable saved histories.

    Compare data preparation revisions, observations with prior or posterior
    simulations, or simulations across model specifications, priors, fits and
    intervention scenarios.

    Left/right name comparison sides, not temporal revisions or an editing language.
    Resolved references retain every selected replicate in request order; a history
    cannot occur twice within one side. Variables are ordered by indicator identity.
    Exogenous indicators supplied by a simulation's model are excluded.

    One-versus-one comparisons expose point changes. One-versus-many comparisons
    evaluate predictive checks against the singleton reference. Many-versus-many
    comparisons retain per-history statistics without pairing draws or evaluating
    reference-based checks. IndicatorComparison owns the exact change semantics.
    Source definitions and complete observations belong to prepare_data/simulate
    results; this report owns only selection identity and newly computed evidence.
    """

    left: tuple[DataRef[GitOid, int], ...]
    right: tuple[DataRef[GitOid, int], ...]
    variables: tuple[IndicatorComparison, ...]
