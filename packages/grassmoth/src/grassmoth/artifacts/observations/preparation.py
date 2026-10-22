"""Data-owned preparation instructions, source references and observation metadata."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from polars._typing import FillNullStrategy
from pydantic import (
    AwareDatetime,
    Field,
    FiniteFloat,
    computed_field,
    model_validator,
)

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.observations.duration import Duration
from grassmoth.artifacts.observations.expressions import WindowExpression
from grassmoth.artifacts.observations.semantics import (
    SummaryOperator,
    derive_indicator_observation_semantics,
)
from grassmoth.artifacts.observations.spec import (
    AuthoredObservationSpec,
    ResolvedObservationSpec,
)

_SEMANTIC_COLLISIONS: tuple[tuple[str, frozenset[SummaryOperator], str], ...] = (
    (
        r"\bcount\b|\bnumber of\b|\bhow many\b",
        frozenset({SummaryOperator.MEAN, SummaryOperator.STD}),
        "how_to_measure implies counting but aggregation computes a statistic",
    ),
    (
        r"\baverage\b|\bmean\b",
        frozenset(
            {
                SummaryOperator.SUM,
                SummaryOperator.FIRST,
                SummaryOperator.LAST,
                SummaryOperator.COUNT,
            }
        ),
        "how_to_measure implies averaging but aggregation is not mean",
    ),
    (
        r"\btotal\b|\bcumulative\b|\bsum\b",
        frozenset({SummaryOperator.MEAN, SummaryOperator.FIRST, SummaryOperator.LAST}),
        "how_to_measure implies summing but aggregation is not sum",
    ),
    (
        r"\blast\b|\bmost recent\b|\bcurrent\b",
        frozenset({SummaryOperator.MEAN, SummaryOperator.SUM}),
        "how_to_measure implies point-in-time but aggregation is a window statistic",
    ),
)


def check_semantic_collisions(
    how_to_measure: str,
    aggregation: SummaryOperator,
) -> list[str]:
    """Check for inconsistencies between how_to_measure text and aggregation."""
    warnings: list[str] = []
    text_lower = how_to_measure.lower()
    for pattern, conflict_aggs, explanation in _SEMANTIC_COLLISIONS:
        match = re.search(pattern, text_lower)
        if aggregation in conflict_aggs and match:
            warnings.append(
                f"Semantic collision: {explanation}. "
                f"how_to_measure contains '{match.group()}' "
                f"but aggregation='{aggregation}'."
            )
    return warnings


class SemanticExtractionSpec(Value):
    """Interpret source records using an explicit measurement rubric."""

    kind: Literal["semantic"] = "semantic"
    how_to_measure: str = Field(
        min_length=1, description="Scoring rubric and extraction instructions."
    )
    source_columns: tuple[str, ...] = Field(
        default=(), description="Source columns exposed to the extraction worker."
    )


class ComputedExtractionSpec(Value):
    """Compute a deterministic support-window measurement from source columns."""

    kind: Literal["computed"] = "computed"
    how_to_measure: str = Field(
        min_length=1, description="Description of the deterministic measurement."
    )
    source_columns: tuple[str, ...] = Field(min_length=1)
    computed_rule: WindowExpression | None = Field(
        default=None,
        description=(
            "Optional deterministic support-window expression over the declared source columns. "
            "It must return one scalar per window with the observation's declared summary operator. "
            "Omitted uses a direct single-column aggregation."
        ),
    )
    fill_null: FillNullStrategy | Annotated[FiniteFloat, Field(strict=True)] | None = Field(
        default=None,
        description=(
            "Optional Polars null filling after aggregation on the sorted time grid within the "
            "selected data span. Use forward, backward, min, max, mean, zero, one, or a numeric "
            "constant. Fills every null, including explicit unknown readings. Omitted leaves "
            "nulls unknown. Forward carries the last value and leaves leading nulls unknown."
        ),
    )
    fill_null_limit: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Maximum consecutive nulls filled by forward/backward; omitted is unlimited. "
            "Only valid when fill_null is forward or backward."
        ),
    )

    @model_validator(mode="after")
    def validate_computation(self) -> Self:
        """Require declared input columns and a fill limit compatible with the chosen extraction rule."""
        if self.fill_null_limit is not None and self.fill_null not in {"forward", "backward"}:
            raise ValueError("fill_null_limit requires fill_null='forward' or 'backward'")
        if self.computed_rule is None:
            if len(self.source_columns) != 1:
                raise ValueError("Direct computed extraction requires exactly 1 source_column")
        else:
            referenced = self.computed_rule.dependencies
            if not referenced:
                raise ValueError("computed_rule must reference at least 1 source_column")
            if unknown := sorted(referenced - set(self.source_columns)):
                raise ValueError(f"computed_rule references undeclared source_columns: {unknown}")
        return self


class RecordedExtractionSpec(Value):
    """Import already measured readings and their actual support without re-bucketing."""

    kind: Literal["recorded"] = "recorded"
    how_to_measure: str = "Import the recorded reading and temporal support unchanged."
    value_column: str = Field(min_length=1)
    anchor_column: str = Field(default="timestamp", min_length=1)
    support_start_column: str = Field(min_length=1)
    support_end_column: str = Field(min_length=1)

    @property
    def source_columns(self) -> tuple[str, ...]:
        """Columns owning the reading and its recorded temporal support."""
        return (
            self.value_column,
            self.anchor_column,
            self.support_start_column,
            self.support_end_column,
        )


type ExtractionSpec = Annotated[
    ComputedExtractionSpec | SemanticExtractionSpec | RecordedExtractionSpec,
    Field(discriminator="kind"),
]


class DataVariableSpec(Value):
    """Compose an observed variable with its data-owned extraction instructions."""

    observation: AuthoredObservationSpec
    extraction: ExtractionSpec

    @model_validator(mode="after")
    def validate_computed_summary(self) -> Self:
        """Require the computed rule to admit the observation's aggregation and measurement kind."""
        if isinstance(self.extraction, ComputedExtractionSpec):
            derive_indicator_observation_semantics(
                self.observation.aggregation,
                self.observation.measurement_dtype,
                self.extraction.computed_rule,
            )
        return self


class DataPreparationSpec(Value):
    """The model-owned observation definitions resolved for extraction."""

    default_window: Duration
    variables: tuple[DataVariableSpec, ...] = Field(min_length=1)
    context: str = Field(
        default="", description="Optional context for interpreting the source data."
    )

    @model_validator(mode="after")
    def unique_variables(self) -> Self:
        """Reject preparation recipes that assign the same observation ID more than once."""
        if len({item.observation.id for item in self.variables}) != len(self.variables):
            raise ValueError("Prepared variables must have unique IDs")
        return self

    def observation_schema(self) -> tuple[ResolvedObservationSpec, ...]:
        """Resolve each observation's window, using the recipe's default where none was authored."""
        return tuple(
            item.observation.resolved(item.observation.observation_window or self.default_window)
            for item in self.variables
        )


class PreparedDataMetadata(Value):
    """An uploaded panel's recipe owns its resolved observation schema."""

    preparation: DataPreparationSpec
    time_origin: AwareDatetime = Field(description="Calendar instant of model day zero.")

    @computed_field
    @property
    def variables(self) -> tuple[ResolvedObservationSpec, ...]:
        """Observation definitions resolved against the clock retained in the preparation recipe."""
        return self.preparation.observation_schema()


class CompletedExtractionWorker(Value):
    """Counts of extracted observations and windows retained from a completed worker."""

    worker_id: int
    n_extractions: int
    n_windows: int
    status: Literal["completed"] = "completed"


class FailedExtractionChunk(Value):
    """A failed extraction chunk with its error, retained counts, and execution details."""

    worker_id: int
    n_extractions: int
    n_windows: int
    status: Literal["failed"] = "failed"
    error: str
    n_llm_calls: int | None = 0
    reused: bool | None = False


type ExtractionWorkerResult = Annotated[
    CompletedExtractionWorker | FailedExtractionChunk, Field(discriminator="status")
]


class DataPreparationResult(Value):
    """Preparation artifacts and the measurements actually retained by extraction."""

    workers: tuple[ExtractionWorkerResult, ...] = ()
    extraction_reused: int | None = None
