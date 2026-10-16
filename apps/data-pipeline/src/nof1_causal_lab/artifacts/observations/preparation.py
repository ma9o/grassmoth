"""Data-owned preparation instructions, source references and observation metadata."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date
from typing import Annotated, Literal, Self

from polars._typing import FillNullStrategy
from pydantic import (
    AfterValidator,
    AwareDatetime,
    Field,
    FiniteFloat,
    computed_field,
    field_validator,
    model_validator,
)

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.observations.duration import Duration
from nof1_causal_lab.artifacts.observations.expressions import WindowExpression
from nof1_causal_lab.artifacts.observations.semantics import (
    SummaryOperator,
    derive_indicator_observation_semantics,
)
from nof1_causal_lab.artifacts.observations.spec import (
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
        return (self.value_column, self.anchor_column, self.support_start_column, self.support_end_column)


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


def _validate_source_folder(value: str) -> str:
    if not value or value in {".", ".."} or any(char in value for char in ("/", "\\", "\0")):
        raise ValueError("Use one folder name under the workspace data directory")
    return value


type SourceFolder = Annotated[
    str,
    Field(
        min_length=1,
        description="Folder of ready-to-use CSV or Parquet tables under data/{workspace_id}/, such as input. Every table must have a date or datetime timestamp column.",
    ),
    AfterValidator(_validate_source_folder),
]


def _validate_source_path(value: str) -> str:
    if any(part in {"", ".", ".."} for part in value.split("/")) or any(
        char in value for char in ("\\", "\0")
    ):
        raise ValueError("Source files must have relative paths within the workspace")
    return value


class FileSourceRef(Value):
    """Captured source files with workspace-relative paths and their content hashes."""

    files: tuple[str, ...] = Field(min_length=1)
    hashes: Mapping[str, Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]] = Field(
        default_factory=dict,
        description="Call-time SHA-256 of every captured source file, retained with the resolved call.",
    )
    start: date | None = Field(default=None, description="Inclusive UTC source-coverage date.")
    end: date | None = Field(default=None, description="Exclusive UTC source-coverage date.")

    @field_validator("files")
    @classmethod
    def validate_filenames(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicate source filenames and parse each as a permitted relative source path."""
        if len(set(values)) != len(values):
            raise ValueError("Source filenames must be unique")
        return tuple(_validate_source_path(value) for value in values)

    @model_validator(mode="after")
    def ordered_bounds(self) -> Self:
        """Require complete file-hash coverage when supplied and a strictly ordered source window."""
        if self.hashes and set(self.hashes) != set(self.files):
            raise ValueError("File hashes must name every source file exactly once")
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("Source coverage start must precede end")
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


class FilePreparationSpec(Value):
    """Ready-to-use source tables and the recipe for preparing their observations."""

    source: FileSourceRef
    definition: DataPreparationSpec


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
