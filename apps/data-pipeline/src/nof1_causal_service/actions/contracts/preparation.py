"""Captured upload sources and extraction requests owned by the service."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Annotated, Self

from pydantic import AfterValidator, Field, field_validator, model_validator

from nof1_causal_lab.public import DataPreparationSpec, Value


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


class FilePreparationSpec(Value):
    """Ready-to-use source tables and the recipe for preparing their observations."""

    source: FileSourceRef
    definition: DataPreparationSpec
