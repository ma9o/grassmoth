"""Read ready-to-use source tables without model calls or generated parsing code."""

from __future__ import annotations

from pathlib import PurePosixPath

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
from temporalio import activity

from nof1_causal_lab.actions.temporal.extraction.messages import ReadSourceDataInput
from nof1_causal_lab.storage import backend as storage
from nof1_causal_lab.storage import paths as data
from nof1_causal_lab.study.artifact_files import parquet_filename
from nof1_causal_lab.study.persistence.store import ArtifactStore
from nof1_causal_lab.study.records import Rejected
from nof1_causal_lab.study.state import ArtifactRecord


@activity.defn
async def read_source_data_activity(
    activity_input: ReadSourceDataInput,
) -> ArtifactRecord | Rejected:
    """Capture CSV or Parquet rows as supplied, preserving their source order."""
    tables: list[pa.Table] = []
    for filename in activity_input.source.files:
        suffix = PurePosixPath(filename).suffix.lower()
        if suffix not in {".csv", ".parquet"}:
            return Rejected(
                code="SOURCE_INVALID",
                subject="inputs",
                detail=f"{filename}: prepare_data requires ready-to-use CSV or Parquet tables.",
            )
        path = storage.join(
            data.scratch_dir(activity_input.workspace_id),
            "source-files",
            activity_input.source.hashes[filename],
            filename,
        )
        try:
            with storage.open_file(path, "rb") as source:
                table = csv.read_csv(source) if suffix == ".csv" else pq.read_table(source)
        except pa.ArrowInvalid as exc:
            return Rejected(code="SOURCE_INVALID", subject="inputs", detail=f"{filename}: {exc}")
        if "timestamp" not in table.column_names:
            return Rejected(
                code="SOURCE_INVALID",
                subject="inputs",
                detail=f"{filename}: a timestamp column is required.",
            )
        timestamp = table.column("timestamp")
        if not (pa.types.is_timestamp(timestamp.type) or pa.types.is_date(timestamp.type)):
            return Rejected(
                code="SOURCE_INVALID",
                subject="inputs",
                detail=f"{filename}: timestamp must contain dates or datetimes.",
            )
        if timestamp.null_count:
            return Rejected(
                code="SOURCE_INVALID",
                subject="inputs",
                detail=f"{filename}: timestamp cannot contain nulls.",
            )
        tables.append(table)
    try:
        table = pa.concat_tables(tables, promote_options="permissive")
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        return Rejected(
            code="SOURCE_INVALID", subject="inputs", detail=f"Incompatible source tables: {exc}"
        )
    if table.num_rows == 0:
        return Rejected(
            code="SOURCE_INVALID", subject="inputs", detail="Source tables contain no records."
        )
    return ArtifactStore(activity_input.workspace_id).write_artifact(
        "raw_data",
        derived_from={},
        produced_by="prepare_data",
        parquet_files={parquet_filename("raw_data", "raw"): table},
    )
