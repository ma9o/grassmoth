"""Read ready-to-use source tables without model calls or generated parsing code."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
from temporalio import activity

from grassmoth_server.actions.temporal.extraction.messages import ReadSourceDataInput
from grassmoth_server.storage import backend as storage
from grassmoth_server.storage import paths as data
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import Rejected
from grassmoth_server.study.state import ArtifactRecord


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
            data.call_dir(activity_input.workspace_id, activity_input.call_id),
            "sources",
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
    directory = Path(data.call_dir(activity_input.workspace_id, activity_input.call_id))
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, directory / "raw.parquet", compression="zstd")
    return ArtifactStore(activity_input.workspace_id).raw_record(activity_input.call_id)
