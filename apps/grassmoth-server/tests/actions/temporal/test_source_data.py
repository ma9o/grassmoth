"""Source tables are read verbatim without a parsing agent."""

import hashlib
from datetime import datetime

import polars as pl
import pytest

from grassmoth_server.actions.temporal.extraction.messages import ReadSourceDataInput
from grassmoth_server.actions.temporal.extraction.source_data import read_source_data_activity
from grassmoth_server.storage import paths as data
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import Rejected
from tests.support.helpers import run_async

pytestmark = pytest.mark.contract


def _read_sources(tmp_path, monkeypatch, sources):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    hashes = {}
    for filename, content in sources.items():
        digest = hashlib.sha256(content).hexdigest()
        hashes[filename] = digest
        path = tmp_path / "ws" / "scratch" / "source-files" / digest / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return run_async(
        read_source_data_activity(
            ReadSourceDataInput(
                workspace_id="ws", source={"files": tuple(sources), "hashes": hashes}
            )
        )
    )


def test_csv_sources_preserve_rows_columns_and_missing_values(tmp_path, monkeypatch):
    result = _read_sources(
        tmp_path,
        monkeypatch,
        {
            "input/a.csv": b"timestamp,score,text\n2026-01-02T08:00:00,2,hello\n",
            "input/nested/b.csv": b"timestamp,score,text\n2026-01-01T08:00:00,,world\n",
        },
    )
    assert not isinstance(result, Rejected)
    table = ArtifactStore("ws").read_parquet_table("raw_data", result.revision, "raw.parquet")
    assert table.to_pydict() == {
        "timestamp": [datetime(2026, 1, 2, 8), datetime(2026, 1, 1, 8)],
        "score": [2, None],
        "text": ["hello", "world"],
    }


def test_parquet_preserves_authored_column_metadata(tmp_path, monkeypatch):
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pl.DataFrame({"timestamp": [datetime(2026, 1, 1)], "score": [2]}).to_arrow()
    table = table.cast(
        pa.schema(
            [field.with_metadata({b"description": field.name.encode()}) for field in table.schema]
        )
    )
    content = io.BytesIO()
    pq.write_table(table, content)
    result = _read_sources(tmp_path, monkeypatch, {"input/scores.parquet": content.getvalue()})
    assert not isinstance(result, Rejected)
    saved = ArtifactStore("ws").read_parquet_table("raw_data", result.revision, "raw.parquet")
    assert saved.equals(table, check_metadata=True)


@pytest.mark.parametrize(
    ("filename", "content", "reason"),
    [
        ("archive.zip", b"raw archive", "ready-to-use CSV or Parquet"),
        ("scores.csv", b"date,score\n2026-01-01,2\n", "timestamp column"),
        ("scores.csv", b"timestamp,score\nunknown,2\n", "dates or datetimes"),
        ("scores.csv", b"timestamp,score\n2026-01-01,2\n,3\n", "cannot contain nulls"),
        ("scores.csv", b"timestamp,score\n", "timestamp must contain dates or datetimes"),
    ],
)
def test_unprepared_sources_are_rejected(tmp_path, monkeypatch, filename, content, reason):
    result = _read_sources(tmp_path, monkeypatch, {f"input/{filename}": content})
    assert isinstance(result, Rejected)
    assert result.code == "SOURCE_INVALID"
    assert reason in result.detail
