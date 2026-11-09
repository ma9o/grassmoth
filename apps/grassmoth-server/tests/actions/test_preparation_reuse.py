"""Preparation reuses effective LLM requests while spans and computation rerun."""

import io
import json
from datetime import UTC, date, datetime
from types import SimpleNamespace
from uuid import uuid4

import polars as pl
import pytest
from pydantic import TypeAdapter

from grassmoth.actions.contracts.io import SourceTable
from grassmoth.actions.prepare_data.aggregations import compute_indicators
from grassmoth.actions.prepare_data.rows import annotate_observation_rows, prepared_time_origin
from grassmoth.artifacts.observations.preparation import (
    ComputedExtractionSpec,
    DataPreparationSpec,
    DataVariableSpec,
    SemanticExtractionSpec,
)
from grassmoth.artifacts.observations.spec import AuthoredObservationSpec, NumericMeasurement
from grassmoth_server.actions.contracts.preparation import FilePreparationSpec, FileSourceRef
from grassmoth_server.actions.prepare_data.context import extraction_context
from grassmoth_server.actions.prepare_data.worker_output import WindowExtraction
from grassmoth_server.actions.temporal.extraction.activities import (
    finalize_extraction_chunk_activity,
    plan_measurements_activity,
)
from grassmoth_server.actions.temporal.extraction.messages import (
    ExtractionChunkFinalizeInput,
    MeasurementChunkContext,
    MeasurementsWorkflowInput,
)
from grassmoth_server.actions.temporal.llm.storage import read_subroutine_json
from grassmoth_server.storage import backend as storage
from grassmoth_server.storage import paths as data
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server_test_support.data import preparation_input
from grassmoth_server_test_support.helpers import run_async
from grassmoth_server_test_support.storage import seed_source_table
from grassmoth_test_support.values import revise_fixture

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("code", ["PreconditionFailed", "412", "AccessDenied"])
def test_remote_cache_uses_the_conditional_write_winner_only(code, monkeypatch):
    from botocore.exceptions import ClientError

    from grassmoth_server.storage.cache import publish

    def conflict(path, value, *, mode):
        assert (path, value, mode) == ("s3://cache/key", b"loser", "create")
        raise OSError("conditional create failed") from ClientError(
            {"Error": {"Code": code}}, "PutObject"
        )

    monkeypatch.setattr(storage, "is_remote", lambda: True)
    monkeypatch.setattr(storage, "get_fs", lambda: SimpleNamespace(pipe_file=conflict))
    monkeypatch.setattr(storage, "open_file", lambda _path, _mode: io.BytesIO(b"winner"))
    if code == "AccessDenied":
        with pytest.raises(OSError, match="conditional create failed"):
            publish("s3://cache/key", b"loser")
    else:
        assert publish("s3://cache/key", b"loser") == b"winner"


def _variable(name):
    return DataVariableSpec(
        observation=AuthoredObservationSpec(
            observation_window=None,
            id=f"indicator:{name}",
            name=name,
            measurement=NumericMeasurement(dtype="continuous"),
            aggregation="last",
        ),
        extraction=SemanticExtractionSpec(
            source_columns=(name,), how_to_measure="Read the daily score"
        ),
    )


def test_extraction_cache_is_shared_and_only_changed_effective_requests_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    raw = pl.DataFrame(
        {"timestamp": [datetime(2026, 1, day) for day in (1, 2, 3)], "x": [1, 2, 3], "y": [4, 5, 6]}
    )

    def plan(workspace, variables, end, context="Daily scores"):
        store = ArtifactStore(workspace)
        artifact = seed_source_table(store, raw, "seq-000001")
        return run_async(
            plan_measurements_activity(
                MeasurementsWorkflowInput(
                    call_id="seq-000001",
                    workspace_id=workspace,
                    seq=1,
                    attempt_id=uuid4(),
                    raw_data_revision=artifact.revision,
                    preparation=preparation_input(
                        store,
                        FilePreparationSpec(
                            source={"files": ["scores.csv"], "start": "2026-01-01", "end": end},
                            definition=DataPreparationSpec(
                                default_window="1d", context=context, variables=variables
                            ),
                        ),
                    ),
                )
            )
        )

    first = plan("first", (_variable("x"),), "2026-01-03")
    assert len(first.chunks) == 2
    for chunk in first.chunks:
        spec = read_subroutine_json(chunk.spec_ref, MeasurementChunkContext)
        assert chunk.n_windows == 1
        assert len(spec.measurement_structure.indicators) == 1
        result_ref = chunk.spec_ref + ".result"
        storage.write_text(
            result_ref,
            json.dumps(
                {
                    "extractions": [
                        {
                            "indicator_id": "indicator:x",
                            "value": 7,
                            "window_start": spec.window_starts[0],
                        }
                    ]
                }
            ),
        )
        request = ExtractionChunkFinalizeInput(
            workspace_id="first",
            run_id=first.run_id,
            worker_id=chunk.worker_id,
            attempt=1,
            n_windows=1,
            n_llm_calls=1,
            spec_ref=chunk.spec_ref,
            result_ref=result_ref,
            conversation_ref="unused",
        )
        completed = run_async(finalize_extraction_chunk_activity(request))
        # A concurrent later validated answer cannot replace the first one.
        storage.write_text(
            result_ref, storage.read_text(result_ref).replace('"value": 7', '"value": 99')
        )
        again = run_async(finalize_extraction_chunk_activity(request))
        assert (
            storage.read_json(again.result_ref)["extractions"]
            == storage.read_json(completed.result_ref)["extractions"]
        )
        assert (
            TypeAdapter(list[WindowExtraction])
            .validate_python(storage.read_json(again.result_ref)["extractions"])[0]
            .value
            == 7
        )
    same = plan("second", (_variable("x"),), "2026-01-03")
    assert all(chunk.cached_result_ref for chunk in same.chunks)
    expanded = plan("third", (_variable("x"), _variable("y")), "2026-01-04")
    assert len(expanded.chunks) == 6
    assert sum(chunk.cached_result_ref is not None for chunk in expanded.chunks) == 2
    changed = plan(
        "fourth", (_variable("x"),), "2026-01-03", context="Use a different interpretation"
    )
    assert all(chunk.cached_result_ref is None for chunk in changed.chunks)

    from grassmoth_server.actions.temporal.extraction import cache as preparation_cache

    with monkeypatch.context() as policy:
        policy.setattr(preparation_cache, "EXTRACTION_POLICY_VERSION", "changed-extraction")
        assert all(
            chunk.cached_result_ref is None
            for chunk in plan("policy", (_variable("x"),), "2026-01-03").chunks
        )
    original_open = storage.open_file

    def evict(path, mode="r", **kwargs):
        if ".preparation-cache" in str(path) and mode == "rb":
            from pathlib import Path

            Path(path).unlink()
        return original_open(path, mode, **kwargs)

    with monkeypatch.context() as eviction:
        eviction.setattr(storage, "open_file", evict)
        assert all(
            chunk.cached_result_ref is None
            for chunk in plan("evicted", (_variable("x"),), "2026-01-03").chunks
        )


def test_span_keeps_complete_windows_and_never_fills_from_excluded_history():
    raw = pl.DataFrame({"timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 3)], "x": [9, 2]})
    variable = revise_fixture(
        _variable("x"),
        extraction=ComputedExtractionSpec(
            how_to_measure="Read the daily score", source_columns=("x",), fill_null="forward"
        ),
    )
    span = FileSourceRef(files=("source.csv",), start=date(2026, 1, 2), end=date(2026, 1, 4))
    bounded = compute_indicators(
        SourceTable(rows=raw.to_arrow(), start=span.start, end=span.end),
        DataPreparationSpec(default_window="1d", variables=(variable,)),
    )
    assert bounded["timestamp"].to_list() == ["2026-01-02T00:00:00", "2026-01-03T00:00:00"]
    assert bounded["value"].to_list() == [None, "2"]
    from grassmoth_server.actions.prepare_data.planning import prepare_semantic_chunks

    semantic_context = extraction_context(
        FilePreparationSpec(
            source=span,
            definition=DataPreparationSpec(default_window="1d", variables=(_variable("x"),)),
        )
    )
    texts, windows, _, empty_output = prepare_semantic_chunks(
        raw_df=raw,
        measurement_structure=semantic_context,
        time_col="timestamp",
        max_events_per_window=300,
    )
    assert windows == [["2026-01-03T00:00:00"]]
    assert len(texts) == 1
    assert empty_output.extractions == (
        WindowExtraction(
            indicator_id="indicator:x", window_start="2026-01-02T00:00:00", value=None
        ),
    )
    assert prepared_time_origin(pl.DataFrame(), date(2026, 1, 2)) == datetime(
        2026, 1, 2, tzinfo=UTC
    )


@pytest.mark.parametrize(("aggregation", "offset"), [("first", 0), ("last", 2)])
def test_multiday_span_and_empty_semantic_windows_materialize_without_requests(
    tmp_path, monkeypatch, aggregation, offset
):
    from grassmoth_server.actions.temporal.extraction.activities import (
        finalize_measurements_activity,
    )
    from grassmoth_server.actions.temporal.extraction.messages import MeasurementsFinalizeInput
    from grassmoth_server.study.queries.data import read_data_source

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    raw = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, day) for day in (1, 3, 5, 9)],
            "x": [9, None, 2, 12],
            "unselected": [99, 99, 99, 99],
        }
    )
    store = ArtifactStore("empty")
    artifact = seed_source_table(store, raw, "seq-000001")
    variable = revise_fixture(
        _variable("x"),
        observation=revise_fixture(_variable("x").observation, aggregation=aggregation),
    )
    preparation = FilePreparationSpec(
        source={"files": ["scores.csv"], "start": "2026-01-02", "end": "2026-01-10"},
        definition=DataPreparationSpec(default_window="2d", variables=(variable,)),
    )
    plan = run_async(
        plan_measurements_activity(
            MeasurementsWorkflowInput(
                call_id="seq-000001",
                workspace_id="empty",
                seq=1,
                attempt_id=uuid4(),
                raw_data_revision=artifact.revision,
                preparation=preparation_input(store, preparation),
            )
        )
    )
    assert len(plan.chunks) == 1  # All-null Jan 3 and no-row Jan 7 never become LLM requests.
    chunk = plan.chunks[0]
    spec = read_subroutine_json(chunk.spec_ref, MeasurementChunkContext)
    assert spec.window_starts == ["2026-01-05T00:00:00"]
    result_ref = chunk.spec_ref + ".result"
    storage.write_text(
        result_ref,
        json.dumps(
            {
                "extractions": [
                    {
                        "indicator_id": "indicator:x",
                        "window_start": spec.window_starts[0],
                        "value": 2,
                    }
                ]
            }
        ),
    )
    result = run_async(
        finalize_extraction_chunk_activity(
            ExtractionChunkFinalizeInput(
                workspace_id="empty",
                run_id=plan.run_id,
                worker_id=chunk.worker_id,
                attempt=1,
                n_windows=1,
                n_llm_calls=0,
                spec_ref=chunk.spec_ref,
                result_ref=result_ref,
                conversation_ref="unused",
            )
        )
    )
    effects = run_async(
        finalize_measurements_activity(
            MeasurementsFinalizeInput(
                workspace_id="empty",
                run_id=plan.run_id,
                pins=plan.pins,
                plan_ref=plan.plan_ref,
                chunk_results=[result],
            )
        )
    )
    revision = effects.effects.produced[0].revision
    from grassmoth.actions.contracts.io import PrepareDataOutput

    output = read_data_source(store, revision)
    assert isinstance(output, PrepareDataOutput)
    series = next(iter(output.observations().series.values()))
    readings = series.history.readings()
    assert [row[1] for row in readings] == [None, 2.0, None]
    assert [row[0] for row in readings] == [day + offset for day in (1, 3, 5)]
    assert [row[2] for row in readings] == [1, 3, 5]
    assert [row[3] for row in readings] == [3, 5, 7]
    assert output.metadata.time_origin == datetime(2026, 1, 2, tzinfo=UTC)
    assert len(list((tmp_path / ".preparation-cache/measurement_extraction").iterdir())) == 1

    computed_variable = DataVariableSpec(
        observation=variable.observation,
        extraction=ComputedExtractionSpec(
            how_to_measure=variable.extraction.how_to_measure,
            source_columns=variable.extraction.source_columns,
        ),
    )
    computed = compute_indicators(
        SourceTable(
            rows=raw.to_arrow(), start=preparation.source.start, end=preparation.source.end
        ),
        DataPreparationSpec(default_window="2d", variables=(computed_variable,)),
    )
    rows = annotate_observation_rows(computed, preparation.definition.observation_schema())
    assert rows["anchor_time"].to_list() == [
        datetime(2026, 1, day + offset).isoformat() for day in (3, 5, 7)
    ]
    assert computed["value"].to_list() == [None, "2", None]
