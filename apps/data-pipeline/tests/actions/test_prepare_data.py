"""One model-driven preparation action retains extraction semantics and data findings."""

import json
from datetime import UTC, datetime
from uuid import uuid4

import polars as pl
import pytest

from nof1_causal_lab.actions.contracts.bodies import PrepareDataInput
from nof1_causal_lab.actions.contracts.requests import PrepareDataRequest
from nof1_causal_lab.actions.data_checks import evaluate_data_checks
from nof1_causal_lab.actions.messages import preparation_messages
from nof1_causal_lab.actions.prepare_data.context import extraction_context
from nof1_causal_lab.actions.temporal.extraction.activities import (
    finalize_measurements_activity,
    plan_measurements_activity,
)
from nof1_causal_lab.actions.temporal.extraction.messages import (
    CompletedExtractionChunk,
    MeasurementsFinalizeInput,
    MeasurementsWorkflowInput,
)
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.observations.preparation import (
    ComputedExtractionSpec,
    DataPreparationSpec,
    DataVariableSpec,
    FilePreparationSpec,
    FileSourceRef,
    SemanticExtractionSpec,
)
from nof1_causal_lab.artifacts.observations.spec import AuthoredObservationSpec
from nof1_causal_lab.storage import backend as storage
from nof1_causal_lab.study.persistence.store import ArtifactStore
from nof1_causal_lab.study.queries.lineage import read_data_metadata
from nof1_causal_lab.study.state import StudyState
from tests.support.helpers import run_async

pytestmark = pytest.mark.contract


def test_preparation_from_model_combines_computed_and_semantic_workers(monkeypatch, tmp_path):
    from nof1_causal_lab.storage import paths as data

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))

    store = ArtifactStore("data-only")
    preparation = DataPreparationSpec(
        default_window="1d",
        context="Daily diary scoring",
        variables=(
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    observation_window=None,
                    id="indicator:steps",
                    name="steps",
                    measurement_dtype="count",
                    aggregation="sum",
                ),
                extraction=ComputedExtractionSpec(
                    how_to_measure="Total recorded steps", source_columns=("steps",), fill_null=0
                ),
            ),
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    observation_window=None,
                    id="indicator:stress",
                    name="stress",
                    measurement_dtype="ordinal",
                    aggregation="last",
                    ordinal_levels=("low", "medium", "high"),
                ),
                extraction=SemanticExtractionSpec(
                    how_to_measure="Score the diary as low, medium or high stress",
                    source_columns=("diary",),
                ),
            ),
        ),
    )
    from tests.support.data import preparation_input

    prepared_input = preparation_input(
        store, FilePreparationSpec(source={"files": ["diary.csv"]}, definition=preparation)
    )
    request = PrepareDataRequest[GitOid, FileSourceRef](input=prepared_input)
    assert isinstance(request.input, PrepareDataInput)
    assert "definition" not in request.input.model_dump()
    raw = store.write_artifact(
        "raw_data",
        derived_from={},
        produced_by="prepare_data",
        parquet_files={
            "raw.parquet": pl.DataFrame(
                {
                    "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 3)],
                    "steps": [4, 8],
                    "diary": ["calm", "overwhelmed"],
                }
            )
        },
    )
    state = StudyState().with_artifacts([raw])
    plan = run_async(
        plan_measurements_activity(
            MeasurementsWorkflowInput(
                workspace_id="data-only",
                seq=1,
                attempt_id=uuid4(),
                raw_data_revision=raw.revision,
                preparation=request.input,
            )
        )
    )
    assert plan.pins == {"raw_data": raw.revision, "model": request.input.dynamical_model_spec_ref}
    scores = {"2026-01-01T00:00:00": "0", "2026-01-03T00:00:00": "2"}
    assert len(plan.chunks) == len(scores)
    results = []
    for chunk in plan.chunks:
        spec = json.loads(storage.read_text(chunk.spec_ref))
        assert chunk.n_windows == 1
        (window_start,) = spec["window_starts"]
        (indicator,) = spec["measurement_structure"]["indicators"]
        assert indicator["observation"]["id"] == "indicator:stress"
        result_path = str(tmp_path / f"worker-{chunk.worker_id}-result.json")
        storage.write_text(
            result_path,
            json.dumps(
                {
                    "dataframe": [
                        {
                            "indicator_id": indicator["observation"]["id"],
                            "value": scores[window_start],
                            "timestamp": window_start,
                        },
                    ]
                }
            ),
        )
        results.append(
            CompletedExtractionChunk(
                worker_id=chunk.worker_id,
                n_windows=chunk.n_windows,
                status="completed",
                n_extractions=1,
                result_ref=result_path,
            )
        )
    effects = run_async(
        finalize_measurements_activity(
            MeasurementsFinalizeInput(
                workspace_id="data-only",
                run_id=plan.run_id,
                plan_ref=plan.plan_ref,
                pins=plan.pins,
                chunk_results=results,
            )
        )
    )
    profile = evaluate_data_checks("data-only", state, effects)
    assert {item.artifact_id for item in effects.effects.produced} == {"panel"}
    panel = next(item for item in effects.effects.produced if item.artifact_id == "panel")
    assert panel.derived_from == {
        "raw_data": raw.revision,
        "model": prepared_input.dynamical_model_spec_ref,
    }
    observations = store.read_parquet_file("panel", panel.revision, "panel.parquet")
    assert observations.select("indicator_id", "value").rows() == [
        ("indicator:steps", 4),
        ("indicator:steps", 0),
        ("indicator:steps", 8),
        ("indicator:stress", 0),
        ("indicator:stress", None),
        ("indicator:stress", 2),
    ]
    metadata = read_data_metadata(store, panel.revision)
    assert metadata.preparation is not None
    assert (
        metadata.preparation.variables[1].extraction.how_to_measure
        == preparation.variables[1].extraction.how_to_measure
    )
    assert metadata.variables[1].ordinal_levels == ("low", "medium", "high")
    assert set(profile.indicators) == {"indicator:steps", "indicator:stress"}
    labels = preparation_messages(profile, effects.result, datetime.now(UTC))
    assert "EXTRACTION_COMPLETED" in {label.code for label in labels}
    assert all(
        set(label.model_dump()) == {"kind", "timestamp", "severity", "code", "subject", "detail"}
        for label in labels
    )


@pytest.mark.parametrize(
    ("value", "expected_code"),
    [("outside", 2), ("Outside", 2), (" outside ", 2), ("park", None)],
)
def test_declared_categorical_codebook_validates_and_encodes_normalized_labels(
    value, expected_code
):
    from nof1_causal_lab.actions.prepare_data.materialization import materialize_panel
    from nof1_causal_lab.actions.prepare_data.worker_output import validate_worker_output
    from nof1_causal_lab.observations.rows import annotate_observation_rows

    preparation = DataPreparationSpec(
        default_window="1d",
        variables=(
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    observation_window=None,
                    id="indicator:place",
                    name="place",
                    measurement_dtype="categorical",
                    aggregation="last",
                    categorical_levels=("home", "work", " Outside "),
                ),
                extraction=SemanticExtractionSpec(how_to_measure="Last stated location"),
            ),
        ),
    )
    from pydantic import TypeAdapter

    from nof1_causal_lab.artifacts.observations.records import ObservationRecord

    context = extraction_context(
        FilePreparationSpec(source=FileSourceRef(files=("source.csv",)), definition=preparation)
    )
    output, errors = validate_worker_output(
        {
            "extractions": [
                {
                    "indicator_id": "indicator:place",
                    "value": value,
                    "window_start": "2026-01-01",
                }
            ]
        },
        context,
        expected_window_starts=["2026-01-01"],
    )
    if expected_code is None:
        assert output is None
        assert len(errors) == 1
        assert "outside the codebook" in errors[0]
        return
    assert errors == []
    assert output is not None
    rows = annotate_observation_rows(
        output.to_dataframe(), preparation.observation_schema()
    ).to_dicts()
    assert materialize_panel(TypeAdapter(list[ObservationRecord]).validate_python(rows), context)[
        "value"
    ].to_list() == [expected_code]
