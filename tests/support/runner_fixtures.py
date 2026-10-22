"""Small canonical artifacts for fixture-backed runner contract tests."""

from datetime import UTC, datetime, timedelta

import polars as pl

from grassmoth.artifacts.observations.preparation import (
    PreparedDataMetadata,
    SemanticExtractionSpec,
)
from grassmoth.artifacts.observations.spec import AuthoredObservationSpec
from grassmoth_server.study.state import StudyState
from tests.support.helpers import fixture_entity_id
from tests.support.model import stress_sleep_model


def panel_frame(n_days=20):
    start = datetime(2024, 1, 1)
    return pl.DataFrame(
        [
            {
                "indicator_id": fixture_entity_id("indicator", indicator),
                "value": value,
                "anchor_time": (start + timedelta(days=day + 1)).isoformat(),
                "support_start": (start + timedelta(days=day)).isoformat(),
                "support_end": (start + timedelta(days=day + 1)).isoformat(),
                "support_kind": "interval",
                "summary_operator": "mean",
                "anchor_policy": "support_end",
                "observation_window": "1d",
            }
            for day in range(n_days)
            for indicator, value in (
                ("stress_score", float(1 + day % 5)),
                ("sleep_score", float(8 - day % 4)),
            )
        ]
    )


def state_from(*infos):
    return StudyState().with_artifacts(list(infos))


def seed_model(store):
    return store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": stress_sleep_model().model_dump(mode="json")},
    )


def seed_panel(store, *, model_revision):
    return store.write_artifact(
        "panel",
        derived_from={},
        json_files={"metadata.json": panel_metadata().model_dump(mode="json")},
        produced_by="prepare_data",
        parquet_files={"panel.parquet": panel_frame()},
    )


def panel_metadata():
    from grassmoth.artifacts.observations.preparation import (
        DataPreparationSpec,
        DataVariableSpec,
    )

    preparation = DataPreparationSpec(
        default_window="1d",
        variables=tuple(
            DataVariableSpec(
                observation=AuthoredObservationSpec(
                    observation_window=None,
                    id=fixture_entity_id("indicator", name),
                    name=name,
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
                extraction=SemanticExtractionSpec(how_to_measure="Read " + name),
            )
            for name in ("stress_score", "sleep_score")
        ),
    )
    return PreparedDataMetadata(
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
        preparation=preparation,
    )
