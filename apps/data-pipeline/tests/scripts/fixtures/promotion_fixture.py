"""Small facts-only Git study for fixture promotion; no numerical execution."""

import json
import sys
from pathlib import Path

import polars as pl

from nof1_causal_lab.actions.contracts.effects import ActionEffects
from nof1_causal_lab.artifacts.model.dynamical_model_spec import ModelEditResult
from nof1_causal_lab.artifacts.observations.preparation import DataPreparationResult
from nof1_causal_lab.llm.traces import LLMTrace, TraceMessage
from nof1_causal_lab.storage import paths as data_module
from nof1_causal_lab.study.persistence.history import StudyRepository
from nof1_causal_lab.study.persistence.store import ArtifactStore
from nof1_causal_lab.study.records import Applied
from nof1_causal_lab.study.state import RetractedArtifact
from tests.integration.runner_fixtures import panel_frame
from tests.support.action import applied_record, question_root
from tests.support.data import metadata_for_model
from tests.support.helpers import fixture_entity_id, make_model


def seed(root, workspace, options):
    data_module._DATA_URI = root
    store, repository = ArtifactStore(workspace), StudyRepository(workspace)
    stamp = "2026-08-07T00:00:00Z"
    directory = Path(root) / workspace
    (directory / "access.json").write_text('{"version": 1}')
    for name in ("input/bundle.zip", "scratch/discard.json", "cache/discard.json"):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture input")
    question_root(workspace)
    # Incomplete authored equations keep this projection contract non-numerical.
    dynamical_model_spec = make_model(
        ["stress_score", "sleep_score"], [("stress_score", "sleep_score")]
    )
    frame = panel_frame(n_days=2).with_columns(
        pl.col("indicator_id").replace_strict(
            {
                fixture_entity_id("indicator", name): next(
                    indicator.observation.id
                    for indicator in dynamical_model_spec.indicators
                    if indicator.observation.name == name + "_obs"
                )
                for name in ("stress_score", "sleep_score")
            }
        ),
        pl.col("anchor_time", "support_start", "support_end").str.to_datetime(),
    )
    old_raw = store.write_artifact(
        "raw_data",
        derived_from={},
        produced_by="prepare_data",
        parquet_files={"raw.parquet": frame},
    )
    raw = store.write_artifact(
        "raw_data",
        derived_from={},
        produced_by="prepare_data",
        parquet_files={"raw.parquet": frame.with_columns((frame["value"] + 1).alias("value"))},
    )
    definition = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": dynamical_model_spec.model_dump(mode="json", round_trip=True)},
    )
    panel = store.write_artifact(
        "panel",
        produced_by="prepare_data",
        derived_from={"raw_data": old_raw.revision if options.get("stalePanel") else raw.revision},
        json_files={
            "metadata.json": metadata_for_model(dynamical_model_spec).model_dump(
                mode="json", round_trip=True
            )
        },
        parquet_files={"panel.parquet": frame},
    )

    def append(operation, produced, traces=(), retracted=()):
        applied = Applied(
            result=DataPreparationResult() if operation == "prepare_data" else ModelEditResult(),
            effects=ActionEffects(produced=tuple(produced), retracted=tuple(retracted)),
        )
        repository.append(
            applied_record(
                workspace,
                applied,
                seq=repository.latest_seq() + 1,
                ts=stamp,
                trace_ids=[trace for _, trace in traces],
            ),
            logs={
                f"traces/{trace}.json": LLMTrace(
                    model="fixture",
                    messages=(TraceMessage(role="assistant", content=f"{label}: {trace}"),),
                )
                .model_dump_json(round_trip=True)
                .encode()
                for label, trace in traces
            },
        )

    append(
        "edit_model",
        (definition,),
        (
            ("latent_structure", "latent-structure"),
            ("measurement_structure", "measurement-structure"),
            ("statistical_model_spec", "model-spec-sleep-attempt-001"),
        ),
    )
    append(
        "prepare_data",
        (raw, panel),
        (("raw_data", "raw-data"), ("measurements", "measurement-chunk-000000-attempt-001")),
    )
    if omitted := options.get("omit"):
        append(
            "edit_model",
            (),
            retracted=(RetractedArtifact(artifact_id=omitted, reason_ref="fixture-omission"),),
        )


if __name__ == "__main__":
    seed(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
