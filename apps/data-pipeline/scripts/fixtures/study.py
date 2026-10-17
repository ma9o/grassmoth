"""Build HEALTHDEMO and workbench fixtures from the saved bundle, or project a study.

``bun run fixture:build`` regenerates Storybook fixtures from the authoritative Git
bundle and blobs.
The ``project`` subcommand supports workspace promotion.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from unittest.mock import patch

from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.model.expressions import coefficient
from nof1_causal_lab.artifacts.model.expressions import state as expr_state
from nof1_causal_lab.llm.traces import LLMTrace
from nof1_causal_lab.models.inputs import input_fingerprints
from nof1_causal_lab.storage import paths as data_module
from nof1_causal_lab.storage.traces import trace_log_path
from nof1_causal_lab.study.artifact_files import artifact_file_spec
from nof1_causal_lab.study.persistence.history import StudyRepository
from nof1_causal_lab.study.persistence.store import ArtifactStore
from nof1_causal_lab.study.state import is_stale
from scripts.fixtures.reader import ModelReader

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId
    from nof1_causal_lab.study.state import StudyState

ROOT = Path(__file__).resolve().parents[4]
HEALTHDEMO_ROOT = ROOT / "data" / "HEALTHDEMO"
WORKBENCH_OUTPUT = ROOT / "apps/web/src/components/__fixtures__/workbench-comparisons.json"
SIMULATION_REPORTS = ROOT / "apps/web/src/components/dag/__fixtures__/simulation-reports.json"

ARTIFACTS: dict[ArtifactId, str] = {
    "model": "model.json",
}
TRACES = {
    "raw_data": "raw-data",
    "latent_structure": "latent-structure",
    "measurement_structure": "measurement-structure",
    "measurements": "measurement-chunk-",
    "statistical_model_spec": "model-spec-",
}


def read_fixture_files(repository: StudyRepository, state: StudyState) -> dict[str, bytes]:
    """Project retained payloads and logs from the selected Git ancestry."""
    records = repository.records(repository.head())
    store = ArtifactStore(repository.workspace_id)
    files = {}
    for aid, filename in ARTIFACTS.items():
        files[f"artifacts/{aid}.json"] = json.dumps(
            store.read_json_file(aid, state.current[aid].revision, filename)
        ).encode()
    for name, prefix in TRACES.items():
        trace, record = next(
            (trace, record)
            for record in reversed(records)
            for trace in sorted(record.record.trace_ids)
            if trace.startswith(prefix)
        )
        raw_trace = repository.read_file(record.commit_id, f"logs/{trace_log_path(trace)}")
        files[f"traces/{name}.json"] = (
            LLMTrace.model_validate_json(raw_trace).model_dump_json(indent=2) + "\n"
        ).encode()
    reader = ModelReader(repository.workspace_id, at=repository.head())
    identification = reader.identification()
    if identification is not None:
        files["artifacts/identification_report.json"] = (
            identification.model_dump_json(indent=2) + "\n"
        ).encode()
    if reader.fit_checks is not None:
        files["artifacts/reports/data_quality.json"] = (
            reader.fit_checks.model_dump_json(indent=2) + "\n"
        ).encode()
    report = reader.inference_report
    files["inference.json"] = (
        (report.model_dump_json(indent=2) + "\n").encode() if report is not None else b"null\n"
    )
    files["predictive_checks.json"] = b"null\n"
    # Published artifacts select their canonical result; only uploaded inputs own files.
    from nof1_causal_lab.study.state import ArtifactFiles

    for aid, info in state.current.items():
        if isinstance(info.source, ArtifactFiles):
            spec = artifact_file_spec(aid)
            for filename in (*spec.json_files.values(), *spec.parquet_files.values()):
                store.file_path(aid, info.revision, filename)
    return files


def project(source: Path, destination: Path | None = None):
    data_module._DATA_URI = str(source.parent)
    repository = StudyRepository(source.name)
    state = repository.state(repository.head())
    required: tuple[ArtifactId, ...] = (
        "raw_data",
        "model",
        "panel",
    )
    missing = [aid for aid in required if not state.has(aid)]
    if missing:
        raise ValueError(
            f"Source workspace is incomplete; missing current artifacts: {', '.join(missing)}."
        )
    stale = [aid for aid in required if is_stale(state, aid)]
    if stale:
        raise ValueError(f"Source workspace has stale current artifacts: {', '.join(stale)}.")
    files = read_fixture_files(repository, state)
    if destination is not None:
        for name, content in files.items():
            path = destination / "fixture" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        subprocess.run(
            [
                "git",
                "--git-dir",
                str(destination / "study" / "history.git"),
                "bundle",
                "create",
                str(destination / "study" / "history.bundle"),
                "--all",
            ],
            check=True,
            capture_output=True,
        )
    return {"artifacts": list(ARTIFACTS), "traces": list(TRACES)}


def workbench_comparisons(snapshot, history):
    dynamical_model_spec = DynamicalModelSpec.model_validate(snapshot["dynamical_model_spec"])
    parameter = next(
        item
        for item in dynamical_model_spec.parameters
        if item.name == "beta_perceived_stress_burden_internalizing_symptom_burden"
    )
    edge = next(
        item for item in dynamical_model_spec.edges if item.id == "edge:9df1507c29b9de944a33"
    )
    pinned_dynamical_model_spec = dynamical_model_spec.with_entities(
        edges=tuple(
            item.revised(
                mechanisms=tuple(
                    mechanism.revised(
                        expression=coefficient(0.0, "weight") * expr_state(edge.cause.id)
                    )
                    for mechanism in item.mechanisms
                )
            )
            if item.id == edge.id
            else item
            for item in dynamical_model_spec.edges
        ),
        parameters=tuple(
            item for item in dynamical_model_spec.parameters if item.id != parameter.id
        ),
        distributions={
            key: law
            for key, law in dynamical_model_spec.distributions.items()
            if key != parameter.distribution
        },
    )
    dynamical_model_specs: dict[str, DynamicalModelSpec | None] = {
        history[str(seq)]["state"]["current"]["model"][
            "revision"
        ]: DynamicalModelSpec.model_validate(history[str(seq)]["dynamical_model_spec"])
        for seq in (2, 3, 4, 7)
    }
    dynamical_model_specs["no-model"] = None
    dynamical_model_specs.update(
        {
            format(n, "x").rjust(40, "a"): candidate
            for n, candidate in [
                (5, dynamical_model_spec),
                (6, dynamical_model_spec),
                (7, pinned_dynamical_model_spec),
            ]
        }
    )
    empty = DynamicalModelSpec.from_entities()
    comparisons = {
        f"{before_version}:{after_version}": {
            "changes": (empty if right is None else right)
            .changes_from(empty if left is None else left)
            .model_dump(mode="json")
        }
        for before_version, left in dynamical_model_specs.items()
        for after_version, right in dynamical_model_specs.items()
    }
    return {
        "pinned_dynamical_model_spec": pinned_dynamical_model_spec.model_dump(mode="json"),
        "pinned_inputs": input_fingerprints(pinned_dynamical_model_spec),
        "comparisons": comparisons,
    }


def build_outputs():
    # A clean restore proves the fixture does not depend on a developer's local
    # repository, generated JSON projections or retired numbered directories.
    with (
        TemporaryDirectory(prefix="nof1-model-fixture-") as directory,
        patch.object(data_module, "_DATA_URI", directory),
    ):
        workspace = Path(directory) / "HEALTHDEMO"
        history = workspace / "study/history.git"
        history.parent.mkdir(parents=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--mirror",
                str(HEALTHDEMO_ROOT / "study/history.bundle"),
                str(history),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "--git-dir", str(history), "config", "nof1.format", "28"],
            check=True,
            capture_output=True,
        )
        shutil.copytree(HEALTHDEMO_ROOT / "store", workspace / "store")
        repository = StudyRepository("HEALTHDEMO")
        reader = ModelReader("HEALTHDEMO", at=repository.head())
        commits = {0: reader.records[0].parent_ids[0]}
        commits.update({record.record.seq: record.commit_id for record in reader.records})
        outputs = {
            HEALTHDEMO_ROOT / "fixture" / name: json.loads(content)
            for name, content in read_fixture_files(repository, reader.state).items()
        }
        outputs.update(
            {
                HEALTHDEMO_ROOT / "fixture/model_snapshot.json": reader.snapshot().model_dump(
                    mode="json"
                ),
                HEALTHDEMO_ROOT / "fixture/model_history.json": {
                    str(seq): ModelReader("HEALTHDEMO", at=commit)
                    .snapshot()
                    .model_dump(mode="json")
                    for seq, commit in commits.items()
                },
            }
        )
        outputs[WORKBENCH_OUTPUT] = workbench_comparisons(
            outputs[HEALTHDEMO_ROOT / "fixture/model_snapshot.json"],
            outputs[HEALTHDEMO_ROOT / "fixture/model_history.json"],
        )
        from nof1_causal_lab.study.persistence.codec import result_payload
        from scripts.fixtures.visuals import restore_fixture, workbench_visuals

        outputs[WORKBENCH_OUTPUT.with_name("workbench-visuals.json")] = workbench_visuals(
            reader, json.loads(WORKBENCH_OUTPUT.with_name("workbench-simulation.json").read_text())
        )
        from nof1_causal_lab.artifacts.simulation import SimulationReport

        outputs[SIMULATION_REPORTS] = [
            result_payload(SimulationReport.model_validate(value))
            for value in restore_fixture(json.loads(SIMULATION_REPORTS.read_text()))
        ]
        return outputs


def rendered_fixtures(outputs):
    """Render fixtures and name their contracts; JSON imports otherwise widen tags/IDs.

    These declarations contain no payloads or shadow schemas. Fixture regeneration
    establishes the types at their Python producers.
    """
    contracts = {
        HEALTHDEMO_ROOT / "fixture/model_snapshot.json": "Domain.ModelSnapshot",
        HEALTHDEMO_ROOT
        / "fixture/model_history.json": "Readonly<Partial<Record<number, Domain.ModelSnapshot>>>",
        HEALTHDEMO_ROOT / "fixture/inference.json": "Domain.InferenceReport | null",
        HEALTHDEMO_ROOT
        / "fixture/predictive_checks.json": "Domain.PosteriorPredictiveChecks | null",
        SIMULATION_REPORTS: "readonly Domain.SimulationReport[]",
        WORKBENCH_OUTPUT: """Readonly<{
  pinned_dynamical_model_spec: Domain.DynamicalModelSpec;
  pinned_inputs: Record<string, string>;
  comparisons: Readonly<Partial<Record<string, Domain.ModelDiffOutput>>>;
}>""",
        WORKBENCH_OUTPUT.with_name("workbench-visuals.json"): """Readonly<{
  note: string;
  simulation: Domain.SimulateOutput;
  observations: Readonly<Partial<Record<Domain.IndicatorId, Domain.ObservationHistory>>>;
}>""",
    }
    contracts.update(
        {HEALTHDEMO_ROOT / f"fixture/traces/{name}.json": "Domain.LLMTrace" for name in TRACES}
    )
    from scripts.fixtures.visuals import render_fixture

    contents = {
        path: json.dumps(render_fixture(value), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
        for path, value in outputs.items()
    }
    for path, contract in contracts.items():
        assert path in outputs
        declaration = path.with_name(path.stem + ".d.json.ts")
        contents[declaration] = (
            "/** AUTO-GENERATED by fixture:build from the production fixture owner. */\n"
            'import type * as Domain from "@nof1-causal-lab/api-types";\n'
            'import type { BinaryFixture } from "@/components/__fixtures__/fixture-value";\n'
            f"declare const value: BinaryFixture<{contract}>;\nexport default value;\n"
        )
    for path, content in contents.items():
        if path == SIMULATION_REPORTS or path.suffix == ".ts":
            content = subprocess.run(
                [
                    "bun",
                    "x",
                    "--no-install",
                    "biome",
                    "format",
                    "--files-max-size",
                    str(len(content.encode())),
                    "--stdin-file-path",
                    str(path),
                ],
                input=content,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
        yield path, content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "build", help="Generate HEALTHDEMO and workbench fixtures from the Git bundle"
    )
    projection = commands.add_parser(
        "project", help="Validate and project a workspace for promotion"
    )
    projection.add_argument("source", type=Path)
    projection.add_argument("destination", type=Path, nargs="?")
    args = parser.parse_args()
    if args.command == "project":
        print(json.dumps(project(args.source.resolve(), args.destination)))
        return

    outputs = build_outputs()
    for path, rendered in rendered_fixtures(outputs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    print(f"Restored the Git bundle and composed {len(outputs)} HEALTHDEMO and workbench fixtures")


if __name__ == "__main__":
    main()
