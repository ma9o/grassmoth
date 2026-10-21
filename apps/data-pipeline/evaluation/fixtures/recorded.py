"""Regenerate a recorded benchmark CSV through public actions, then apply its declared mask."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, TypeAdapter

from evaluation.client import ActionClient
from evaluation.declarations import CaseRecipe, RecordedData
from evaluation.recipes import require_findings
from evaluation.sources import read_source
from nof1_causal_lab.public.contracts import (
    ActionInput,
    ModelTime,
    ObservationInstant,
    QuestionSpec,
    SimulationSpec,
    Value,
)

if TYPE_CHECKING:
    from nof1_causal_lab.public.contracts import SimulateOutput

_REQUEST = TypeAdapter(ActionInput)


class GenerationPlan(Value):
    """Authored question and deterministic simulation controls for the saved dataset."""

    question: QuestionSpec
    simulation: SimulationSpec
    seed: int = Field(ge=0)
    num_replicates: Literal[1]


class ObservationMask(Value):
    """Mask values at grid row indices, preserving all original timestamps and support."""

    description: str
    keep: Mapping[str, tuple[Annotated[int, Field(ge=0)], ...]] = Field(default_factory=dict)
    drop: Mapping[str, tuple[Annotated[int, Field(ge=0)], ...]] = Field(default_factory=dict)


def mask_csv(source: Path, destination: Path, mask: ObservationMask) -> None:
    """Apply the saved missingness declaration without transforming any retained reading."""
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    if fields is None or not (set(mask.keep) | set(mask.drop)) <= set(fields):
        raise ValueError("The mask must name columns in the generated CSV")
    for index, row in enumerate(rows):
        for name, indices in mask.keep.items():
            if index not in indices:
                row[name] = ""
        for name, indices in mask.drop.items():
            if index in indices:
                row[name] = ""
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_csv(result: SimulateOutput, destination: Path) -> None:
    """Export the retained observation history, including its production support and codebooks."""
    evidence = result.report.evidence
    origin = ObservationInstant(evidence.time_origin)

    def instant(day: float | None) -> str:
        return ModelTime(day).at(origin).value.isoformat() if day is not None else ""

    rows: dict[float, dict[str, str | float | int | None]] = {
        day: {"timestamp": instant(day)} for day in evidence.times
    }
    fields = ["timestamp"]
    for variable in evidence.observation_layout.variables:
        history = result.data[0][variable.id]
        fields.extend((variable.name, variable.name + "_start", variable.name + "_end"))
        for day, value, start, end in history.readings():
            reading: str | float | int | None = value
            if value is not None and variable.categorical_levels is not None:
                reading = variable.categorical_levels[int(value)]
            elif value is not None and variable.measurement_dtype == "binary":
                reading = int(value)
            rows[day].update(
                {
                    variable.name: reading,
                    variable.name + "_start": instant(start),
                    variable.name + "_end": instant(end),
                }
            )
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows.values())


def generate(recipe_path: Path, endpoint: str, workspace: str, output: Path) -> None:
    """Author literal truth and simulate one replicate; never fit or run an experiment."""
    output.mkdir(parents=True, exist_ok=False)
    client = ActionClient(endpoint, workspace, output / "calls")
    recipe_root = recipe_path.parent
    recipe = CaseRecipe.model_validate_json(recipe_path.read_text())
    if not isinstance(recipe.data, RecordedData):
        raise TypeError("The recipe must declare its recorded source provenance")
    sources = {Path(source.path).name: source for source in recipe.data.provenance_files}
    content = {
        name: read_source(sources[name], recipe_root)
        for name in ("generation-request.json", "truth.json", "mask.json")
    }
    plan = GenerationPlan.model_validate_json(content["generation-request.json"])
    question = client.call(
        _REQUEST.validate_python(
            {
                "action": "edit_question",
                "input": {"question": plan.question.model_dump(mode="json")},
            }
        )
    )
    truth = client.call(
        _REQUEST.validate_python(
            {
                "action": "edit_model",
                "input": {
                    "parent_ref": client.artifact_revision(question.call_id, "question"),
                    "dynamical_model_spec": json.loads(content["truth.json"]),
                },
            }
        )
    )
    if truth.action != "edit_model":
        raise TypeError("The action response must match the requested action")
    require_findings(
        truth.body.checks.specification, recipe.allowed_findings, recipe.unavailable_findings
    )
    require_findings(
        truth.body.checks.question.findings, recipe.allowed_findings, recipe.unavailable_findings
    )
    if not set(recipe.expected_identified) <= set(truth.body.identification.estimable_treatments):
        raise ValueError("The truth model does not identify the declared treatment")
    truth_revision = client.artifact_revision(truth.call_id, "model")
    request = _REQUEST.validate_python(
        {
            "action": "simulate",
            "input": {
                "dynamical_model_spec_ref": truth_revision,
                "simulation": plan.simulation.model_dump(mode="json"),
                "seed": plan.seed,
                "num_replicates": plan.num_replicates,
            },
        }
    )
    generated = client.call(request)
    if generated.action != "simulate":
        raise TypeError("The action response must match the requested action")
    require_findings(
        generated.body.report.findings, recipe.allowed_findings, recipe.unavailable_findings
    )
    export_csv(generated.body, output / "generated.csv")
    mask_csv(
        output / "generated.csv",
        output / "observations.csv",
        ObservationMask.model_validate_json(content["mask.json"]),
    )
    (output / "generation.json").write_text(
        json.dumps(
            {
                "workspace": workspace,
                "question_revision": client.artifact_revision(question.call_id, "question"),
                "truth_revision": truth_revision,
                "simulate": request.model_dump(mode="json"),
                "simulation_revision": generated.commit_id,
                "simulate_call_id": generated.call_id,
                "hashes": {
                    **{
                        name: hashlib.sha256(payload).hexdigest()
                        for name, payload in content.items()
                    },
                    **{
                        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                        for name in ("generated.csv", "observations.csv")
                    },
                },
                "model_findings": truth.body.checks.model_dump(mode="json"),
                "identification": truth.body.identification.model_dump(mode="json"),
                "simulation_findings": [
                    item.model_dump(mode="json") for item in generated.body.report.findings
                ],
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Write a new regeneration directory against an explicitly chosen local stack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate(args.recipe, args.endpoint, args.workspace, args.output)


if __name__ == "__main__":
    main()
