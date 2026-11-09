"""Rebuild the France development study through HTTP, without fitting."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from grassmoth.public import DataRef, GitOid
from grassmoth_server.actions.contracts.identity import ArtifactId, RevisionSelector
from grassmoth_server.actions.contracts.inputs import (
    DataDiffInput,
    EditModelInput,
    ModelDiffInput,
    PrepareDataInput,
    SimulateInput,
)
from grassmoth_server.actions.contracts.polling import ActionResult, ActionSuccess
from grassmoth_server.actions.contracts.preparation import SourceFolder
from grassmoth_server.actions.contracts.requests import (
    ActionInput,
    DataDiffRequest,
    EditModelRequest,
    ModelDiffRequest,
    PrepareDataRequest,
    SimulateRequest,
)
from grassmoth_server.actions.contracts.timeline import TimelineResponse

SOURCE = Path(__file__).with_name("france1940simple")


@dataclass(frozen=True)
class Recipe:
    initial_model: EditModelRequest[RevisionSelector]
    model: EditModelRequest[RevisionSelector]
    preparation: PrepareDataRequest[RevisionSelector, SourceFolder]
    simulation: SimulateRequest[RevisionSelector]

    @classmethod
    def read(cls) -> Recipe:
        return cls(
            initial_model=EditModelRequest[RevisionSelector].model_validate_json(
                (SOURCE / "01-create_model.json").read_bytes()
            ),
            model=EditModelRequest[RevisionSelector].model_validate_json(
                (SOURCE / "02-edit_model.json").read_bytes()
            ),
            preparation=PrepareDataRequest[RevisionSelector, SourceFolder].model_validate_json(
                (SOURCE / "03-prepare_data.json").read_bytes()
            ),
            simulation=SimulateRequest[RevisionSelector].model_validate_json(
                (SOURCE / "04-simulate_prior.json").read_bytes()
            ),
        )


def checked(response: httpx.Response) -> httpx.Response:
    if response.is_error:
        raise RuntimeError(f"{response.request.url}: HTTP {response.status_code}\n{response.text}")
    return response


@dataclass(frozen=True)
class Study:
    client: httpx.Client
    workspace: str
    timeout: float

    @property
    def path(self) -> str:
        return f"/api/studies/{self.workspace}"

    def timeline(self) -> TimelineResponse:
        return TimelineResponse.model_validate_json(
            checked(self.client.get(f"{self.path}/timeline")).content
        )

    def run(self, request: ActionInput) -> ActionSuccess:
        url = f"{self.path}/{request.action}"
        deadline = time.monotonic() + self.timeout
        response = checked(
            self.client.post(
                url,
                content=request.model_dump_json(exclude_unset=True),
                headers={"Content-Type": "application/json"},
            )
        )
        poll = ActionResult(payload=response.content).decode()
        print(f"{request.action}: {poll.call_id}", flush=True)
        while poll.status == "running":
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{request.action} still running; poll {self.client.base_url}{url[1:]}/{poll.call_id}. "
                    "The server continues the call. Rerun with the same --workspace to resume."
                )
            time.sleep(1)
            poll = ActionResult(
                payload=checked(self.client.get(f"{url}/{poll.call_id}")).content
            ).decode()
        if poll.status == "failed":
            raise RuntimeError(
                f"{request.action} failed ({poll.commit_id}):\n"
                + "\n".join(message.model_dump_json() for message in poll.messages)
            )
        print(f"{request.action}: success {poll.commit_id}", flush=True)
        return poll

    def produced(self, result: ActionSuccess, artifact: ArtifactId) -> GitOid:
        entry = next(
            item for item in self.timeline().attempts if item.commit_id == result.commit_id
        )
        outcome = entry.record.attempt.outcome
        if outcome.status != "applied":
            raise RuntimeError(f"Successful call {result.call_id} has no applied timeline entry")
        return next(
            item.revision for item in outcome.effects.produced if item.artifact_id == artifact
        )


def rebuild(study: Study, recipe: Recipe) -> None:
    """Pin every dependency to this run's outputs, including reused calls on resume."""
    with (SOURCE / "historical_inputs.csv").open("rb") as source:
        checked(
            study.client.post(
                "/api/upload",
                data={"workspaceId": study.workspace},
                files={"file": ("historical_inputs.csv", source, "text/csv")},
            )
        )
    initial_model = study.produced(study.run(recipe.initial_model), "model")
    model = study.produced(
        study.run(
            EditModelRequest[RevisionSelector](
                action="edit_model",
                input=EditModelInput[RevisionSelector](
                    parent_ref=initial_model,
                    dynamical_model_spec=recipe.model.input.dynamical_model_spec,
                ),
                reasoning=recipe.model.reasoning,
            )
        ),
        "model",
    )
    preparation = recipe.preparation.input
    panel = study.produced(
        study.run(
            PrepareDataRequest[RevisionSelector, SourceFolder](
                action="prepare_data",
                input=PrepareDataInput[RevisionSelector, SourceFolder](
                    dynamical_model_spec_ref=model,
                    source=preparation.source,
                    extraction=preparation.extraction,
                    context=preparation.context,
                ),
                reasoning=recipe.preparation.reasoning,
            )
        ),
        "panel",
    )
    simulation = recipe.simulation.input
    simulated = study.run(
        SimulateRequest[RevisionSelector](
            action="simulate",
            input=SimulateInput[RevisionSelector](
                dynamical_model_spec_ref=model,
                simulation=simulation.simulation,
                seed=simulation.seed,
                num_replicates=simulation.num_replicates,
            ),
            reasoning=recipe.simulation.reasoning,
        )
    )
    study.run(
        DataDiffRequest[RevisionSelector](
            action="data_diff",
            input=DataDiffInput[RevisionSelector](
                left_ref=(
                    DataRef[RevisionSelector, int | None](revision=panel, replicate_index=0),
                ),
                right_ref=(
                    DataRef[RevisionSelector, int | None](
                        revision=simulated.commit_id, replicate_index=None
                    ),
                ),
            ),
            reasoning="Compare the recorded panel with every prior-predictive history before fitting.",
        )
    )
    study.run(
        ModelDiffRequest[RevisionSelector](
            action="model_diff",
            input=ModelDiffInput[RevisionSelector](before_ref=initial_model, after_ref=model),
            reasoning="Inspect the definitions added to the initial outcome model.",
        )
    )
    timeline = study.timeline()
    print(f"Ready: {study.workspace} ({len(timeline.attempts)} saved actions; no fit submitted).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8100")
    parser.add_argument("--web-url", default="http://localhost:3000")
    parser.add_argument(
        "--workspace",
        default=f"FRANCE1940SIMPLE_{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}",
        help="Fresh study by default; pass an existing rebuild's ID to resume its saved calls.",
    )
    parser.add_argument(
        "--timeout", type=float, default=900, help="Polling limit per action, seconds."
    )
    parser.add_argument(
        "--check", action="store_true", help="Parse the recipe without contacting a server."
    )
    args = parser.parse_args()
    recipe = Recipe.read()
    if args.check:
        print("France recipe matches the current action contracts (no fit).")
        return
    print(f"Rebuilding {args.workspace} through {args.api_url}", flush=True)
    with httpx.Client(
        base_url=args.api_url,
        headers={"Accept": "application/msgpack"},
        timeout=120,
    ) as client:
        rebuild(Study(client=client, workspace=args.workspace, timeout=args.timeout), recipe)
    print(f"Open {args.web_url.rstrip('/')}/workspaces/{args.workspace}")


if __name__ == "__main__":
    main()
