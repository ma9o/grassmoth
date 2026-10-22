"""Generate synthetic HTTP fixtures using the server's actual contracts and codec.

Run through root codegen. --check detects stale payloads without a running stack,
stored study, scientific execution or another fixture schema. Only the scenario
names are fixture metadata; all requests, timelines and responses use their owners.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from grassmoth.public import EditQuestionOutput, GitOid, ModelDiffOutput, Value
from grassmoth_server.actions.contracts.polling import ActionPoll, FailedPoll, RunningPoll
from grassmoth_server.actions.contracts.requests import (
    DataDiffRequest,
    ModelDiffRequest,
    ScientificActionRequest,
    call_identity,
)
from grassmoth_server.actions.contracts.timeline import TimelineResponse
from grassmoth_server.study.persistence.codec import pack_result, result_payload
from grassmoth_server.study.records import (
    Applied,
    Raised,
    Rejected,
    StudyRevision,
    applied_attempt,
    argument_parents,
    failed_attempt,
    record_dependencies,
)
from scripts.codegen.story_data import (
    EXTRACTION,
    OBSERVATIONS,
    ORIGIN,
    OUTCOME,
    TREATMENT,
    comparison,
    fit,
    model,
    oid,
    preparation,
    simulation,
)

if TYPE_CHECKING:
    from grassmoth_server.actions.contracts.identity import ActionId

type Request = ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid]
REQUEST = TypeAdapter(Request)
POLL = TypeAdapter(ActionPoll)
OUTPUT = (
    Path(__file__).resolve().parents[4]
    / "apps/grassmoth-client/src/components/__fixtures__/generated/study.ts"
)
WORKSPACE = "SYNTHETIC"
QUESTION = {"text": "Synthetic example: how does treatment affect the outcome?", "outcome": OUTCOME}
DESIGN = {
    "start": "2026-01-01",
    "horizon": "3d",
    "interventions": [{"target": TREATMENT, "after": "1d", "value": 2}],
}


@dataclass(frozen=True)
class Example:
    """A named call's input and output; journal metadata derives from these owners."""

    name: str
    request: Request
    poll: ActionPoll
    revision: StudyRevision | None


def examples() -> tuple[Example, ...]:
    """Declare the states the viewer needs without executing any scientific action."""
    entries: list[Example] = []
    revisions: list[StudyRevision] = []

    def add(
        name: str,
        action: ActionId,
        inputs: dict[str, object],
        body: Value | None,
        *,
        artifacts: dict[str, dict[str, str]] | None = None,
        failure: Raised | Rejected | None = None,
        running: bool = False,
    ) -> None:
        seq = len(entries) + 1
        request = REQUEST.validate_python(
            {
                "action": action,
                "input": inputs,
                "reasoning": "Synthetic interface scenario; no scientific action was executed.",
            }
        )
        call_id = call_identity(request)
        if running:
            entries.append(
                Example(name, request, RunningPoll(call_id=call_id, action=action), None)
            )
            return
        messages: list[dict[str, object]] = (
            [{"kind": "failure", "timestamp": ORIGIN, "failure": failure}]
            if failure
            else [
                {
                    "kind": "log",
                    "timestamp": ORIGIN,
                    "severity": "info",
                    "code": "SYNTHETIC_EXAMPLE",
                    "subject": "action",
                    "detail": "Illustrative UI data; no scientific computation was performed.",
                }
            ]
        )
        trace_ids = ("synthetic-authoring",) if name == "authored" else ()
        log_messages = tuple(messages)
        if trace_ids:
            messages.append(
                {
                    "kind": "trace",
                    "timestamp": ORIGIN,
                    "trace_id": trace_ids[0],
                    "trace": {
                        "messages": [
                            {
                                "role": "assistant",
                                "content": "Synthetic authoring conversation for the story trace panel.",
                                "reasoning": "Illustrates the trace UI; no model was called.",
                            }
                        ],
                        "model": "synthetic-example",
                    },
                }
            )
        poll = POLL.validate_python(
            {
                "call_id": call_id,
                "action": action,
                "status": "failed" if failure else "success",
                "commit_id": oid(seq),
                "body": body,
                "messages": messages,
            }
        )
        effects = {
            "produced": [
                {
                    "artifact_id": kind,
                    "revision": oid(seq + (200 if kind == "raw_data" else 100)),
                    "derived_from": pins,
                    "produced_by": action,
                    "created_at": ORIGIN,
                    "source": {"kind": "result", "result": oid(1000 + seq)},
                }
                for kind, pins in (artifacts or {}).items()
            ]
        }
        outcome = Applied[GitOid].model_validate({"result": oid(1000 + seq), "effects": effects})
        attempt = failed_attempt(request, failure) if failure else applied_attempt(request, outcome)
        revision = StudyRevision.model_validate(
            {
                "commit_id": oid(seq),
                "parent_ids": argument_parents(attempt, revisions),
                "record": {
                    "seq": seq,
                    "ts": ORIGIN,
                    "attempt": attempt,
                    "messages": [] if failure else log_messages,
                    "trace_ids": trace_ids,
                },
            }
        )
        revisions.append(revision)
        entries.append(Example(name, request, poll, revision))

    question = EditQuestionOutput.model_validate({"question": QUESTION})
    structure, measured, authored, edited = (
        model(stage) for stage in ("structure", "measurement", "authored", "edited")
    )
    add("question", "edit_question", {"question": QUESTION}, question, artifacts={"question": {}})
    add(
        "structure",
        "edit_model",
        {"parent_ref": oid(101), "dynamical_model_spec": structure.dynamical_model_spec},
        structure,
        artifacts={"model": {"question": oid(101)}},
    )
    add(
        "measurement",
        "edit_model",
        {"parent_ref": oid(102), "dynamical_model_spec": measured.dynamical_model_spec},
        measured,
        artifacts={"model": {"question": oid(101), "model": oid(102)}},
    )
    add(
        "authored",
        "edit_model",
        {"parent_ref": oid(103), "dynamical_model_spec": authored.dynamical_model_spec},
        authored,
        artifacts={"model": {"question": oid(101), "model": oid(103)}},
    )
    add(
        "prepared",
        "prepare_data",
        {
            "dynamical_model_spec_ref": oid(104),
            "source": {
                "files": ["input/synthetic.csv"],
                "hashes": {"input/synthetic.csv": "0" * 64},
                "start": None,
                "end": None,
            },
            "extraction": {str(item["id"]): EXTRACTION for item in OBSERVATIONS},
        },
        preparation(),
        artifacts={"raw_data": {}, "panel": {"model": oid(104), "raw_data": oid(205)}},
    )
    add(
        "editFailed",
        "edit_model",
        {"parent_ref": oid(104), "dynamical_model_spec": {}},
        None,
        failure=Raised(
            error_type="SyntheticProposalError",
            error_message="Synthetic example: model proposal failed before publication.",
            details=("Example execution details for the failure panel.",),
        ),
    )
    target = {
        "kind": "new",
        "dynamical_model_spec_ref": oid(104),
        "data_ref": {"revision": oid(105), "replicate_index": 0},
    }
    add(
        "fitRejected",
        "fit",
        {"target": target, "settings": {"hardware": "local"}, "repeat": 0},
        None,
        failure=Rejected(
            code="SYNTHETIC_REJECTION",
            subject="fit",
            detail="Synthetic example: the selected inputs cannot be fitted.",
        ),
    )
    add(
        "fitted",
        "fit",
        {"target": target, "settings": {"hardware": "local"}, "repeat": 1},
        fit(),
        artifacts={"model": {"question": oid(101), "model": oid(104), "panel": oid(105)}},
    )
    add(
        "simulated",
        "simulate",
        {"dynamical_model_spec_ref": oid(104), "simulation": DESIGN, "num_replicates": 2},
        simulation(causal=True),
    )
    add(
        "edited",
        "edit_model",
        {"parent_ref": oid(104), "dynamical_model_spec": edited.dynamical_model_spec},
        edited,
        artifacts={"model": {"question": oid(101), "model": oid(104)}},
    )
    add(
        "effectUnavailable",
        "simulate",
        {"dynamical_model_spec_ref": oid(110), "simulation": DESIGN, "num_replicates": 2},
        simulation(causal=False),
    )
    add(
        "modelCompared",
        "model_diff",
        {"before_ref": oid(104), "after_ref": oid(110)},
        ModelDiffOutput(
            changes=edited.dynamical_model_spec.changes_from(authored.dynamical_model_spec)
        ),
    )
    compared = comparison()
    add(
        "dataCompared",
        "data_diff",
        {"left_ref": compared.report.left, "right_ref": compared.report.right},
        compared,
    )
    add(
        "fitRunning",
        "fit",
        {
            "target": {**target, "dynamical_model_spec_ref": oid(110)},
            "settings": {"hardware": "local"},
            "repeat": 2,
        },
        None,
        running=True,
    )
    return tuple(entries)


def _typescript(value: object) -> str:
    """Spell validated wire values as typed JavaScript for isolated unit consumers."""
    if isinstance(value, bytes):
        return f"new Uint8Array({json.dumps(list(value), separators=(',', ':'))})"
    if isinstance(value, Mapping):
        return (
            "{"
            + ",".join(
                f"{json.dumps(str(key))}: {_typescript(item)}" for key, item in value.items()
            )
            + "}"
        )
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(_typescript(item) for item in value) + "]"
    return json.dumps(value, allow_nan=False)


def render() -> str:
    """Emit exact HTTP bytes plus typed names, without copied snapshots or result objects."""
    entries = examples()
    revisions = tuple(entry.revision for entry in entries if entry.revision is not None)
    timeline = TimelineResponse.model_validate(
        {
            "attempts": [
                {
                    "call_id": entry.poll.call_id,
                    "commit_id": entry.revision.commit_id,
                    "parent_ids": entry.revision.parent_ids,
                    "record": {
                        "seq": entry.revision.record.seq,
                        "ts": entry.revision.record.ts,
                        "messages": entry.revision.record.messages,
                        "trace_ids": entry.revision.record.trace_ids,
                        "attempt": {
                            "action": entry.request.action,
                            "request": entry.request,
                            "outcome": Applied[None](
                                result=None, effects=entry.revision.record.attempt.outcome.effects
                            )
                            if isinstance(entry.revision.record.attempt.outcome, Applied)
                            else entry.revision.record.attempt.outcome,
                        },
                    },
                }
                for entry in entries
                if entry.revision is not None
            ],
            "dependencies": record_dependencies(revisions),
            "running": [
                {
                    "call_id": entry.poll.call_id,
                    "action": entry.poll.action,
                    "request": entry.request,
                    "messages": entry.poll.messages,
                }
                for entry in entries
                if isinstance(entry.poll, RunningPoll)
            ],
        }
    )
    names = {
        entry.name: (
            "RunningPoll"
            if isinstance(entry.poll, RunningPoll)
            else "FailedPoll"
            if isinstance(entry.poll, FailedPoll)
            else f'Extract<ActionSuccess, {{ action: "{entry.poll.action}" }}>'
        )
        for entry in entries
    }
    responses = [
        f"  {entry.name}: {{ action: {json.dumps(entry.poll.action)}, callId: {json.dumps(str(entry.poll.call_id))}, payload: new Uint8Array({json.dumps(list(pack_result(entry.poll)), separators=(',', ':'))}) }},"
        for entry in entries
    ]
    return "\n".join(
        [
            "/** AUTO-GENERATED by scripts.codegen.export_story_scenarios. Synthetic UI examples only. */",
            'import type { ActionSuccess, FailedPoll, RunningPoll, TimelineResponse } from "@grassmoth/api-types";',
            f"export const scenarioWorkspace = {json.dumps(WORKSPACE)};",
            f"export const scenarioQuestion = {json.dumps(QUESTION['text'])};",
            f"export const scenarioTimeline: TimelineResponse = {timeline.model_dump_json(indent=2)};",
            "export interface ScenarioResults {",
            *(f"  {name}: {value};" for name, value in names.items()),
            "}",
            "export const scenarioResults = {",
            *(f"  {entry.name}: {_typescript(result_payload(entry.poll))}," for entry in entries),
            "} satisfies ScenarioResults;",
            "export const scenarioResponses = {",
            *responses,
            "} as const;",
            "",
        ]
    )


def main(*, check: bool = False) -> int:
    """Write or check the one generated scenario artifact."""
    content = render()
    if check:
        if not OUTPUT.exists() or OUTPUT.read_text() != content:
            print("Story scenarios are stale. Run `bun run codegen`.")
            return 1
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(content)
    print(f"Synthetic story scenarios {'checked' if check else 'generated'}.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    raise SystemExit(main(check=parser.parse_args().check))
