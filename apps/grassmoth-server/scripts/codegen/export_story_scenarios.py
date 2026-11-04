"""Export France's saved happy path as native HTTP responses, without a running stack."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "apps/grassmoth-client/src/components/__fixtures__/generated"
WORKSPACE = "FRANCE1940SIMPLE"


async def export() -> dict[Path, bytes]:
    """Read the normal timeline and completed-call bytes from the local study repository.

    The happy path is each action's first successful call; the last successful fit is the
    full fit. Other calls in the study are left out.
    """
    os.environ["GRASSMOTH_DATA_ROOT"] = str(ROOT / "data")
    os.environ["READ_ONLY_FACADE"] = "1"
    from grassmoth_server.actions.contracts.polling import StudyError
    from grassmoth_server.study.calls import TemporalClientProvider, get_timeline
    from grassmoth_server.study.persistence.history import StudyRepository
    from grassmoth_server.study.queries.action_outputs import completed_call_msgpack
    from grassmoth_server.study.records import Applied

    history = ROOT / "data" / WORKSPACE / "history.git"
    if not history.is_dir():
        raise FileNotFoundError(f"Story codegen requires the local France history: {history}")
    repository = StudyRepository(WORKSPACE)
    applied = tuple(
        revision
        for revision in repository.attempts()
        if isinstance(revision.record.attempt.outcome, Applied)
    )
    first = {revision.record.attempt.action: revision for revision in reversed(applied)}
    last = {revision.record.attempt.action: revision for revision in applied}
    calls = {
        "question": first["edit_question"],
        "authored": first["edit_model"],
        "prepared": first["prepare_data"],
        "simulated": first["simulate"],
        "dataCompared": first["data_diff"],
        "pilot": first["fit"],
        "fitted": last["fit"],
    }
    timeline = await get_timeline(WORKSPACE, TemporalClientProvider())
    if isinstance(timeline, StudyError):
        raise RuntimeError(timeline.detail)
    selected = {revision.record.seq for revision in calls.values()}
    timeline = timeline.revised(
        attempts=tuple(item for item in timeline.attempts if item.record.seq in selected),
        dependencies=tuple(
            edge
            for edge in timeline.dependencies
            if edge.seq in selected and edge.source_seq in selected
        ),
    )
    files: dict[Path, bytes] = {}
    responses: list[str] = []
    entries = {entry.record.seq: entry for entry in timeline.attempts}
    for name, revision in calls.items():
        entry = entries[revision.record.seq]
        action = revision.record.attempt.action
        filename = f"{revision.record.seq:02d}-{action}.msgpack"
        files[Path("responses") / filename] = completed_call_msgpack(WORKSPACE, revision)
        responses.append(
            f"  {name}: {{ action: {json.dumps(action)}, callId: {json.dumps(entry.call_id)}, "
            f"label: {json.dumps(f'{action} · {revision.commit_id[:7]}')}, "
            f"url: {json.dumps('/story-responses/' + filename)} }},"
        )
    files[Path("study.ts")] = "\n".join(
        [
            "/** AUTO-GENERATED from FRANCE1940SIMPLE's saved HTTP responses. */",
            'import type { TimelineResponse } from "@grassmoth/api-types";',
            f"export const scenarioWorkspace = {json.dumps(WORKSPACE)};",
            f"export const scenarioTimeline: TimelineResponse = {timeline.model_dump_json(indent=2)};",
            "export const scenarioResponses = {",
            *responses,
            "} as const;",
            "",
        ]
    ).encode()
    return files


def main() -> int:
    """Write native response assets and their small typed timeline manifest."""
    files = asyncio.run(export())
    size = sum(len(payload) for payload in files.values())
    if size > 50_000_000:
        raise RuntimeError(
            f"France's generated fixture is {size:,} bytes; stop at the brief's 50 MB limit"
        )
    for relative, payload in files.items():
        path = OUTPUT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    expected = {OUTPUT / relative for relative in files}
    for path in (OUTPUT / "responses").glob("*.msgpack"):
        if path not in expected:
            path.unlink()
    print(f"France story responses generated: {size:,} bytes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
