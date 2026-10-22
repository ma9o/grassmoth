# Running benchmarks

The [benchmark workflow](../../benchmarks/README.md#running)
owns execution, compute limits and case ordering. Each stage contains public HTTP
request bodies and `notes.md`; there is no separate benchmark runner.

Read the [integration testing guide](agentic_integration_testing.md) before
starting a stack. Use the generated
[study API skill](../../.agents/skills/grassmoth-study-api/SKILL.md) to submit
requests and poll saved calls. Stage 00's
[notes](../../benchmarks/00_current_sampler/notes.md#http-request-files)
identify its revision placeholders and sparse-data upload format.

Resolve request placeholders to the required produced artifacts' `revision`
values from the successful calls' timeline entries
(`record.attempt.outcome.effects.produced`). The action's `commit_id` names its
journal entry, not an input artifact. Save the concrete request bodies with the run in
`scratchpad/`. Submit a body to its named action:

```bash
curl -sS "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/fit" \
  -X POST -H 'Content-Type: application/json' \
  --data-binary @scratchpad/benchmarks/RUN/04-fit-C0-A0.json \
  -o scratchpad/benchmarks/RUN/fit.msgpack
```

Poll the returned call ID as the skill describes; POST and polling responses
are MessagePack. The server retains the scientific result and execution log.
Record the study workspace, call IDs, outcomes and proposed decision in the
stage's `notes.md`, and retain billing evidence with the run outputs.

Identical resolved requests reuse saved results, including failures. Increment
`input.repeat` for a deliberate timing repeat. A patched sampler needs its own
checkout, server and study because code version is excluded from call identity.
The [sampler contract](../../packages/grassmoth/src/grassmoth/artifacts/sampling.py)
owns numerical settings and defaults; missing capabilities belong at their
framework owners and are exercised through the same public actions.
