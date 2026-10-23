---
name: grassmoth-study-api
description: "Run and inspect grassmoth studies through the HTTP API with curl. Use for model editing, data preparation, fitting, simulation and saved results."
---

# grassmoth study API

> Auto-generated from `packages/api-types/schemas/openapi.json` (the FastAPI OpenAPI spec) by `apps/grassmoth-server/scripts/codegen/export_api.py`. Edit Python type/field descriptions, the API description or route docstrings, not this file.

Seven actions carry out a study. Set the question with `edit_question` and build
the model with `edit_model`. Upload tables with `/api/upload`, extract
observations with `prepare_data`, and condition the model on them with `fit`.
`simulate` samples an authored or fitted model; `data_diff` and `model_diff`
compare saved revisions. `/api/docs` has complete input/output schemas, defaults
and alternatives.

- **Submit and poll:** POST JSON `{action, input, reasoning}` to
  `/api/studies/{workspace_id}/{action}`; `reasoning` is optional. Poll GET
  `/api/studies/{workspace_id}/{action}/{call_id}` until `status` is `success` or
  `failed`. GET never starts or retries work; an unknown call returns 404.
- **Results and errors:** `body` contains the action's scientific output only on
  success; otherwise it is null. `messages` holds progress, traces and failure
  details. Completed results are saved and GET never recomputes them.
- **JSON responses:** send `Accept: application/json` on action POST/GET to read
  results with curl and jq. Numerical values are nested lists; non-finite values
  are null. Workspace, timeline, upload and HTTP error responses are also JSON.
- **Call reuse:** revision selectors accept exact Git hashes or `"latest"`, pinned
  once at POST. Identical resolved inputs reuse running calls, successes **and
  failures**. Changing `reasoning` does not create a new call; change the inputs
  to make a new attempt.
- **Artifact references:** response `commit_id` identifies the call's journal entry.
  For subsequent inputs, GET `/api/studies/{workspace_id}/timeline`, find the call,
  and select the required artifact by `artifact_id` in
  `record.attempt.outcome.effects.produced`. Use its `revision` for model,
  question, data and fit references.
- **Read-only access:** `/api/workspaces` reports `X-Actions-Enabled`. With
  `READ_ONLY_FACADE=1`, saved calls remain readable; an unsaved POST returns 403.

## References

- The [study API reference](../../../docs/reference/study-api.md) lists every endpoint's input and success-body fields with an example request. Read an action's section before calling it.
- Each action's flow chart shows its checks, branches and failure outcomes: [`edit_question`](../../../docs/assets/action-flows/edit-question.svg), [`edit_model`](../../../docs/assets/action-flows/edit-model.svg), [`prepare_data`](../../../docs/assets/action-flows/prepare-data.svg), [`fit`](../../../docs/assets/action-flows/fit.svg), [`simulate`](../../../docs/assets/action-flows/simulate.svg), [`data_diff`](../../../docs/assets/action-flows/data-diff.svg), [`model_diff`](../../../docs/assets/action-flows/model-diff.svg).
- [Assumptions and limits](../../../docs/assumptions.md) state the modeling commitments behind every result.
- The [likelihood](../../../docs/reference/statistical-model-spec/likelihoods.md) and [parameter and prior](../../../docs/reference/statistical-model-spec/parameters.md) references describe parts of the `dynamical_model_spec` language.
