# grassmoth study API reference

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

## Endpoints

Example requests come from the [synthetic interface scenarios](../../apps/grassmoth-server/scripts/codegen/story_data.py). Replace their revision hashes with `latest` or revisions from your study's timeline.

### POST `/api/studies/{workspace_id}/data_diff`

Compare input.left_ref and input.right_ref, each a nonempty list of {revision, replicate_index} references. Omit the index to compare all recorded histories. Data latest selects prepared user data; simulations require exact Git hashes. Retain the complete comparison in Git. GET polls call_id. Identical resolved calls reuse successes and failures; the comparison is the body.

**Parameters**

- `workspace_id` (path, required)

**Input: [`DataDiffInput<RevisionSelector>`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/inputs.py)**

Compare saved histories statistically; left/right do not imply before/after revisions.

DataComparisonReport owns selection and comparison semantics. The action reads
retained observations without generating new draws or changing scientific state.

| Field | Type | Presence / default | Description |
|---|---|---|---|
| `left_ref` | [`DataSelection<RevisionSelector>`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/inputs.py) | required | References on the left side. Each reference selects one replicate by index, or all its retained histories when the index is omitted. |
| `right_ref` | [`DataSelection<RevisionSelector>`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/inputs.py) | required | References on the right side, using the same selection rule. |

**Success body: [`DataDiffOutput`](../../packages/grassmoth/src/grassmoth/actions/contracts/bodies.py)**

The computed comparison report; original histories remain in their producing results.

| Field | Type | Presence / default | Description |
|---|---|---|---|
| `report` | [`DataComparisonReport`](../../packages/grassmoth/src/grassmoth/artifacts/reports/data_comparison.py) | required | Exact history selections, point changes, per-history statistics, compatibility findings and predictive checks. |

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/data_diff" -H 'Accept: application/json' -o /tmp/action.json \
  -X POST \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "action": "data_diff",
  "input": {
    "left_ref": [
      {
        "revision": "0000069000000000000000000000000000000000",
        "replicate_index": 0
      }
    ],
    "right_ref": [
      {
        "revision": "0000009000000000000000000000000000000000",
        "replicate_index": 0
      },
      {
        "revision": "0000009000000000000000000000000000000000",
        "replicate_index": 1
      }
    ]
  }
}
JSON
```

### POST `/api/studies/{workspace_id}/edit_model`

Merge input.dynamical_model_spec into input.parent_ref: start empty for a question revision, or retain omitted fields from a model revision and its pinned question. Null entity entries delete their IDs. Prune constructs outside the outcome ancestry with warnings, then validate and evaluate data-independent model checks. The body contains the produced model and its findings; messages retain all execution logging. GET polls the returned call_id.

**Parameters**

- `workspace_id` (path, required)

**Input: [`EditModelInput<RevisionSelector>`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/inputs.py)**

Create or revise a model from a question or model parent.

| Field | Type | Presence / default | Description |
|---|---|---|---|
| `parent_ref` | [`RevisionSelector`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/identity.py) | required | Question or model revision to start from. A model parent supplies its pinned question. 'latest' selects the current model, otherwise the current question. An exact Git hash can select either parent without being the current head. |
| `dynamical_model_spec` | [`DynamicalModelSpec`](../../packages/grassmoth/src/grassmoth/artifacts/model/dynamical_model_spec.py) | required | Scientific definitions merged by identity into a model parent, or into an empty model for a question parent. Omitted fields are retained; null entity entries delete their identities. The action retains the merged spec and its findings. |

**Success body: [`EditModelOutput`](../../packages/grassmoth/src/grassmoth/actions/contracts/bodies.py)**

The produced model and its recorded scientific findings.

| Field | Type | Presence / default | Description |
|---|---|---|---|
| `dynamical_model_spec` | [`DynamicalModelSpec`](../../packages/grassmoth/src/grassmoth/artifacts/model/dynamical_model_spec.py) | required | Saved scientific definition. |
| `checks` | [`ModelCheckReport`](../../packages/grassmoth/src/grassmoth/artifacts/reports/model_checks.py) | required | Combined findings retained by the action that produced the model. |
| `identification` | [`IdentificationReport`](../../packages/grassmoth/src/grassmoth/artifacts/reports/identification.py) | required | Causal identification findings. |
| `pruning` | [`ModelEditResult`](../../packages/grassmoth/src/grassmoth/artifacts/model/dynamical_model_spec.py) | required | Entities removed because they do not serve the question. |

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/edit_model" -H 'Accept: application/json' -o /tmp/action.json \
  -X POST \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "action": "edit_model",
  "input": {
    "parent_ref": "0000065000000000000000000000000000000000",
    "dynamical_model_spec": {
      "constructs": {
        "construct:00000000000000000001": {
          "name": "baseline",
          "description": "Synthetic baseline for interface examples.",
          "indicators": {},
          "dynamics": {},
          "coefficients": [],
          "innovation_family": "gaussian",
          "distribution": null,
          "role": "endogenous",
          "temporal_status": "time_invariant"
        },
        "construct:00000000000000000002": {
          "name": "treatment",
          "description": "Synthetic treatment for interface examples.",
          "indicators": {},
          "dynamics": {},
          "coefficients": [],
          "innovation_family": "gaussian",
          "distribution": null,
          "role": "endogenous",
          "temporal_status": "time_varying"
        },
        "construct:00000000000000000003": {
          "name": "outcome",
          "description": "Synthetic outcome for interface examples.",
          "indicators": {},
          "dynamics": {},
          "coefficients": [],
          "innovation_family": "gaussian",
          "distribution": null,
          "role": "endogenous",
          "temporal_status": "time_varying"
        }
      },
      "edges": {
        "edge:00000000000000000001": {
          "mechanisms": {},
          "description": "Baseline affects treatment.",
          "sources": [],
          "cause": "construct:00000000000000000001",
          "effect": "construct:00000000000000000002"
        },
        "edge:00000000000000000002": {
          "mechanisms": {},
          "description": "Treatment affects the outcome.",
          "sources": [],
          "cause": "construct:00000000000000000002",
          "effect": "construct:00000000000000000003"
        }
      },
      "parameters": {},
      "distributions": {},
      "law_layouts": {},
      "measurement_clock": null
    }
  }
}
JSON
```

### POST `/api/studies/{workspace_id}/edit_question`

Set the study question from input.question. POST returns a call_id; GET polls it. Identical resolved calls reuse both successes and failures.

**Parameters**

- `workspace_id` (path, required)

**Input: [`EditQuestionInput`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/inputs.py)**

The study question, its required outcome, and any named intervention queries.

| Field | Type | Presence / default | Description |
|---|---|---|---|
| `question` | [`QuestionSpec`](../../packages/grassmoth/src/grassmoth/artifacts/model/question.py) | required | The replacement study question, including the required outcome and any intervention queries to assess against candidate models. |

**Success body: [`EditQuestionOutput`](../../packages/grassmoth/src/grassmoth/actions/contracts/bodies.py)**

The question saved by this action.

| Field | Type | Presence / default | Description |
|---|---|---|---|
| `question` | [`QuestionSpec`](../../packages/grassmoth/src/grassmoth/artifacts/model/question.py) | required | The saved question, including its outcome and intervention queries. |

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/edit_question" -H 'Accept: application/json' -o /tmp/action.json \
  -X POST \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "action": "edit_question",
  "input": {
    "question": {
      "text": "Synthetic example: how does treatment affect the outcome?",
      "outcome": "construct:00000000000000000003",
      "queries": {}
    }
  }
}
JSON
```

### POST `/api/studies/{workspace_id}/fit`

Fit target.new model and data, or continue target.fit_ref at its last retained state. Counts request additional draws; zero warmup freezes tuning and algorithm changes require warmup. Zero warmup and zero samples compile both sweep programs without execution. Zero-draw fits return checks and inference without a conditioned model. repeat distinguishes timing attempts. GET polls call_id.

**Parameters**

- `workspace_id` (path, required)

**Input: [`FitInput<RevisionSelector>`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/inputs.py)**

Start or continue a fit; sample counts always mean additional retained draws.

| Field | Type | Presence / default | Description |
|---|---|---|---|
| `target` | [`FitTarget<RevisionSelector>`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/inputs.py) (`kind`: `continue`, `new`) | required | A model and saved history to fit, or a prior fit to continue. |
| `settings` | [`FitSettingsSpec`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/settings.py) | optional | Sampler overrides for this call. |
| `repeat` | `integer` (≥ 0) | optional; default `0` | Timing repeat identity; identical repeats reuse their saved result. |

**Success body: [`FitOutput`](../../packages/grassmoth/src/grassmoth/actions/contracts/bodies.py)**

A conditioned model, its completed checks, and self-contained inference evidence.

| Field | Type | Presence / default | Description |
|---|---|---|---|
| `dynamical_model_spec` | [`DynamicalModelSpec`](../../packages/grassmoth/src/grassmoth/artifacts/model/dynamical_model_spec.py) \| `null` | required | The combined conditioned model, or null when this fit and its parents have no retained draws. Zero-draw continuations reuse their parent's model without producing a revision. |
| `checks` | [`FitCheckReport`](../../packages/grassmoth/src/grassmoth/artifacts/reports/inference.py) | required | Completed compatibility and question findings for the selected history. |
| `inference` | [`InferenceReport`](../../packages/grassmoth/src/grassmoth/artifacts/reports/inference.py) | required | Run provenance, native sampler evidence, diagnostics and parameter summaries. |
| `profile` | [`FitProfileReport`](../../packages/grassmoth/src/grassmoth/artifacts/reports/profiling.py) \| `null` | required; default `null` | Timings for this execution only when settings.profile is true. |

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/fit" -H 'Accept: application/json' -o /tmp/action.json \
  -X POST \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "action": "fit",
  "input": {
    "target": {
      "kind": "new",
      "dynamical_model_spec_ref": "0000068000000000000000000000000000000000",
      "data_ref": {
        "revision": "0000069000000000000000000000000000000000",
        "replicate_index": 0
      }
    },
    "settings": {
      "stop_when_converged": null,
      "profile": false,
      "hardware": "local",
      "time_limit_seconds": null,
      "algorithm": null,
      "num_samples_per_chain": null,
      "num_warmup": null,
      "num_chains": null,
      "seed": null
    },
    "repeat": 0
  }
}
JSON
```

### POST `/api/studies/{workspace_id}/model_diff`

Compare input.before_ref and input.after_ref as saved DynamicalModelSpec documents. Return changes using the same partial DynamicalModelSpec contract as edit_model: omissions are unchanged and null map entries delete identities. Retain the patch in Git. GET polls call_id. Identical resolved calls reuse successes and failures; the comparison is the body.

**Parameters**

- `workspace_id` (path, required)

**Input: [`ModelDiffInput<RevisionSelector>`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/inputs.py)**

Base and target specs for a directional patch, selected by question or model revisions.

The selections need not be chronologically ordered or share an editing parent.
Before/after determine patch direction; ModelDiffOutput defines its contract.

| Field | Type | Presence / default | Description |
|---|---|---|---|
| `before_ref` | [`RevisionSelector`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/identity.py) | required | Question or model revision used as the base; a question denotes an empty spec. |
| `after_ref` | [`RevisionSelector`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/identity.py) | required | Question or model revision reconstructed by applying the patch to the base. |

**Success body: [`ModelDiffOutput`](../../packages/grassmoth/src/grassmoth/actions/contracts/bodies.py)**

Directional changes between saved specs, expressed in the model's editing language.

DynamicalModelSpec.changes_from owns the document comparison. Applying its patch with
merge_fields to the before spec reconstructs the after spec; equal specs yield
an empty document. Only saved specification fields participate, including their
exact law-buffer references. No execution findings or statistical comparisons
are computed here; those remain owned by their producing actions.

| Field | Type | Presence / default | Description |
|---|---|---|---|
| `changes` | [`DynamicalModelSpec`](../../packages/grassmoth/src/grassmoth/artifacts/model/dynamical_model_spec.py) | required | Merge this document into the before spec to obtain the after spec. Omitted fields are unchanged, supplied fields are added or updated, and null map entries delete their identities. A question revision denotes the empty spec. |

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/model_diff" -H 'Accept: application/json' -o /tmp/action.json \
  -X POST \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "action": "model_diff",
  "input": {
    "before_ref": "0000068000000000000000000000000000000000",
    "after_ref": "000006e000000000000000000000000000000000"
  }
}
JSON
```

### POST `/api/studies/{workspace_id}/prepare_data`

Prepare observations from uploaded tables using the selected model's definitions. input.source is a folder under data/{workspace_id}/, including subfolders. CSV and Parquet tables must supply a date or datetime timestamp column and are concatenated in captured file order. Source bytes and revision selectors are pinned before computing call identity. GET polls call_id.

**Parameters**

- `workspace_id` (path, required)

**Input: [`PrepareDataInput<RevisionSelector, SourceFolder>`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/inputs.py)**

Extract observations using the definitions of the selected model.

| Field | Type | Presence / default | Description |
|---|---|---|---|
| `dynamical_model_spec_ref` | [`RevisionSelector`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/identity.py) | required | Model revision owning the clock and observation definitions. |
| `source` | [`SourceFolder`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/preparation.py) | required | Folder under `data/{workspace_id}/` at the HTTP boundary; the captured file reference in the saved request. |
| `extraction` | map<[`IndicatorId`](../../packages/grassmoth/src/grassmoth/artifacts/identity.py), [`ExtractionSpec`](../../packages/grassmoth/src/grassmoth/artifacts/observations/preparation.py) (`kind`: `computed`, `recorded`, `semantic`)> (min entries 1) | required | Computed, recorded-reading or semantic extraction instructions keyed by the model's observation IDs. |
| `context` | `string` | optional; default `""` | Additional background for interpreting the uploaded source tables. |

**Success body: [`PrepareDataOutput`](../../packages/grassmoth/src/grassmoth/actions/contracts/bodies.py)**

Prepared observations, their resolved metadata, and data-quality findings.

| Field | Type | Presence / default | Description |
|---|---|---|---|
| `data` | [`ObservationData`](../../packages/grassmoth/src/grassmoth/artifacts/observations/history.py) | required | Full observation histories keyed by indicator ID, including measurement support intervals and missing values. |
| `metadata` | [`PreparedDataMetadata`](../../packages/grassmoth/src/grassmoth/artifacts/observations/preparation.py) | required | Preparation recipe, resolved observation schema, and the calendar origin used to interpret model time. |
| `profile` | [`DataProfileReport`](../../packages/grassmoth/src/grassmoth/artifacts/reports/data_quality.py) | required | Empirical statistics and data-quality findings for the prepared panel. |
| `extraction` | [`DataPreparationResult`](../../packages/grassmoth/src/grassmoth/artifacts/observations/preparation.py) | required | Retained extraction workers, failed chunks and reuse counts. |

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/prepare_data" -H 'Accept: application/json' -o /tmp/action.json \
  -X POST \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "action": "prepare_data",
  "input": {
    "dynamical_model_spec_ref": "0000068000000000000000000000000000000000",
    "source": "input",
    "extraction": {
      "indicator:00000000000000000001": {
        "kind": "semantic",
        "how_to_measure": "Synthetic example reading.",
        "source_columns": []
      },
      "indicator:00000000000000000002": {
        "kind": "semantic",
        "how_to_measure": "Synthetic example reading.",
        "source_columns": []
      }
    },
    "context": ""
  }
}
JSON
```

### POST `/api/studies/{workspace_id}/simulate`

Simulate input.dynamical_model_spec_ref using input.simulation. The model owns deterministic inputs and retained trajectory coordinates; authored initial states apply at simulation.start. body.data contains an array of observation histories of the same type returned by prepare_data. The successful body requires data and report; report owns single or paired numerical evidence, calculated summaries and causal findings; messages retain traces. GET polls call_id.

**Parameters**

- `workspace_id` (path, required)

**Input: [`SimulateInput<RevisionSelector>`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/inputs.py)**

Generate histories from a selected model and an explicit simulation design.

| Field | Type | Presence / default | Description |
|---|---|---|---|
| `dynamical_model_spec_ref` | [`RevisionSelector`](../../apps/grassmoth-server/src/grassmoth_server/actions/contracts/identity.py) | required | Revision supplying the authored or fitted generative law. |
| `simulation` | [`SimulationSpec`](../../packages/grassmoth/src/grassmoth/artifacts/simulation.py) | required | Requested start, horizon, and interventions. |
| `seed` | `integer` (≥ 0) | optional; default `0` | Generation seed, retained with the histories. |
| `num_replicates` | `integer` (≥ 1) | optional; default `100` | Number of independent histories. |

**Success body: [`SimulateOutput`](../../packages/grassmoth/src/grassmoth/actions/contracts/bodies.py)**

Generated histories and their complete scientific report, including owned numerical evidence.

| Field | Type | Presence / default | Description |
|---|---|---|---|
| `data` | array<[`ObservationData`](../../packages/grassmoth/src/grassmoth/artifacts/observations/history.py)> (min items 1) | required | One observation history per retained replicate, in replicate order. |
| `report` | [`SimulationReport`](../../packages/grassmoth/src/grassmoth/artifacts/simulation.py) | required | Generation evidence and evaluated findings. |

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/simulate" -H 'Accept: application/json' -o /tmp/action.json \
  -X POST \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "action": "simulate",
  "input": {
    "dynamical_model_spec_ref": "0000068000000000000000000000000000000000",
    "simulation": {
      "start": "2026-01-01",
      "horizon": "3d",
      "interventions": [
        {
          "target": "construct:00000000000000000002",
          "after": "1d",
          "value": 2.0
        }
      ]
    },
    "seed": 0,
    "num_replicates": 2
  }
}
JSON
```

### GET `/api/studies/{workspace_id}/timeline`

Slim call log: call IDs, retained arguments, outcome summaries and dependencies. The viewer reads complete results and messages with GET using each call_id.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/timeline"
```

### GET `/api/studies/{workspace_id}/{action}/{call_id}`

Read an existing call by ID. Never resolve inputs, start a workflow, execute, or retry. The envelope and accumulated messages match POST, including cached failures.

**Parameters**

- `workspace_id` (path, required)
- `action` (path, required)
- `call_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/ACTION/CALL_ID" -H 'Accept: application/json' -o /tmp/action.json
```

### POST `/api/upload`

Stage one named raw input file in the workspace's input folder. prepare_data reads that folder when input.source is "input" and captures each file's SHA-256.

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/upload" \
  -X POST \
  -F "file=@/path/to/file" \
  -F "workspaceId=WORKSPACEID"
```

### GET `/api/workspaces`

Available workspaces and their immutable study questions.

X-Actions-Enabled reports whether the facade accepts new calls.

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/workspaces"
```
