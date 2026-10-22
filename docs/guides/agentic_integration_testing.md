# Agentic Integration Testing

## Design Principles

| Concern | Mode | Why |
|---------|------|-----|
| File placement, scientific actions, run registration | **Programmatic** (`curl`) | Reliable, fast, no UI fragility |
| Runtime errors, route discovery, build state | **Next.js devtools MCP** | Reads the running app directly and catches server/runtime errors before UI debugging |
| UI rendering verification, visual regression | **browser automation** (`browser_eval` / Playwright) | Only way to see rendered output |

## Prerequisites

### Test concerns

Every Python test declares at least one [pytest concern marker](../../pyproject.toml),
including standalone notebook support tests. The top-level groups are `contract`,
`inference`, and `workflow`. Numerical tests declare a specific child of `inference`:

| Group | Child concern | Covers |
|-------|---------------|--------|
| `contract` | — | Isolated routing, validation, schema, identity, and projection without scientific numerical execution |
| `inference` | `sampling` | Parameterization, exact probability targets, particle sampling, and posterior diagnostics |
| `inference` | `warmup` | LAPS initialization, proposal preconditioning, and exact joint-density gradients |
| `inference` | `simulation` | State dynamics, deterministic/stochastic trajectories, steady states, and interventions |
| `inference` | `predictive` | Prior/posterior predictive generation, observation sampling, and predictive diagnostics |
| `inference` | `recovery` | Fitted accuracy and uncertainty checked against known generating parameters |
| `workflow` | — | Temporal workflows and worker orchestration |

Numerical tests use `pytest.mark.inference(concern="warmup")`, with the appropriate
child name. Pytest's [native keyword matching](https://docs.pytest.org/en/stable/example/markers.html#marking-test-functions-and-selecting-them-for-a-run)
makes `-m inference` select the whole numerical group and
`-m "inference(concern='warmup')"` select that child alone. No parent markers are
injected. A bare `inference` marker is invalid: every numerical test must name a
child concern. Count each test once when reporting totals for its parent group,
even when it has multiple child owners.

Apply a marker when a test executes that subsystem. Use `contract` for the isolated
checks in the default selection. Small inputs, fixed draws, a mocked optimizer, or
shape tracing do not make scientific execution a contract: exact density and
parameter-gradient checks belong to `sampling`, predictive draws and statistical
checks to `predictive`, and initialization solvers to `warmup`. A test that checks
both exact density and observation draws owns both `sampling` and `predictive`.
Keep routing tests in `contract` when the numerical execution is stubbed, and
validation tests when they reject inputs before execution. Attribute real Temporal
child workflows to `workflow`, even without a complete study journey.

Use function or class markers in mixed files;
a module's `pytestmark` applies only when every test shares the concern. A test can
have multiple owners: an initialization recovery test carries both
`pytest.mark.inference(concern="warmup")` and `pytest.mark.inference(concern="recovery")`.
The [collection check](../../conftest.py) rejects unowned cases
and missing or invalid inference concerns before `-m` or `-k` can deselect them.
Parameterization and timeout markers do not count as owners, and missing ownership
is never assigned automatically.

`bun run --cwd <python-workspace> test` and direct `uv run pytest` runs
use all available CPU workers (`-n auto`) and run the default contracts. Select specialized suites explicitly
when requested or directly affected by the change:

```bash
bun run --cwd apps/grassmoth-server test -m contract
bun run --cwd apps/grassmoth-server test -m inference
bun run --cwd apps/grassmoth-server test -m "inference(concern='sampling')"
bun run --cwd apps/grassmoth-server test -m "inference(concern='predictive')"
bun run --cwd apps/grassmoth-server test -m "inference(concern='warmup') and not inference(concern='recovery')"
bun run --cwd apps/grassmoth-server test -m "inference and not inference(concern='recovery')"
```

Add `--collect-only` to inspect a selection without executing it. `test:all`
includes every concern and requires an explicit request. Use `-n 0` for a serial
run or `-n N` to limit the worker count. Pytest reports durations separately and rejects
unregistered markers. Keep test files under their owning workspace’s `tests/` directory.
The default selection is `not (inference or workflow)`, which excludes every
numerical child, including tests that also carry `contract`. Web unit tests belong to the named `contract` Vitest project;
the separate `visual` project owns screenshot stories. The root
`test:fixture-promotion` command owns fixture-promotion checks.

The [shared pytest fixtures](../../conftest.py) force fitting onto
`local` and reject live Modal app runs. Hardware routing contracts exercise the
production selector with a mocked Modal boundary.

### Choosing tests for a change

Before testing, identify the behavior changed and its direct consumers, then choose
the smallest set of test files or node IDs that covers them. Select the relevant
contracts and numerical children from the concern table above. Test ownership
follows the behavior executed, so a source directory alone does not determine
which concerns need to run.

| Changed behavior | Relevant selection |
|------------------|--------------------|
| Schema, routing, validation, identity, or projection | Affected `contract` tests |
| Exact density, parameterization, particle sampling, or posterior diagnostics | `inference(concern='sampling')` and affected contracts |
| Initialization, proposal preconditioning, or joint-density gradients | `inference(concern='warmup')` and affected contracts |
| Dynamics, trajectories, steady states, or interventions | `inference(concern='simulation')` and affected contracts |
| Prior/posterior prediction, observation draws, or predictive checks | `inference(concern='predictive')` and affected contracts |
| Estimation accuracy or uncertainty against known generating parameters | `inference(concern='recovery')` plus the changed fitting or warmup concern |
| Temporal orchestration | Affected `workflow` tests and routing contracts |
| Test selection or ownership configuration | `tests/test_concerns.py`, then collection without scientific execution |
| Documentation only | `bun run docs:check` |

Include multiple children when the change affects their shared behavior. For
example, an emission change may affect both exact density and predictive draws;
an initialization-only change calls for warmup checks. Use the whole `inference`
group when shared numerical changes affect all its children, or when explicitly
requested. Run recovery tests when the change can affect fitted accuracy or
uncertainty, rather than merely because it touches a fitting-related file.

For a file-scoped Python run, invoke pytest directly from the repository root.
Library tests belong to `packages/grassmoth/tests`, service tests to
`apps/grassmoth-server/tests`, and shared test-selection checks to `tests/`:

```bash
uv run pytest packages/grassmoth/tests/models/ssm/inference/test_laps_initialization.py -m "contract or inference(concern='warmup')" -n 0
uv run pytest packages/grassmoth/tests/models/ssm/execution/test_emissions.py packages/grassmoth/tests/models/ssm/execution/test_observation_sampling.py -m "contract or inference(concern='sampling') or inference(concern='predictive')"
uv run pytest tests/test_concerns.py -m contract -n 0
```

Always supply the numerical or workflow selector when targeting those tests:
the default excludes them even when their file is named explicitly. The Bun
`test` wrapper includes `tests/` in its command, so appending a file does not
narrow it to that file. Reserve that wrapper for a whole concern suite. Use
`-n 0` for small selections when worker startup would dominate; the configured
default remains `-n auto`.

Use `--collect-only` if the selection is uncertain, and check that the intended
tests are selected. Once focused checks pass, broaden only for affected shared
behavior, failure diagnosis, or an explicit broader request. Record the command,
selected concerns, and outcome in the handoff. Follow the separate required
lint, type, documentation, and duplicate checks when they apply to the change.

### Local study history

New studies initialize their local bare repository on first use. On a fresh checkout, restore the tracked `HEALTHDEMO` history bundle before using its backend:

```bash
git clone --mirror data/HEALTHDEMO/study/history.bundle data/HEALTHDEMO/study/history.git
git --git-dir=data/HEALTHDEMO/study/history.git config grassmoth.format 28
```

#### Storage format

The [storage owner](../../apps/grassmoth-server/src/grassmoth_server/study/persistence/git_objects.py)
requires format 28. Start a new study or restore a format-28 bundle. The application
reads only this format and does not perform automatic migrations.

### Local stack

```bash
bun run integration:start
```

Starts the stack under [process-compose](https://github.com/F1bonacc1/process-compose)
supervision (`brew install f1bonacc1/tap/process-compose`; config:
[`process-compose.yaml`](../../process-compose.yaml)): the Temporal dev
server on port `7233` (ephemeral state, binary auto-downloaded on first
use), the study worker (task queue `grassmoth-studies`), the tool server
with the study facade on port `8100`, and the web app on port `3000`.
Startup order is health-gated (`depends_on` + readiness probes) and
crashed processes restart automatically. The script **stays in the
foreground** — wait until `curl -s http://localhost:8100/api/workspaces`
answers before proceeding (pass `-t=false` to disable the TUI when
redirecting output to a file).

To tear down the stack, kill the script (`Ctrl+C`, or `kill <pid>` if
backgrounded). All child processes are cleaned up automatically.

The process-compose API is pinned to port `8181` for targeted operations
against the running stack:

```bash
process-compose --port 8181 process list
process-compose --port 8181 process restart worker
```

The worker caches `apps/grassmoth-server/config.yaml` at first read
(`lru_cache`), so after editing pipeline config, restart the `worker`
process — no need to bounce the whole stack. If the supervisor was started with
`up --no-deps web`, `process-compose project update` can disable its dependency
processes; check that Temporal is still running afterward.

For local-study GPU fits, select `inference.sampler.hardware: L40S` (the default) in
[`config.yaml`](../../apps/grassmoth-server/config.yaml) and restart the worker.
Use `local` to fit on this machine; the other
[hardware enum values](../../packages/grassmoth/src/grassmoth/artifacts/sampling.py)
select Modal GPUs. Select `inference.sampler.algorithm: {method: nuts_da}` for
joint parameter/state NUTS. The particle samplers are `method: mpgibbs` and
`method: pgibbs`; the latter selects conditional MH or NUTS through
`parameter_kernel: {method: mh}` or `{method: nuts_da}`. Both use the shared
`dsmc` settings in the [sampler contract](../../packages/grassmoth/src/grassmoth/artifacts/sampling.py).
The [Modal compute adapter](../../apps/grassmoth-server/src/grassmoth_server/actions/fit/remote.py)
uses the installed Modal credentials, creates an ephemeral
`grassmoth-server` app with the current source, and reuses
`grassmoth-fit-cache:/jax`. The study remains local; the
[fit chart](../assets/action-flows/fit.svg) owns its failure and publication flow.

The Temporal dev server
persists its event history to `.local/agentic-integration-stack/temporal.db`
(the `--db-filename` on its command), so restarting the `temporal`
process — to serve the UI, pick up a change, or recover from a crash —
**resumes** in-flight study workflows exactly where they left off
rather than orphaning them. To start genuinely fresh, delete that
`temporal.db` before boot and wipe that workspace's `store/`, `study/`,
`scratch/`, and `cache/` directories. The Web UI is served at
`http://localhost:8233`.

## Workspace Layout

```text
data/
├── <WORKSPACE_ID>/        # User-facing workspace
│   ├── input/             # Ready-to-use CSV or Parquet tables for prepare_data
│   ├── store/             # Uploaded tables and transient execution buffers
│   ├── study/             # Complete results, logs and traces in each Git commit
│   ├── cache/             # Temporary compilation and file materialization
│   └── scratch/           # Live progress events and run-scoped execution state
└── HEALTHDEMO/                  # Tracked mock fixture workspace (evals + manual sampling)
```

Back up the whole workspace directory: Git retains complete action results and their numerical buffers; `store/` retains uploaded input tables. See the [storage owner](../../apps/grassmoth-server/src/grassmoth_server/study/persistence/store.py). Temporary cache entries are safe to delete at any time. `uv run grassmoth-sweep WORKSPACE_ID` expires telemetry and caches, and offline maintenance can add `--collect-runs` to remove finished run scratch.

### Promoting a workspace to the HEALTHDEMO fixture

Keep ordinary data workspaces and the committed fixture separate. Once a candidate
workspace has a complete, fresh artifact chain, promote it explicitly:

```bash
bun run fixture:promote --from <WORKSPACE_ID>
```

The command validates the selected Git snapshot, copies the durable workspace into
`data/HEALTHDEMO`, and rebuilds stable JSON and trace copies under `data/HEALTHDEMO/fixture/`
for Storybook. It replaces `data/HEALTHDEMO` as a unit rather than merging,
and excludes `cache/` and `scratch/`. It exports all Git refs and objects to
`study/history.bundle` so the tracked fixture retains attempt lineage and action
results while its local bare repository remains gitignored. The files in
`store/` retain uploaded input tables.

The tracked `data/HEALTHDEMO/study/history.bundle` and `data/HEALTHDEMO/store/` are the fixture's authoritative inputs. Files under `data/HEALTHDEMO/fixture/` are generated projections for Storybook. General behavior tests use small, test-owned fixtures independent of HEALTHDEMO. Regenerate Storybook fixtures with:

```bash
bun run fixture:build
```

The command restores the bundle into an isolated temporary repository and uses the production readers to project artifacts, logs, traces, historical snapshots and workbench comparisons in one pass. It does not read the local `history.git` or the generated projections as inputs. HEALTHDEMO has no numbered artifact directories or separate journal and trace directories.

The bundle preserves the authored and extracted facts through the last model edit. HEALTHDEMO retained no posterior samples or inference report, so it contains no successful fit. Reports and checks load the findings retained with [complete action results](../../packages/grassmoth/src/grassmoth/actions/contracts/bodies.py). Regeneration does not fit, simulate, or invent missing scientific artifacts. The [frontend renderer](../../apps/grassmoth-client/src/lib/model-asset/authored-prior-plot.ts) derives authored prior plots from the retained laws.

### Publishing a workspace

Use the current copier only for a stopped synthetic workspace and a destination with no previous copy. The hosted viewer serves the copied payloads:

```bash
uv run --project apps/grassmoth-server grassmoth-publish SYNTHETIC_WORKSPACE --exclude input
```

`--exclude input` withholds uploaded source files only. `--exclude raw_data` matches `store/raw_data/`; it does not remove current raw-table blobs under `store/blobs/` or their Git history. Existing remote files, including mutable Git refs, are skipped, so repeated uploads do not synchronize study history.

## Step-by-Step Flow

### 1. Create the workspace and start the study

```bash
WORKSPACE_ID="T3ST42"
QUESTION="How does screen time affect sleep?"

curl -s -X POST http://localhost:8100/api/upload \
  -F "workspaceId=$WORKSPACE_ID" \
  -F "file=@data/HEALTHDEMO/input/observations.csv"

curl -s -X POST http://localhost:8100/api/studies/$WORKSPACE_ID/edit_question \
  -H 'Content-Type: application/json' \
  -o /tmp/action.msgpack \
  -d "{\"action\":\"edit_question\",\"input\":{\"question\":{\"text\":\"$QUESTION\"}}}"
```

`GET /api/workspaces` includes the `X-Actions-Enabled` capability header. Poll the returned `call_id` using the [action API contract](../../.agents/skills/grassmoth-study-api/SKILL.md). A read-only facade serves saved calls, including failures and comparisons; unsaved POST calls return 403. The [action charts](../../README.md#documentation) show what each action does.

### 2. Observe the study

The study facade (tool server, port `8100`) is the source of truth:

```bash
# Slim journal: every recorded call's arguments, status, messages and dependencies
curl -s http://localhost:8100/api/studies/$WORKSPACE_ID/timeline \
  | jq '.attempts[] | {commit_id, seq: .record.seq, attempt: .record.attempt}'

# Read a running call without submitting work
curl -s http://localhost:8100/api/studies/$WORKSPACE_ID/timeline > /tmp/study-timeline.json
ACTION=$(jq -r '.running[0].action' /tmp/study-timeline.json)
CALL_ID=$(jq -r '.running[0].call_id' /tmp/study-timeline.json)
curl -s "http://localhost:8100/api/studies/$WORKSPACE_ID/$ACTION/$CALL_ID" \
  -o /tmp/action.msgpack
uv run --directory apps/grassmoth-server python - <<'PY'
from pathlib import Path
from pprint import pprint
from grassmoth_server.study.persistence.codec import unpack_result
result = unpack_result(Path('/tmp/action.msgpack').read_bytes())
pprint({key: value for key, value in result.items() if key != 'body'})
PY

# The viewer uses the same GET route for saved results and execution logs.
```

### 3. Verify via browser automation

- `http://localhost:3000/v2/{WORKSPACE_ID}` is the model workbench, the default destination from the workspace list.
- After each backend action, check that the workbench shows the question, graph, entity details, data, findings, history and action log. The workbench is read-only: the agent submits every change through the backend API.
- Storybook's **V2 / Model / Workbench / Complete** story covers the workbench with mocked responses. Extend it rather than adding separate stories. Its pinned responses are included in the [fixture build](#promoting-a-workspace-to-the-healthdemo-fixture).

If the UI behaves unexpectedly, check Next.js devtools MCP errors before debugging the browser script.

## Resuming after a failed action

A failed execution is a `raised` outcome in the journal. Scientific state is unchanged, and the outcome retains its error. Expected input rejections carry a `rejected` outcome:

```bash
curl -s http://localhost:8100/api/studies/$WORKSPACE_ID/timeline \
  | jq '.attempts[] | select(.record.attempt.outcome.status=="raised") | {seq: .record.seq, attempt: .record.attempt}'
```

Correct the scientific inputs and submit the changed call through its named action route. An identical resolved call returns its saved failure; changing only `reasoning` does not create another call. See the [action API contract](../../.agents/skills/grassmoth-study-api/SKILL.md).
