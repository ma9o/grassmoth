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
| `inference` | `warmup` | Initialization, proposal preconditioning, and exact joint-density gradients |
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
The [collection check](../../packages/grassmoth/tests/grassmoth_test_support/pytest_policy.py) rejects unowned cases
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
Compilation dominates numerical tests, so the [shared pytest policy](../../packages/grassmoth/tests/grassmoth_test_support/pytest_policy.py) keeps
JAX's persistent cache in `~/.cache/grassmoth/jax-tests`; delete it to time a cold run.
The default selection is `not (inference or workflow)`, which excludes every
numerical child, including tests that also carry `contract`. Web unit tests belong to the named `contract` Vitest project.

The [server pytest fixtures](../../apps/grassmoth-server/tests/conftest.py) force fitting onto
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
| Test selection or ownership configuration | Collection of the affected workspace without scientific execution |
| Documentation only | `bun run --cwd docs lint` |

Include multiple children when the change affects their shared behavior. For
example, an emission change may affect both exact density and predictive draws;
an initialization-only change calls for warmup checks. Use the whole `inference`
group when shared numerical changes affect all its children, or when explicitly
requested. Run recovery tests when the change can affect fitted accuracy or
uncertainty, rather than merely because it touches a fitting-related file.

For a file-scoped Python run, invoke pytest directly from the repository root
with the owning package's configuration. Run the two suites separately; each
owns its test selection and import paths.
Library tests belong to `packages/grassmoth/tests`, service tests to
`apps/grassmoth-server/tests`. Each suite owns its fixtures and test support:

```bash
uv run pytest -c packages/grassmoth/pyproject.toml packages/grassmoth/tests/models/ssm/inference/test_particle_initialization.py -m "contract or inference(concern='warmup')" -n 0
uv run pytest -c packages/grassmoth/pyproject.toml packages/grassmoth/tests/models/ssm/execution/test_emissions.py packages/grassmoth/tests/models/ssm/execution/test_observation_sampling.py -m "contract or inference(concern='sampling') or inference(concern='predictive')"
bun run --cwd packages/grassmoth test --collect-only -m contract -n 0
bun run --cwd apps/grassmoth-server test --collect-only -m contract -n 0
```

Always supply the numerical or workflow selector when targeting those tests:
the default excludes them even when their file is named explicitly. The Bun
`test` command runs pytest from its package, using that package's `testpaths`
when no file is supplied. File arguments are relative to that package. Use
`-n 0` for small selections when worker startup would dominate; the configured
default remains `-n auto`.

Use `--collect-only` if the selection is uncertain, and check that the intended
tests are selected. Once focused checks pass, broaden only for affected shared
behavior, failure diagnosis, or an explicit broader request. Record the command,
selected concerns, and outcome in the handoff. Follow the separate required
lint, type, documentation, and duplicate checks when they apply to the change.

### Local study history

New studies initialize `history.git/` at the workspace root on first use. The
[storage owner](../../apps/grassmoth-server/src/grassmoth_server/study/persistence/git_objects.py)
requires format 32. Start a new study for older formats; there is no automatic
migration or bundle restore step.

A tracked study includes its packed `history.git/`. With its workers stopped and
no calls pending, prepare the repository before committing it:

```bash
STUDY_HISTORY="data/WORKSPACE_ID/history.git"
git --git-dir="$STUDY_HISTORY" repack -ad
git --git-dir="$STUDY_HISTORY" pack-refs --all
rm -rf "$STUDY_HISTORY/hooks" "$STUDY_HISTORY/info" "$STUDY_HISTORY/description"
```

Never prune Git objects while a call is pending: unpublished result objects are
not yet reachable from a saved call ref. Copying the packed repository, including
its format configuration, is sufficient to read every saved result and timeline.

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
select Modal GPUs. `inference.sampler.algorithm` selects MPGibbs (the default) or NUTS-DA.
MPGibbs owns `algorithm.path_update`: Particle-mGRAD, twisted Particle-aGRAD or
auxiliary Kalman in the [sampler contract](../../packages/grassmoth/src/grassmoth/artifacts/sampling.py).
NUTS owns `algorithm.smoother`: `joint` moves parameters and paths together;
selecting an existing path smoother alternates parameter NUTS with that update.
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
`temporal.db` before boot and remove that test workspace's `history.git/` and
`tmp/` directories while the stack is stopped. The Web UI is served at
`http://localhost:8233`.

## Workspace Layout

```text
data/
└── <WORKSPACE_ID>/
    ├── history.git/       # Saved calls, exact served results, records, logs and sources
    └── tmp/               # Ignored by Git
        ├── input/         # Uploaded CSV or Parquet tables awaiting capture
        └── <call id>/     # Captured sources, parsed tables, extraction, events and profiles
```

Saved calls own everything needed for reads inside `history.git/`; Modal fits
also retain an immutable full-result pointer there for continuation and analysis.
The [fit chart](../assets/action-flows/fit.svg) owns those branches.

`uv run grassmoth-sweep WORKSPACE_ID` removes temporary directories only for
saved calls, including failures, by checking call refs without parsing saved calls.
The same cleanup runs after publication in that workspace. Startup opens no studies;
crash leftovers wait for the next publication or an explicit sweep.
If publication fails, the temporary work remains. With the workspace stopped,
`uv run grassmoth-sweep WORKSPACE_ID --offline` clears all of `tmp/`, including
uploads and captures abandoned before acceptance. Nothing is removed by age.

A saved `prepare_data` GET needs no uploads. To repeat its POST, first put its
source files back in the named folder: POST resolves source hashes before reuse.
Hand-authored relative source folders remain valid alongside `tmp/input/`.

### Publishing a workspace

Hosted R2 serving is not implemented for format 32.

## Step-by-Step Flow

### 1. Create the workspace and start the study

```bash
WORKSPACE_ID="T3ST42"
QUESTION="How does screen time affect sleep?"

cat > /tmp/grassmoth-observations.csv <<'CSV'
timestamp,screen_hours,sleep_hours
2026-01-01,3,7
2026-01-02,2,8
CSV

curl -s -X POST http://localhost:8100/api/upload \
  -F "workspaceId=$WORKSPACE_ID" \
  -F "file=@/tmp/grassmoth-observations.csv"

curl -s -X POST http://localhost:8100/api/studies/$WORKSPACE_ID/edit_question \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -d "{\"action\":\"edit_question\",\"input\":{\"question\":{\"text\":\"$QUESTION\",\"outcome\":\"construct:sleep\"}}}"
```

`GET /api/workspaces` includes the `X-Actions-Enabled` capability header. Poll the returned `call_id` using the [study API reference](../reference/study-api.md). A read-only facade serves saved calls, including failures and comparisons; unsaved POST calls return 403. The [action charts](../README.md#documentation) show what each action does.

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
  -H 'Accept: application/json' | jq 'del(.body)'

# The viewer uses the same GET route for saved results and execution logs.
```

### 3. Verify via browser automation

- `http://localhost:3000/workspaces/{WORKSPACE_ID}` is the model workbench, the default destination from the workspace list.
- After each backend action, check that the workbench shows the question, graph, entity details, data, findings, history and action log. The workbench is read-only: the agent submits every change through the backend API.

If the UI behaves unexpectedly, check Next.js devtools MCP errors before debugging the browser script.

## Resuming after a failed action

A failed execution is a `raised` outcome in the journal. Scientific state is unchanged, and the outcome retains its error. Expected input rejections carry a `rejected` outcome:

```bash
curl -s http://localhost:8100/api/studies/$WORKSPACE_ID/timeline \
  | jq '.attempts[] | select(.record.attempt.outcome.status=="raised") | {seq: .record.seq, attempt: .record.attempt}'
```

Correct the scientific inputs and submit the changed call through its named action route. An identical resolved call returns its saved failure; changing only `reasoning` does not create another call. See the [study API reference](../reference/study-api.md).
