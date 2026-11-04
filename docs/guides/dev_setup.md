# Local Development Setup

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Node.js | 23 (`.node-version`) | `node --version` |
| Bun | 1.3.14 (`package.json` `packageManager`) | `bun --version` |
| Python | 3.12+ (`.python-version`) | `python3 --version` |
| uv | Latest | `uv --version` |

## Setup

```bash
# 1. JS/TS deps (also sets up git hooks)
bun install --frozen-lockfile

# 2. Python workspace deps (one root lockfile and environment)
uv sync --frozen --all-packages --group dev
# Optional: --group gpu on Linux

# 3. Environment
cp .env.example.dev .env
# Fill in OPENROUTER_API_KEY (required)
# Optional: EXA_API_KEY, TOOL_SERVER_URL

# 4. Code generation
bun run codegen
```

Edit `.env` and fill in at minimum:

- `OPENROUTER_API_KEY` — the ambient credential for semantic measurement extraction (the only key mechanism; there is no per-user handoff)

Optional keys:

- `EXA_API_KEY` — literature search
- `TOOL_SERVER_URL` — override the study facade / tool server URL (default `http://localhost:8100`)
- `TEMPORAL_ADDRESS` — override the Temporal dev server address (default `localhost:7233`)
- `READ_ONLY_FACADE=1` — serve reads only (what the hosted viewer's facade sets)

### 4. Generate API Artifacts and Docs

Generate committed API artifacts and documentation from Python sources following the [API generation workflow](../../packages/api-types/AGENTS.md#code-generation) and [documentation generation guidance](../AGENTS.md#generated-documentation).

### 5. Start development servers

```bash
# From repo root — starts all apps via Turbo:
bun run dev
```

Or individually:

| App | Command | Port |
|-----|---------|------|
| Web viewer | `cd apps/grassmoth-client && bun run dev` | 3000 |
| Temporal dev server | `cd apps/grassmoth-server && uv run python scripts/dev/temporal_dev_server.py` | 7233 |
| Study worker | `cd apps/grassmoth-server && uv run python -m grassmoth_server.actions.temporal.worker` | — |
| Tool server / study facade | `cd apps/grassmoth-server && bun run dev` | 8100 |

The web viewer works standalone with mock data. Live studies also need the Temporal dev server, the study worker, and the tool server — `bun run integration:start` brings up the whole stack (see the [integration testing guide](agentic_integration_testing.md)).

## Workspace layout

The Python library lives in `packages/grassmoth` (`grassmoth`), the service in
`apps/grassmoth-server` (`grassmoth_server`), the repository checkers in
`packages/checks` (`grassmoth_checks`), and the viewer in
`apps/grassmoth-client`. The service declares its library dependency explicitly.
The `docs` workspace owns the documentation, its generators and the README; the
root `README.md` is a symlink to `docs/README.md`.
The [root Python configuration](../../pyproject.toml) owns the uv workspace and
shared Ruff and `basedpyright` defaults; `uv.lock` and `.venv/` also live at the root.
The [library](../../packages/grassmoth/pyproject.toml),
[service](../../apps/grassmoth-server/pyproject.toml) and
[checks](../../packages/checks/pyproject.toml) packages own their tool scopes and
pytest configuration; the library and service also own dependency checks, module
roles and public exports.

Library tests live in `packages/grassmoth/tests`, service tests in
`apps/grassmoth-server/tests`, checker tests in `packages/checks/tests`, and client
contracts in `apps/grassmoth-client/tests`.
The library and service suites each own their fixtures and a distinct
test-support package inside their `tests/` directory. Server tests may import `grassmoth_test_support` from the
library suite; library tests never depend on server test support or configuration.
Package commands run tools in their own directory. The custom checkers live in
the [checks package](../../packages/checks/src/grassmoth_checks); packages run them
as `python -m grassmoth_checks.<checker>`, which reads the invoking package's
configuration.
Each Vulture source pass scans one package. Declared public exports and their
signature types count as external uses; another package's references cannot
keep internal code alive. Tests, scripts and notebooks have separate passes.

## Common Commands

Run these commands from the repo root. Shared development and quality tasks use
Turbo across the workspaces; generators and maintenance commands run in their
owning package.

```bash
bun run lint          # Every static check, including types, docs and generated-artifact drift
bun run lint:fix      # Apply lint and formatting fixes
bun run test          # Python contract suites, checker tests and client contracts
bun run codegen:check # Generated API and documentation drift
```

Python source docstrings use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
The standard Python lint command checks public docstring presence
and formatting with Ruff, then checks structured docstrings against signatures,
returns, yields, exceptions, and attributes with `pydoclint`. The
[package configuration](../../packages/grassmoth/pyproject.toml) keeps types in annotations
and property descriptions beside their implementations. Summary-only docstrings
are allowed for simple operations. Describe behavior, units, shapes, ownership,
and meaningful absence or failure conditions where relevant; repeating a name
or type adds no documentation.

Each field has one documentation owner. Prefer Pydantic
[`Field(description=...)`](https://docs.pydantic.dev/latest/concepts/fields/#customizing-json-schema)
for model field meanings, so the same prose reaches JSON Schema, OpenAPI, and
generated clients. The
[pydoclint adapter](../../packages/checks/src/grassmoth_checks/run_pydoclint.py)
recognizes these descriptions, including `Annotated` metadata, without duplicate
`Attributes:` entries. Use `Attributes:` for ordinary classes and dataclasses.
The class docstring explains the record's purpose and relationships. Add a
description when it explains semantics, units, alignment, or absence; an empty
`Field()` or a repetition of the field name adds no documentation.

### Script organization

Each `package.json` groups scripts by workflow: development, quality checks,
generation, maintenance, and install hooks. Aggregate commands come before their
subcommands. Related commands share a colon namespace; artifact checks end in
`:check`, fixes in `:fix`, and updates in `:update`.

Each workspace owns its commands. Use the same names for the same jobs:
`lint`, `lint:fix`, `format`, `format:check`, `test`, `codegen`, `codegen:check`,
`dev` and `build`, where applicable. Each command has one owner; do not keep
aliases or forwarding scripts at another level of the tree.

Every workspace's `lint` invokes `turbo run lint:workspace --continue=always`.
Its `lint:workspace` holds the local checks, while
[`turbo.json`](../../turbo.json) declares the prerequisite checks that a change
in that workspace can affect. Client lint includes Knip; API-types lint includes
API drift; docs lint includes generated-math drift. Both Python workspace lints
run their local checks and the API drift check. Turbo runs each prerequisite once
per invocation.

The root fans out workspace commands and owns only the checks that need the whole
tree, under `lint:root`: exported-field readers (`lint:fields`, FIELD003), callers
outside the Python packages (`lint:public`, ARCH011), and the duplicate audit
(`lint:duplicates`). `similarity` is the matching on-demand report.
Package-local FIELD001/002 check stored-field redundancy. The service checks its
own callers and configuration; the library owns import contracts and shape-check
wiring. `lint:fix` updates each owner's files, so the library's shape list is
synchronized once.

The library and service suites load the
[pytest policy](../../packages/grassmoth/tests/grassmoth_test_support/pytest_policy.py)
from the library's test support for concern validation, compilation caching and
the library's runtime shape instrumentation; the checks suite holds only
contracts and runs without it. CI runs each Python package's `test` as its own
step, so each suite uses its own configuration and two `-n auto` pools never
overlap.

| Workflow | Root commands |
|----------|---------------|
| Development | `dev`, `build`, `integration:start` |
| Quality | `lint`, `lint:fix`, `test`, `test:all`, `similarity`; whole-tree checks under `lint:root` |
| Generation | `codegen`, `codegen:check` |

Use `bun run` to list root scripts, or `bun run --cwd <workspace>` to list one
package's scripts. Run focused tasks in their workspace, for example:

```bash
bun run --cwd apps/grassmoth-client lint:knip
bun run --cwd apps/grassmoth-server lint:ruff
bun run --cwd packages/grassmoth lint:type-boundaries:tests
bun run --cwd packages/grassmoth complexity
bun run --cwd docs lint
bun run lint:public scratchpad/my_experiment
bun run --cwd packages/api-types codegen:schemas:check
```

Check and update variants reuse their base command with the relevant flag. Keep
the underlying command in one place when adding a new variant. `test:all` includes
every test concern; see [test selection by concern](agentic_integration_testing.md#test-concerns)
before using it.
