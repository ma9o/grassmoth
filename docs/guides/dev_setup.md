# Local Development Setup

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Node.js | 23 (`.node-version`) | `node --version` |
| Bun | 1.3.14 (`package.json` `packageManager`) | `bun --version` |
| Python | 3.12+ (`apps/data-pipeline/.python-version`) | `python3 --version` |
| uv | Latest | `uv --version` |

## Setup

```bash
# 1. JS/TS deps (also sets up git hooks)
bun install --frozen-lockfile

# 2. Python deps
cd apps/data-pipeline
uv sync --frozen --group dev
# Optional: --group cloud (Modal, R2/S3)

# 3. Environment
cd ../..
cp .env.example.dev .env
# Fill in OPENROUTER_API_KEY (required)
# Optional: EXA_API_KEY, TOOL_SERVER_URL

# 4. Code generation
bun run codegen
bun run docs:codegen
```

Edit `.env` and fill in at minimum:

- `OPENROUTER_API_KEY` — the ambient credential for semantic measurement extraction (the only key mechanism; there is no per-user handoff)

Optional keys:

- `EXA_API_KEY` — literature search
- `TOOL_SERVER_URL` — override the study facade / tool server URL (default `http://localhost:8100`)
- `TEMPORAL_ADDRESS` — override the Temporal dev server address (default `localhost:7233`)
- `READ_ONLY_FACADE=1` — serve reads only (what the hosted viewer's facade sets)

### 4. Generate API Artifacts and Docs

Generates committed API artifacts and documentation from Python sources. See the [code generation guide](codegen.md) for details.

### 5. Start development servers

```bash
# From repo root — starts all apps via Turbo:
bun run dev
```

Or individually:

| App | Command | Port |
|-----|---------|------|
| Web viewer | `cd apps/web && bun run dev` | 3000 |
| Temporal dev server | `cd apps/data-pipeline && uv run python scripts/dev/temporal_dev_server.py` | 7233 |
| Study worker | `cd apps/data-pipeline && PYTHONHASHSEED=0 uv run python -m nof1_causal_lab.actions.temporal.worker` | — |
| Tool server / study facade | `cd apps/data-pipeline && bun run dev` | 8100 |

The web viewer works standalone with mock data. Live studies also need the Temporal dev server, the study worker, and the tool server — `bun run integration:start` brings up the whole stack (see the [integration testing guide](agentic_integration_testing.md)).

## Common Commands

Run these commands from the repo root. Shared development and quality tasks use
Turbo across the workspaces; generators and maintenance commands run in their
owning package.

```bash
bun run lint          # Every static check, including types and generated-artifact drift
bun run lint:fix      # Apply lint and formatting fixes
bun run test          # Lightweight backend tests (one worker) and Vitest
bun run codegen:check # Generated API artifact drift
bun run docs:check    # Generated documentation drift and markdown
```

Python source docstrings use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
The standard Python lint command checks public docstring presence
and formatting with Ruff, then checks structured docstrings against signatures,
returns, yields, exceptions, and attributes with `pydoclint`. The
[configuration](../../apps/data-pipeline/pyproject.toml) keeps types in annotations
and property descriptions beside their implementations. Summary-only docstrings
are allowed for simple operations. Describe behavior, units, shapes, ownership,
and meaningful absence or failure conditions where relevant; repeating a name
or type adds no documentation.

Each field has one documentation owner. Prefer Pydantic
[`Field(description=...)`](https://docs.pydantic.dev/latest/concepts/fields/#customizing-json-schema)
for model field meanings, so the same prose reaches JSON Schema, OpenAPI, and
generated clients. The
[pydoclint adapter](../../apps/data-pipeline/scripts/checks/run_pydoclint.py)
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

| Workflow | Root commands |
|----------|---------------|
| Development | `dev`, `build`, `storybook`, `integration:start` |
| Quality | `lint`, `lint:fix`, `test`, `test:all`, `test:fixture-promotion`, `knip`, `complexity` |
| API generation | `codegen`, `codegen:check` |
| Documentation | `docs:codegen`, `docs:check`; individual tasks under `docs:distribution`, `docs:latex`, `docs:markdown:check`, `docs:spell:check` |
| Fixtures | `fixture:promote`, `fixture:build` |

Use `bun run` to list root scripts, or `bun run --cwd <workspace>` to list one
package's scripts. Run focused tasks in their workspace, for example:

```bash
bun run --cwd apps/web storybook:build
bun run --cwd apps/data-pipeline lint:type-boundaries:tests
bun run --cwd packages/api-types codegen:schemas:check
```

Check and update variants reuse their base command with the relevant flag. Keep
the underlying command in one place when adding a new variant. `test:all` includes
every test concern; see [test selection by concern](agentic_integration_testing.md#test-concerns)
before using it.
