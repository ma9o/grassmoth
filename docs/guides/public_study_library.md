# Public study library

The service lives in `grassmoth_server` beside the scientific library,
`grassmoth`, in the same distribution. The
[ARCH012 import contract](../../pyproject.toml)
forbids library imports of the service, including type-only imports.
Library actions receive resolved values and return complete scientific outputs.
The service owns reference resolution, extraction workers, runtime configuration,
compute dispatch and persistence. Fit receives its clocks, logger and compilation
functions from the service; the on-disk compilation cache stays there.

Callers import the library only through
[`grassmoth.public`](../../packages/grassmoth/src/grassmoth/public.py).
This module re-exports only the seven actions and their input, output and failure
types. The
[public import checker](../../apps/grassmoth-server/scripts/checks/check_public_imports.py)
enforces this boundary with ARCH011 for external callers and ARCH013 for the
service, including its maintenance tools. Both rules check type-only and dynamic
imports. Tests may import internals.

Benchmarks and experiments use the server's public study actions over HTTP,
following the [benchmark workflow](../../benchmarks/README.md#running).
Direct library measurements are reserved for quantities no action exposes, such
as kernel timing or initializer scoring. The service contract is its request and
response bodies, documented by the
[OpenAPI schema](../../packages/api-types/schemas/openapi.json) and the generated
[study API reference](../reference/study-api.md). The
[action charts](../../README.md#documentation) own execution flow.

Experiments needing a new capability patch its framework owner in an isolated
checkout with its own server and study, then exercise that patch through the
same actions.
Keep the patch and experiment declaration in `scratchpad/`.

The [ARCH011 checker](../../apps/grassmoth-server/scripts/checks/check_public_imports.py)
checks all tracked and new non-ignored Python callers during normal lint, across
every workspace, including scripts and notebooks. Explicit paths narrow the
selection and can include gitignored experiments:

```bash
bun run --cwd apps/grassmoth-server lint:public
bun run --cwd apps/grassmoth-server lint:public ../../scratchpad/my_experiment
bun run --cwd apps/grassmoth-server lint:public notebooks/my_notebook.py
```

It follows local helpers and package initializers, checks type-only and literal
dynamic imports, and rejects private imports and runtime import-path changes.
The library implementation owns its internal imports; the service implementation
owns service-internal imports but must use the library's public surface. Test
entry points may use internals, but helpers imported by non-test callers are
checked even when they live in tests. Import checks constrain dependencies;
reviewing an experiment still establishes whether it delegates scientific work
to the actions.

## Analyze cached fit results

The [CPU analysis helper](../../apps/grassmoth-server/src/grassmoth_server/compute/analysis.py)
runs a script beside a fit result already cached on Modal:

```bash
uv run grassmoth-server analyze --workspace STUDY --call call:HASH -- analysis.py
```

The script receives the verified MessagePack body path as its first argument and
in `GRASSMOTH_RESULT_PATH`; subsequent command arguments follow it. Its output and
exit status return to the caller. Only the script and call metadata are sent from
the local machine. Results belong on the Volume, never in an image.

The [cache owner](../../apps/grassmoth-server/src/grassmoth_server/compute/analysis_cache.py)
keeps the complete remote fit output by content hash, including earlier phase
traces on continuations. The study journal owns publication through a
[compact receipt](../../apps/grassmoth-server/src/grassmoth_server/compute/remote_receipts.py).
Explicit result reads resolve and verify its complete body. Missing or corrupt
bodies fail explicitly, without repeating computation.

Deploy the [daily retention job](../../apps/grassmoth-server/src/grassmoth_server/compute/analysis_deployment.py):

```bash
uv run modal deploy apps/grassmoth-server/src/grassmoth_server/compute/analysis_deployment.py
```

The cache owner defines temporary-data expiry and the retained namespace for receipt-backed results on the shared Volume.
