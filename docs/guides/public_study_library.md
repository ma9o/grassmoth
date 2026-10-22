# Public study library

The service lives in `nof1_causal_service` beside the scientific library,
`nof1_causal_lab`, in the same distribution. The
[ARCH012 import contract](../../apps/data-pipeline/pyproject.toml)
forbids library imports of the service, including type-only imports.
Library actions receive resolved values and return complete scientific outputs.
The service owns reference resolution, extraction workers, runtime configuration,
compute dispatch and persistence. Fit receives its clocks, logger and compilation
functions from the service; the on-disk compilation cache stays there.

The service imports the library only through
[`nof1_causal_lab.public`](../../apps/data-pipeline/src/nof1_causal_lab/public.py).
This module re-exports only the seven actions and their input, output and failure
types. The
[ARCH013 checker](../../apps/data-pipeline/scripts/checks/check_public_imports.py)
enforces this import boundary, including type-only and dynamic imports.

Callers outside the framework use the
[HTTP client](../../apps/data-pipeline/src/nof1_causal_service/public/client.py),
`StudyClient`, from `nof1_causal_service.public.client`. It submits requests, polls
saved results, reads the timeline and stages files through the upload route.
Request, result and referenced value types, including `ActionResult` and
`StudyError`, come from
[contracts.py](../../apps/data-pipeline/src/nof1_causal_service/public/contracts.py).

Decode an `ActionResult` with its `decode()` method. Read saved observations and
posterior draws through the returned values' accessors. Diagnostics and convergence
are retained by fit, including each execution's segment. The
[action charts](../../README.md#documentation) own execution flow.

The maintained [evaluation runner](../../apps/data-pipeline/evaluation/README.md)
uses this boundary. Experiments needing a new capability patch its framework
owner in an isolated checkout, then exercise that patch through the same actions.
Keep the patch and experiment declaration in `scratchpad/`.

The [ARCH011 checker](../../apps/data-pipeline/scripts/checks/check_public_imports.py)
checks evaluation during normal lint. Select other callers explicitly, including
gitignored experiments and notebooks:

```bash
bun run --cwd apps/data-pipeline lint:public
bun run --cwd apps/data-pipeline lint:public ../../scratchpad/my_experiment
bun run --cwd apps/data-pipeline lint:public notebooks/my_notebook.py
```

It follows local helpers and package initializers, checks type-only and literal
dynamic imports, and rejects private imports and runtime import-path changes.
The framework package, including its API server, is implementation and is outside
this caller rule. Import checks constrain dependencies; reviewing an experiment
still establishes whether it delegates scientific work to the actions.
