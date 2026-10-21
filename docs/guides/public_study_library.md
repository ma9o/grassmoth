# Public study library

Callers outside the framework import `nof1_causal_lab.public`. The
[study operations](../../apps/data-pipeline/src/nof1_causal_lab/public/studies.py)
share the production action lifecycle. The
[HTTP client](../../apps/data-pipeline/src/nof1_causal_lab/public/client.py)
submits requests, polls saved results and reads the timeline; it also stages files
through the existing upload route. Request, result and referenced value types
come from [contracts.py](../../apps/data-pipeline/src/nof1_causal_lab/public/contracts.py).

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
