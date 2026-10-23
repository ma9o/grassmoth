- DO NOT USE WORKTREES. DO YOUR WORK ON MASTER UNLESS SPECIFIED

- Add tests only when necessary to cover broad behavior; this repo's test suite is already costly.

- Before running tests, follow the [test selection guide](docs/guides/agentic_integration_testing.md#choosing-tests-for-a-change). Before integration testing, starting services, health checks, or manual pipeline runs, read and follow [docs/guides/agentic_integration_testing.md](docs/guides/agentic_integration_testing.md).

- TODO references mean the gitignored `scratchpad/TODO.md`.

- `cp` means "commit and push". Split commits atomically, but prioritize shared-workspace safety: never disrupt another agent's work with stashing or rebasing.

- Never add backwards compatibility code or defensive fallbacks.

- There is no baselining in this repo. Never record existing lint, type or test failures in a baseline, a bulk suppression or a "known failures" list. A new check ships with all its findings fixed, and a failing check gets fixed, not recorded. The only exception is an inline suppression of a genuine false positive, with its reason on the same line.

- The core follows functional programming. It is every module whose role is `domain`, `compiler`, `execution` or `projection` in `apps/grassmoth-server/scripts/checks/architecture_roles.py`, which owns module roles; only `edge` and `shell` modules do I/O or read clocks and configuration.
  - Values are immutable: frozen, with `tuple`, `frozenset` and `Mapping` collections. Build records whole instead of mutating them.
  - Composition over inheritance: a value type subclasses only `Value` and composes other values as fields. Alternatives are sum types, not a tag plus optional fields.
  - Parse, don't validate: each invariant has one owner, a smart constructor or an edge parser, and nothing downstream re-checks it. Pass the owner, not values derived from it. Projections are total; if one seems to need a check, fix the core type.
  - Return expected failures as typed outcomes instead of raising. Only bugs and infrastructure failures raise, and the shell handles them.

- Run lint for static feedback and before committing: `bun run --cwd <workspace> lint` when a change touches only one workspace, otherwise `bun run lint` at the repo root, which adds the cross-package checks (knip, API types, and docs drift).
  Fix findings at the owner named by the checker; rule details live in the checker docstrings.

- Keep dependencies explicit and preserve strong typing end to end. Follow the [type naming conventions](docs/guides/codegen.md#type-naming-conventions).

# Notebooks

- `apps/grassmoth-server/notebooks/` contains marimo Python notebooks.
- Name every top-level `@app.cell` function with a descriptive snake_case name. Keep inner helpers `_`-prefixed.
- From `apps/grassmoth-server`, validate edits with `uv run marimo check --strict notebooks/<name>.py`.
- Launch the editor from the repo root with `bun run --cwd apps/grassmoth-server notebooks`.
- For `marimo-pair`, open a specific notebook in the browser first; the file browser exposes no session. Attach with `execute-code.sh --url http://localhost:2718 --token <access_token>`, using the startup banner's token. Pass it via `--token` or `MARIMO_TOKEN`, never in the `--url` query string.

# Docs

- Place references beside the claims they support or hyperlink the relevant terms.

- Each fact has one maintained owner: the action charts in `docs/assets/action-flows` own control flow, code owns field meanings and the API, and `docs/assumptions.md` owns modeling commitments and limits. Link to the owner instead of restating it.

- A change to an action's behavior updates its chart in the same commit.

# Web app

- Keep domain logic and statistical computations in the backend, except for frontend visualization: evaluate or sample saved probability laws, evaluate saved model expressions, and transform aligned draws. Preserve joint dependence and declared parameter transforms. These display evaluations never feed inference, trajectory generation, diagnostics, predictive checks, or scientific decisions, which remain backend-owned.

- v2 (`/v2/{workspaceId}`) is the only interface. v1 lives at the annotated `v1-reference` tag as a reference for things v2 might surface; run it from a separate checkout of that tag with its own fixtures.

# Data Pipeline

- Experiments and comparisons use the server's public study actions over HTTP, following the [benchmark workflow](benchmarks/README.md#running) and [study API skill](.agents/skills/grassmoth-study-api/SKILL.md). Develop missing capabilities as the smallest coherent patch to their framework owners, applied in an isolated checkout with its own server and study; retain the patch and experiment declaration in `scratchpad/`. Exercise the patched framework through the same actions. Do not recreate model preparation, initialization, sampling orchestration, diagnostics or convergence gates in experiment harnesses.

- External callers use the [public study library](docs/guides/public_study_library.md). Check each scratchpad experiment or notebook with `bun run --cwd apps/grassmoth-server lint:public <caller-path>`; paths are relative to `apps/grassmoth-server`.

- These are agreed contracts:
  - the library surface: `grassmoth/public.py` and the signatures of the functions it lists;
  - HTTP request and response bodies;
  - the boundary rules ARCH011–ARCH013 and their configuration.

  Change a contract only when the user approves that specific change. If a task seems to need one, stop and ask. Never widen a surface or loosen a rule to make a check pass.

- Budget GPU benchmarks carefully: a B200 on Modal costs $6/hour.
- Run independent GPU benchmark scenarios concurrently on separate GPUs. Do not serialize them on one L40S; isolate each scenario's GPU so timing comparisons remain meaningful, and preserve shared initialization dependencies.

- Represent structural assumptions as DAGs with explicit latent confounders. ADMGs are only for internal projection into y0's identification algorithm, never user-facing.

- The latent SSM is continuous-time **nonlinear**. Linearization and Gaussian approximations are allowed **only for particle-sampler initialization**: parameter positions, proposal preconditioner, and cSMC reference trajectory.
- Production posteriors, all diagnostics, posterior-predictive checks, and counterfactual/predictive outputs must use the exact engines in [docs/assumptions.md](docs/assumptions.md#model-class).
- Before reintroducing linearization, run [test_linearization_init_only.py](packages/grassmoth/tests/models/ssm/test_linearization_init_only.py), which restricts Laplace imports to warmup/init.
