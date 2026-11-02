# Repository guidance

This file owns shared working rules and cross-workspace constraints. Read the scoped guidance for each area your task concerns, and for the owners of behavior your change consumes:

- [Web client](apps/grassmoth-client/AGENTS.md): the workbench and frontend visualization.
- [Study service](apps/grassmoth-server/AGENTS.md): HTTP, orchestration, persistence and compute dispatch.
- [Scientific library](packages/grassmoth/AGENTS.md): scientific actions and exact model computation.
- [API types](packages/api-types/AGENTS.md): generated TypeScript contracts and transport codecs.
- [TypeScript configuration](packages/typescript-config/AGENTS.md): shared compiler configuration.
- [Experiments and benchmarks](benchmarks/AGENTS.md): scientific experiments, including those in `scratchpad/`.
- [Documentation](docs/AGENTS.md): documentation and action charts, including chart updates when action behavior changes.

Keep scope-specific guidance in its directory's `AGENTS.md` and link it from the nearest ancestor `AGENTS.md`. App and package files own their architecture and link to more specific scopes. Keep shared rules here and link to existing owners instead of duplicating their guidance.

## Working rules

- Add tests only when necessary to cover broad behavior; this repo's test suite is already costly.

- Before running tests, follow the [test selection guide](docs/guides/agentic_integration_testing.md#choosing-tests-for-a-change). Before integration testing, starting services, health checks, or manual pipeline runs, read and follow [docs/guides/agentic_integration_testing.md](docs/guides/agentic_integration_testing.md).

- TODO references mean the gitignored `scratchpad/TODO.md`.

- `cp` means "commit and push". Split commits atomically, but prioritize shared-workspace safety: never disrupt another agent's work with stashing or rebasing.

- Work in the shared checkout. Don't create a git worktree or a separate clone unless a task strictly requires an isolated checkout.

- Never add backwards compatibility code or defensive fallbacks.

- There is no baselining in this repo. Never record existing lint, type or test failures in a baseline, a bulk suppression or a "known failures" list. A new check ships with all its findings fixed, and a failing check gets fixed, not recorded. The only exception is an inline suppression of a genuine false positive, with its reason on the same line.

- Run lint for static feedback and before committing: `bun run --cwd <workspace> lint` when a change touches only one workspace, otherwise `bun run lint` at the repo root, which adds the cross-package checks (knip, API types, and docs drift).
  Fix findings at the owner named by the checker; rule details live in the checker docstrings.

## Shared architecture

- The core follows functional programming. It is every module whose role is `domain`, `compiler`, `execution` or `projection` in [architecture_roles.py](apps/grassmoth-server/scripts/checks/architecture_roles.py), which owns module roles; only `edge` and `shell` modules do I/O or read clocks and configuration.
  - Values are immutable: frozen, with `tuple`, `frozenset` and `Mapping` collections. Build records whole instead of mutating them.
  - Composition over inheritance: a value type subclasses only `Value` and composes other values as fields. Alternatives are sum types, not a tag plus optional fields.
  - Parse, don't validate: each invariant has one owner, a smart constructor or an edge parser, and nothing downstream re-checks it. Pass the owner, not values derived from it. Projections are total; if one seems to need a check, fix the core type.
  - Return expected failures as typed outcomes instead of raising. Only bugs and infrastructure failures raise, and the shell handles them.

- Keep dependencies explicit and preserve strong typing end to end. Follow the [type naming conventions](packages/grassmoth/AGENTS.md#type-naming-conventions). Web and API TypeScript consumers enable `noUncheckedIndexedAccess` and `exactOptionalPropertyTypes`: sparse lookups retain absence, and omitted request options stay absent.

- These are agreed contracts:
  - the library surface: [grassmoth/public.py](packages/grassmoth/src/grassmoth/public.py) and the signatures of the functions it lists;
  - HTTP request and response bodies;
  - the boundary rules ARCH011–ARCH013 and their configuration.

  Change a contract only when the user approves that specific change. If a task seems to need one, stop and ask. Never widen a surface or loosen a rule to make a check pass.
