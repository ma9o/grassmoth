# API types architecture

- Python contracts and typed routes own the API. Change the owner or generator and regenerate exported schemas and TypeScript declarations; never hand-edit generated artifacts.

- [generate.ts](scripts/generate.ts) derives operations, named contracts and generic declarations from one OpenAPI AST. Preserve the Python owners' names and generic parameters rather than introducing a parallel TypeScript domain model. Generic operands come from the exported Python metadata; never recover them by parsing schema names.

- [client.ts](src/client.ts), [msgpack.ts](src/msgpack.ts) and [arrays.ts](src/arrays.ts) own transport decoding and numerical-buffer access. Preserve operation-owned response types and array dimensions through that boundary; keep scientific computation and UI presentation in their owning workspaces.

## Code generation

- After changing an artifact, read model, study record, endpoint contract or facade response, run `bun run codegen` from the repository root and commit the owner and generated changes together. `bun run codegen:check` verifies API drift.
- The [Python exporter](../../apps/grassmoth-server/AGENTS.md#contract-exports) owns `schemas/openapi.json`; [generate.ts](scripts/generate.ts) owns `src/generated/models.ts` and `src/generated/model-api.ts`. [src/index.ts](src/index.ts) owns hand-written re-exports. Register new artifact contracts with their Python owner and expose needed named exports here.
- Keep generated output properties, arrays and tuples readonly. Preserve named map keys as `Readonly<Partial<Record<Id, T>>>` and fixed tuple dimensions. For presence mismatches, inspect validation and serialization schemas separately; [Value](../grassmoth/src/grassmoth/artifacts/base.py) owns whether fields are emitted, nullable, defaulted or excluded.
- Upload inputs use native `Blob` values, serialized as `FormData` by the caller. Decode responses through the shared client before reading scientific fields.
- Keep `openapi-fetch` at 0.16.0 while [0.17.0's response mapping](https://github.com/openapi-ts/openapi-typescript/blob/openapi-fetch@0.17.0/packages/openapi-typescript-helpers/src/index.ts) widens fixed tuples into arrays. The [client type test](src/model-snapshot.type-test.ts) checks response dimensions and parity between generated operations and the exported schema.
