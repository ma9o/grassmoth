# API types architecture

- Python contracts and typed routes own the API. Follow the [code generation ownership](../../docs/guides/codegen.md#api-artifacts): change the owner or generator and regenerate exported schemas and TypeScript declarations; never hand-edit generated artifacts.

- [generate.ts](scripts/generate.ts) derives operations, named contracts and generic declarations from one OpenAPI AST. Preserve the Python owners' names and generic parameters rather than introducing a parallel TypeScript domain model.

- [client.ts](src/client.ts), [msgpack.ts](src/msgpack.ts) and [arrays.ts](src/arrays.ts) own transport decoding and numerical-buffer access. Preserve operation-owned response types and array dimensions through that boundary; keep scientific computation and UI presentation in their owning workspaces.
