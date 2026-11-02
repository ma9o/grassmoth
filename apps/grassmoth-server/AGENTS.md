# Study service architecture

Read the [notebook guidance](notebooks/AGENTS.md) for work in `notebooks/`.

- The service owns reference resolution, LLM extraction workers, runtime configuration, compute dispatch and persistence, as defined by the [library/service boundary](../../docs/guides/public_study_library.md). Delegate scientific computation to the [scientific library](../../packages/grassmoth/AGENTS.md).

- Service code and maintenance tools import the library through `grassmoth.public`; the [public import checker](scripts/checks/check_public_imports.py) owns ARCH013. Keep reference-bearing [action inputs](src/grassmoth_server/actions/contracts/inputs.py), [request envelopes](src/grassmoth_server/actions/contracts/requests.py) and [polling envelopes](src/grassmoth_server/actions/contracts/polling.py) in the service; resolved scientific inputs and outputs belong to the library.

- Apply the root functional-core rules according to [module roles](scripts/checks/architecture_roles.py), including pure modules inside the service. Keep I/O and runtime acquisition in the designated edge and shell owners.

- Export contracts from their Python owners and typed routes through [the API exporter](scripts/codegen/export_api.py). [Story scenarios](../../docs/guides/codegen.md#story-scenarios) use production response models and serialization; their synthetic values are interface examples, never scientific evidence.
