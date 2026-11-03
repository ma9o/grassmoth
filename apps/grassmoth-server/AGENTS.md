# Study service architecture

Read the [notebook guidance](notebooks/AGENTS.md) for work in `notebooks/`.

- The service owns reference resolution, LLM extraction workers, runtime configuration, compute dispatch and persistence, as defined by the [library/service boundary](../../docs/guides/public_study_library.md). Delegate scientific computation to the [scientific library](../../packages/grassmoth/AGENTS.md).

- Service code and maintenance tools import the library through `grassmoth.public`; the [public import checker](../../packages/checks/src/grassmoth_checks/check_public_imports.py) owns ARCH013. Keep reference-bearing [action inputs](src/grassmoth_server/actions/contracts/inputs.py), [request envelopes](src/grassmoth_server/actions/contracts/requests.py) and [polling envelopes](src/grassmoth_server/actions/contracts/polling.py) in the service; resolved scientific inputs and outputs belong to the library.

- Apply the root functional-core rules according to [module roles](../../packages/checks/src/grassmoth_checks/architecture_roles.py), including pure modules inside the service. Keep I/O and runtime acquisition in the designated edge and shell owners.

## Contract exports

- Export contracts from their Python owners, [artifact registrations](scripts/artifact_contracts.py) and [typed routes](src/grassmoth_server/api/studies.py) through [export_api.py](scripts/codegen/export_api.py). Update the owning field or endpoint descriptions and follow the [code generation workflow](../../packages/api-types/AGENTS.md#code-generation); the exporter also owns the study API reference and study API agent skill.
- [type_system_catalog.py](scripts/codegen/type_system_catalog.py) owns generic parameter names, declaration bodies and typed applications. Keep one declaration per Python owner. Preserve separate validation and serialization schemas.
- Use the [result codec](src/grassmoth_server/study/persistence/codec.py) for shared numerical buffers. Read projections select owned values and declared compact report types; they do not filter or reparse scientific values to recreate a response.

## Story scenarios

- Scenarios may use real study responses as well as synthetic examples declared in [story_data.py](scripts/codegen/story_data.py). Either way, [export_story_scenarios.py](scripts/codegen/export_story_scenarios.py) validates each response with production response models and serializes it with the normal MessagePack codec, call identity and dependency owners.
- Generate the scenarios through root `bun run codegen` without a running stack or executing scientific actions, so real responses must be committed. [Client stories](../grassmoth-client/AGENTS.md#story-scenarios) consume the generated responses; story values are rendering examples, never scientific evidence.
