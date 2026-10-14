# Code Generation

Generated API artifacts and generated documentation have separate ownership and commands.

## API Artifacts

`bun run codegen` exports the Python contracts and API together, then generates the TypeScript types and facade client:

- **Contract graph**: [`artifact_contracts.py`](../../apps/data-pipeline/scripts/artifact_contracts.py), the domain owners, and typed FastAPI routes are the source of truth. [`export_api.py`](../../apps/data-pipeline/scripts/codegen/export_api.py) generates their validation and serialization schemas together in `openapi.json`, including stored roots and generic bodies.
- **Types and client**: [`generate.ts`](../../packages/api-types/scripts/generate.ts) runs [openapi-typescript](https://openapi-ts.dev/node) once. Named contract aliases and generic declarations come from that AST; [openapi-fetch](https://openapi-ts.dev/openapi-fetch/) supplies the runtime client. Upload inputs use native `Blob` values, serialized as `FormData` by the caller.
- **Agent skill**: The exporter generates `nof1-study-api` from the application's endpoint descriptions and [action contracts](../../apps/data-pipeline/src/nof1_causal_lab/api/studies.py).

Native NumPyro distributions use the shared [JSON codec](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/probability/serialization.py) on scientific parameters and compiled sites. Export derives constructor signatures from native distribution arguments and constraints. There is no separate prior-parameter class hierarchy. Numerical arguments belong to their native law; inference and simulation evidence own their numerical values directly. The [result codec](../../apps/data-pipeline/src/nof1_causal_lab/study/persistence/codec.py) shares repeated binary buffers when saving or serving a result. Its Python decoder and the [TypeScript client](../../packages/api-types/src/client.ts) restore those values before consumers read the scientific fields.

```bash
bun run codegen       # regenerate API artifacts
bun run codegen:check # verify API artifact drift
```

Generated API files are committed. Run `codegen` after editing an artifact, read model, study record, endpoint contract, or facade response.

Python generic owners export their parameter names, declaration bodies and typed
applications through [`type_system_catalog.py`](../../apps/data-pipeline/scripts/codegen/type_system_catalog.py).
The TypeScript generator emits one generic declaration per owner and uses its
applications directly. Concrete JSON Schema definitions still validate each
specialization. The facade client reads the same generic metadata to reference
the canonical declarations. Generic operands come from Python types; generated
schema names are never parsed to recover them.

`openapi.json` owns the exported component graph. `x-contract-roots` identifies stored payloads and public read contracts; `x-typescript-generics` references component bodies with `x-typescript-parameters`. Components retain `x-python-module`, `x-layer` and `x-concern` for the type diagram. `panel` combines JSON metadata with a Parquet file declared by the artifact contracts. The [artifact contracts](../../apps/data-pipeline/scripts/artifact_contracts.py) name durable facts; [action outcomes](../../apps/data-pipeline/src/nof1_causal_lab/actions/contracts/effects.py) retain computed reports and checks, which [readers load from Git](../../apps/data-pipeline/src/nof1_causal_lab/study/persistence/history.py).

## Documentation Artifacts

`bun run docs:codegen` updates the generated distribution reference sections, then rewrites math in `README.md` and `docs/` (`$...$`, `$$...$$`, `\(...\)`, `\[...\]`) as SVG embeds under [`docs/assets/generated/latex`](../assets/generated/latex). The source is retained in nearby `docs-latex` metadata comments because GitHub math rendering is unreliable across Markdown contexts.

```bash
bun run docs:codegen # regenerate documentation artifacts
bun run docs:check   # verify documentation drift, Markdown, and spelling
```

`bun run lint` runs both drift checks with every other static check.

### Action-flow charts

Each chart in [`docs/assets/action-flows`](../assets/action-flows) is a native Excalidraw scene, and the scene is the source. After editing a scene, export its SVG, then render and inspect the result. Use `edit_model` as the visual reference and keep its individual check nodes.

Keep the shared styling: 16 px sans-serif body text, monospace titles and outcomes, solid 2 px outlines, rounded stacked cards, slate connectors and pale gray section panels. Requests are indigo, warnings amber, failures red and commits green. Skipped or unavailable results and subsequent actions use dashed gray cards.

## Type Naming Conventions

Name types for the role their instances serve. Python contracts and generated
TypeScript exports use the same names.

| Role | Convention | Examples |
|------|------------|----------|
| Declarative model or configuration definition, consumed by validation or compilation | `...Spec` | `DynamicalModelSpec`, `ParameterSpec`, `LikelihoodSpec`, `DynamicsSpec` |
| Symbolic formula or formula node | `...Expression` | `StateExpression`, `CoefficientExpression`, `BinaryExpression` |
| Compiled implementation or execution state | `Compiled...` or `...Runtime` | `CompiledDynamicalModel`, `CompiledDynamics`, `CompiledObservationModel`, `ObservationSupportRuntime` |
| Executable mathematical operation | Name the operation | `ObservationKernel`, `ObservationOperator`, `VectorField` |
| Validation, identification, or inference findings | `...Report` or a specific finding name | `IdentificationReport`, `InferenceReport`, `ValidationIssue` |
| Completed operation output | `...Result` | `CausalEffectResult`, `PriorPredictiveResult` |
| Recorded observation or event | `...Record` or `...Event` | `ObservationRecord`, `RuntimeEvent` |
| Persistent scalar identity | `...Id` | `ConstructId`, `IndicatorId`, `ParameterId` |
| Structured reference | `...Ref` | `ConstructRef`, `GitRef`, `ParameterRef` |
| Exact study version | `...Revision` | `StudyRevision` |
| API request and its input values | `...Request` or `...Input` | `SimulateRequest`, `SimulationSpec` |
| Public action input and output bodies | `<Action>Input` and `<Action>Output` | `SimulateInput`, `SimulateOutput`, `ModelDiffInput`, `ModelDiffOutput` |

Public action input and output types are paired in
[`actions/contracts/bodies.py`](../../apps/data-pipeline/src/nof1_causal_lab/actions/contracts/bodies.py),
with request and polling envelopes defined separately.

The scientific definition types are `DynamicalModelSpec`, `ConstructSpec`, `CausalEdgeSpec`,
`IndicatorSpec`, `DriftMechanismSpec`, `PotentialMechanismSpec`, `LikelihoodSpec`,
`ObservationLawSpec` (the closed union of per-family law specifications),
and `ParameterSpec`. A spec may be partial during authoring, complete before
execution, or enriched with conditioned distributions after inference. Its suffix
describes its declarative role throughout those revisions.

Name fields and variables by the same role: `dynamical_model_spec`,
`compiled_dynamical_model`, and `dynamical_model` for the instantiated Dynestyx
value. Use `dynamical_model_spec_ref` for references and `model_snapshot` for
snapshots. TypeScript locals and view properties use the corresponding camelCase
names; generated API fields retain their Python names.

Use the scientific nouns in prose and UI labels: "construct", "indicator", and
"observation law". Keep identity names and serialized field names tied to those
concepts, such as `ConstructId`, `ConstructRef`, `indicators`, and `law`.
The suffix belongs to the definition type's name.

Choose names by meaning rather than by the base class or whether an object is
serializable. An expression represents a formula; a runtime object supplies
execution operations; a report records findings. These roles keep their own names.
When renaming a public type, update its consumers and regenerate the API artifacts
without retaining compatibility aliases.

### Identity and revision conventions

Use a scalar ID when a field identifies one known kind of entity: scenario
`target` and `outcome`, question `outcome`, and validation `indicator_id`.
Keep tagged references for mixed entity kinds.
[`DynamicalModelSpec`](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/model/dynamical_model_spec.py)
owns ID-keyed construct, edge and parameter definitions. Its edges refer to scalar
construct IDs; resolved Python edges hold canonical `ConstructSpec` objects.
The same document represents creation, partial edits and materialized models.
See the [edit action chart](../assets/action-flows/edit-model.svg) for assembly,
pruning and validation order.

Scientific IDs are nominal Python string subclasses whose constructors and
Pydantic schemas enforce the same grammar. Construct IDs explicitly when
allocating identities; parse serialized input at its owner or external boundary.
The constructor proves the ID grammar. Resolve entity membership against the
selected model. Named JSON Schema
definitions preserve the corresponding TypeScript ID types.

Entity identity survives renames and revisions. Exact provenance remains separate:
`GitRef` records a workspace, a Git object ID and a path within that object;
`StudyRevision` records one commit of the study history with its action log. A
commit is not a model artifact version.
Snapshots carry `workspace_id`, `selected_seq`, and the artifact selections from
recorded action dependencies. Scientific values are returned directly; action
records own their input references and produced artifact revisions.

## Changing the schema

Workflow: **edit Python → `bun run codegen` → commit both**.

Follow the [type naming conventions](#type-naming-conventions). `DynamicalModelSpec` is the
scientific definition retained inside the owning [action result](../../apps/data-pipeline/src/nof1_causal_lab/actions/contracts/bodies.py); `ConstructSpec`, `IndicatorSpec`, and
`ParameterSpec` retain their canonical ownership inside it, while [`Value`](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/base.py) supplies the shared immutable
base. Server-composed views and study records share this export. Frontend code
owns presentation and [display-only law composition](../../AGENTS.md#web-app).

Owned values expose tuples and read-only mappings. Generated outputs have
readonly properties, arrays and tuples; an ID-keyed map preserves its named key
type as `Readonly<Partial<Record<Id, T>>>`. A lookup may be absent. A serialized
defaulted field is required, including a nullable field emitted as `null`.
Validation schemas describe inputs independently, so an input default may be
omitted. Fields excluded from serialization do not appear in output schemas.

[Observation specifications](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/observations/spec.py)
share one generic owner for authored and resolved windows. Prepared metadata,
saved simulations and dataset readers consume the resolved specialization.
[Simulation effects](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/reports/effects.py)
and [predictive comparisons](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/reports/data_comparison.py)
own their result alternatives; consumers narrow their discriminator before reading
the payload or its absence reason.

[Assessments](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/reports/checks.py) carry a producer's typed subject and evidence, or its explicit reason for unavailable evaluation. Consume those alternatives directly. Scientific classifications and inference plot series come from the backend. The [law composition renderer](../../apps/web/src/lib/model-asset/flow-plots.ts) evaluates saved laws and expressions from `DynamicalModelSpec` solely for display; those plot points are not saved action results. [Inference reports](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/reports/inference.py) compose a compact core with full plot detail; snapshot fields declare the core type, so serialization omits detail without filtering or reparsing owned values.

- **New/changed field**: edit the owning Python model.
- **New artifact contract**: add the payload class in `artifacts/`, register it in `ARTIFACT_CONTRACTS`, add re-export in `index.ts`.
- **New/changed endpoint**: update its typed FastAPI contract and regenerate the OpenAPI schema and agent skill.

## File ownership

| File | Source |
|------|--------|
| `src/generated/models.ts` | Generated — do not edit |
| `src/generated/model-api.ts` | Every OpenAPI operation, referencing canonical domain declarations |
| `src/client.ts` | Runtime client factory using the generated operation signatures |
| `src/generated/metadata.ts` | Generated artifact IDs, file layout, and distribution metadata |
| `src/index.ts` | Hand-written re-exports |

## Client Version

`openapi-fetch` is pinned to 0.16.0 because [0.17.0's response mapping](https://github.com/openapi-ts/openapi-typescript/blob/main/packages/openapi-typescript-helpers/index.d.ts) widens fixed
tuples into arrays. The client type test checks that a fetched batch retains the
canonical `ModelSnapshot` contract, including posterior draw dimensions.

The [client type test](../../packages/api-types/src/model-snapshot.type-test.ts)
also compares all generated paths and methods with the exported OpenAPI schema.
Web and API consumers enable `noUncheckedIndexedAccess` and
`exactOptionalPropertyTypes`: sparse lookups retain absence, and omitted request
options stay absent rather than being assigned `undefined`.

[Fixture generation](../../apps/data-pipeline/scripts/fixtures/study.py) emits
`.d.json.ts` declarations alongside canonical JSON projections. They reference
the production domain types without copying payloads or declaring another
schema. `fixture:build` owns both outputs. Stored results and
trace bytes pass through their Python contract owners before export;
the declarations preserve discriminators and scientific IDs in JSON imports.

## Troubleshooting

- **Optional vs required mismatch**: `Value` owns serialization presence through Pydantic configuration; check the field's validation and serialization schemas separately.
- **Circular imports**: artifact contracts import other contracts; numerical implementations import those contracts. Keep the `artifacts` package initializer free of re-exports.
