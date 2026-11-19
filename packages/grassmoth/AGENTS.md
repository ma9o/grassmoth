# Scientific library architecture

- Scientific actions receive resolved values and return complete scientific outputs through their [owned contracts](src/grassmoth/actions/contracts/io.py). Keep the library independent of the service, including type-only imports, as required by the [library/service boundary](../../docs/guides/public_study_library.md).

- [grassmoth.public](src/grassmoth/public.py) re-exports the agreed actions, boundary values and helpers from their implementation owners; it contains no logic. Keep internal implementation behind that surface.

## Type naming conventions

Name types by the role their values serve. Python contracts and generated TypeScript exports keep the same names; a spec remains a spec through partial authoring, completion and conditioning.

| Role | Convention |
|------|------------|
| Declarative model or configuration | `...Spec` |
| Symbolic formula or formula node | `...Expression` |
| Compiled implementation or execution state | `Compiled...` or `...Runtime` |
| Executable mathematical operation | Name the operation, such as `ObservationKernel` |
| Validation, identification or inference findings | `...Report` or a specific finding name |
| Completed operation output | `...Result` |
| Recorded observation or event | `...Record` or `...Event` |
| Persistent scalar identity | `...Id` |
| Structured reference | `...Ref` |
| Exact study version | `...Revision` |
| API request and its input values | `...Request` or `...Input` |
| Public action bodies | `<Action>Input` and `<Action>Output` |

- Name fields and variables by the same role: `dynamical_model_spec`, `compiled_dynamical_model`, `dynamical_model` for the instantiated Dynestyx value, `dynamical_model_spec_ref` for references, and `model_snapshot` for snapshots. TypeScript locals and view properties use camelCase; generated API fields retain their Python names.
- Use "construct", "indicator" and "observation law" in prose and UI labels, and keep IDs and serialized field names tied to those concepts. Type suffixes describe the definition's role, not a base class or serialization format.
- When renaming a public type, update its consumers and [regenerate API artifacts](../api-types/AGENTS.md#code-generation) without compatibility aliases.

## Values and identities

- Use scalar IDs for one known entity kind and tagged references for mixed kinds. Construct scientific IDs explicitly when allocating identities; parse serialized IDs at their owner or external boundary. The constructor owns grammar, and the selected model owns membership.
- Preserve entity identity across renames and revisions. Keep identity separate from provenance: a study commit is not a model artifact version. Follow [identity owners](src/grassmoth/artifacts/identity.py) and the [edit action chart](../../docs/assets/action-flows/edit-model.svg) for reference meanings and model assembly.
- Keep contracts below their numerical consumers and the `artifacts` package initializer free of re-exports. Shared generic owners, such as [observation specifications](src/grassmoth/artifacts/observations/spec.py), own authored and resolved variants; downstream readers consume the appropriate specialization.
- Native probability laws use the [shared JSON codec](src/grassmoth/artifacts/probability/serialization.py). Keep numerical arguments with their law and derive constructor signatures from native arguments and constraints; do not introduce a separate prior-parameter hierarchy.
- Expected report alternatives retain their typed subject, evidence or absence reason. Consumers narrow the discriminator before reading the payload; [assessment owners](src/grassmoth/artifacts/reports/checks.py) own classifications.

## Scientific model

- Everything is float32, on GPU and CPU alike: library code, tests, checks and experiments. Never enable x64 or compute in float64, including for a reference, a replay or an oracle; compare float32 results with tolerances.

- Represent structural assumptions as DAGs with explicit latent confounders. ADMGs are only for internal projection into y0's identification algorithm, never user-facing.

- The latent SSM is continuous-time **nonlinear**, and every target is the exact nonlinear model. Never replace a nonlinear drift, emission or recurrence with a linear or Gaussian one in a target, a likelihood, a diagnostic or a reported quantity.
- Linearizations and Gaussian approximations may shape proposals at any stage of sampling, as long as the move stays exact:
  - its correction (Metropolis–Hastings ratio or particle weights) uses the true densities;
  - the approximation is built only from what the move holds fixed (the auxiliary point u, or the parameters during a path update), unless the acceptance ratio also accounts for rebuilding it around the proposed state;
  - adapting it from the chain's history happens only during warmup;
  - each such proposal documents where it loses efficiency (poor fit, multiple modes). A bad approximation may cost mixing, never correctness.
- Production posteriors, all diagnostics, posterior-predictive checks, and counterfactual/predictive outputs must use the exact engines in [docs/assumptions.md](../../docs/assumptions.md#model-class).
- [test_no_linearized_likelihood.py](tests/models/ssm/test_no_linearized_likelihood.py) enforces the first rule: it fails if a linearized likelihood backend appears anywhere in production source.
- Every tuning constant and default that shapes sampling or its diagnostics in [inference](src/grassmoth/models/ssm/inference) and [sampler settings](src/grassmoth/artifacts/sampling.py) states its source in code, because comment sweeps remove `#` comments. A setting uses its `Field(description=...)`, a module constant uses `Annotated[T, Field(description=...)]`, and a number inside a computation moves to a named constant or the docstring of the function that owns it. The source is a paper's section or algorithm and where we differ, our measurement (what was compared and the result), or `Heuristic:` when no paper or [recorded benchmark](../../benchmarks/suite/notes.md) sets it. Check each citation against the paper itself; [docs/papers](../../docs/papers) lists the main ones.

## Sampler design

Judge every design for the [exact-posterior sampler](../../docs/assumptions.md#inference-policy) against four principles from the papers it implements, [Corenflos 2025, v4][c25] and [Corenflos & Finke 2024, v1][cf24]. When proposing or evaluating a design, state which principles it follows and which it breaks.

- **Integrate the path out of the parameter move.** Alternating between the path and the parameters given the path mixes slowly when the two are correlated, however good the conditional sampler. The parameter move marginalizes the path instead of conditioning on it ([C25][c25] abstract, §1, §4).
- **Keep the parameter ensemble small.** More candidates per move are not the remedy: the paper uses two, citing the limits of ensemble samplers for static parameters ([C25][c25] §4, §5).
- **Use the time structure.** Gradient or preconditioned moves that treat the whole path as one block ignore that states decorrelate over time, and break down as the series grows. Path updates propose and select per time step with particles; gradients belong inside those per-step proposals ([CF24][cf24] §1.2, §1.3).
- **Build on conditional SMC, not pseudo-marginal acceptance.** Marginal parameter moves come from a conditional particle filter with backward sampling, which stays stable as observations accumulate, not from accepting or rejecting on noisy likelihood estimates ([C25][c25] §1, §5).

[c25]: https://arxiv.org/abs/2505.04611v4
[cf24]: https://arxiv.org/abs/2401.14868v1
