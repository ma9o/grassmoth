# grassmoth

[![CI](https://github.com/ma9o/grassmoth/actions/workflows/ci.yml/badge.svg)](https://github.com/ma9o/grassmoth/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](/LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776ab?logo=python&logoColor=white)
![Next.js 16](https://img.shields.io/badge/Next.js-16-000?logo=next.js)
![NumPyro + JAX](https://img.shields.io/badge/NumPyro-JAX-9b59b6)

**grassmoth** is an opinionated LLM harness for end-to-end Bayesian causal inference on longitudinal data.

<kbd>![The FRANCE1940SIMPLE study after a prior-predictive simulate: action history, causal model from the legal hours cap to cumulative aircraft with simulated draws in each node, the question's outcome, simulated states and observations, simulation checks and the action log](/docs/assets/screenshots/france1940-model.png)</kbd>

> Would the Battle of France have gone differently had the French not limited the work week to 40 hours? ([Willemyns & McClements, 2026](https://juliawillemyns.substack.com/p/nobody-decides-to-decline))

The harness exposes a DSL of six actions over a [dependency graph of versioned artifacts](/apps/grassmoth-server/src/grassmoth_server/study/state.py): the agent declares the desired next state, the framework resolves the dependencies and constraints to reach it, and the user follows progress and inspects results in the web UI. Because state is declarative and abstracted at the DSL level, the LLM does no state management and the UI can show every modeling choice with its motivation.

Such restrictions applied to the [Bayesian Workflow](https://users.aalto.fi/~ave/Bayesian-Workflow.pdf) aim to address the tendency of LLMs to adopt different ad-hoc machinery and vocabulary across studies - often chosen with the answer or existing literature in mind - rendering modeling choices hard to compare.

Every study is a [continuous-time nonlinear state-space model](/docs/assumptions.md#model-class), with support for irregular observations - either pointwise or window aggregates - and popular transition functions, likelihood and distribution families. We enforce [first-order Markov](/docs/assumptions.md#a3-markov-dynamics) dynamics with no explicit delays or higher-order lags. This can seem overly restrictive in complex scenarios and overly pedantic in simpler ones, but nonetheless enforces a common structure that fundamentally enables comparisons between different levels of model complexity for the same problem.

## Quick Start

```bash
bun install --frozen-lockfile
uv sync --frozen --all-packages --group dev

# Start the web viewer and the tool server
# (bun run integration:start runs the full stack, including Temporal and the worker)
bun run dev
```

Then open the repository in Claude Code or Codex and give the agent your question and data. The [study API skill](/.agents/skills/grassmoth-study-api/SKILL.md) teaches it the six actions; follow the study in the web UI at `http://localhost:3000/workspaces/<workspace-id>`.

See the [dev setup guide](/docs/guides/dev_setup.md) for full details including environment variables and optional dependencies.

## Documentation

- **How each action works:** [`edit_model`](/docs/assets/action-flows/edit-model.svg), [`prepare_data`](/docs/assets/action-flows/prepare-data.svg), [`fit`](/docs/assets/action-flows/fit.svg), [`simulate`](/docs/assets/action-flows/simulate.svg), [`data_diff`](/docs/assets/action-flows/data-diff.svg) and [`model_diff`](/docs/assets/action-flows/model-diff.svg).
- **[Bayesian workflow](/docs/assets/bayesian-workflow.svg):** the workflow of [Gelman, Vehtari & McElreath (2026)](https://users.aalto.fi/~ave/Bayesian-Workflow.pdf) mapped onto the six actions.
- **[Assumptions and limits](/docs/assumptions.md):** the modeling commitments behind every result.
- **Supported laws:** the [observation law](/docs/reference/statistical-model-spec/likelihoods.md) and [parameter and prior](/docs/reference/statistical-model-spec/parameters.md) references.
- **API:** one POST per action returns running progress or the complete saved result; the remaining routes serve the slim timeline, workspaces and uploads. See the Swagger UI at `http://localhost:8100/api/docs` while the tool server runs, and the generated [study API reference](/docs/reference/study-api.md).
- **Guides:** [dev setup](/docs/guides/dev_setup.md), [integration testing](/docs/guides/agentic_integration_testing.md), [code generation](/packages/api-types/AGENTS.md#code-generation), [public study library](/docs/guides/public_study_library.md) and [benchmarks](/docs/guides/running_evals.md).
- **Local references:** the [reference papers](/docs/papers/README.md) and a searchable [Bayesian Workflow wiki](/docs/wiki/README.md), downloaded and built on demand.

## Credits

- [NumPyro](https://github.com/pyro-ppl/numpyro): distributions, effect handlers and the warmup window schedule.
- [dynestyx](https://github.com/BasisResearch/dynestyx): continuous-time dynamical models, Euler–Maruyama transitions and masked observation likelihoods.
- [Diffrax](https://github.com/patrick-kidger/diffrax): SDE and ODE solvers for predictive and counterfactual simulation.
- [BlackJAX](https://github.com/blackjax-devs/blackjax): step-size adaptation.
- [y0](https://github.com/y0-causal-inference/y0): the ID algorithm behind each identification check.
- [ArviZ](https://github.com/arviz-devs/arviz-stats): ESS, R̂, MCSE and LOO.
- [Pyro](https://github.com/pyro-ppl/pyro): the automatic reparameterization strategies, ported to NumPyro.

The papers behind its methods are cited where those methods appear, above and in [assumptions and limits](/docs/assumptions.md).
