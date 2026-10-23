# grassmoth

[![CI](https://github.com/ma9o/grassmoth/actions/workflows/ci.yml/badge.svg)](https://github.com/ma9o/grassmoth/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776ab?logo=python&logoColor=white)
![Next.js 16](https://img.shields.io/badge/Next.js-16-000?logo=next.js)
![NumPyro + JAX](https://img.shields.io/badge/NumPyro-JAX-9b59b6)

**grassmoth** is an opinionated LLM harness for end-to-end Bayesian causal inference on longitudinal data.

The ultimate goal of the project is to facilitate epistemically optimal decision-making at the individual level, using dense digital trace datasets (medical records, chatbot conversation logs, browsing history, etc.) while transparently incorporating existing scientific knowledge, where available, in the form of prior distributions and modeling assumptions.

The framework uses defaults and a constrained modeling language to limit ad hoc choices made to fit the wording of a question. Its [assumptions and limits](docs/assumptions.md) make those scientific commitments explicit.

The user supplies a scientific question and observational data. The agent carries out the scientific work through seven public actions: [edit the question](docs/assets/action-flows/edit-question.svg), [edit the model](docs/assets/action-flows/edit-model.svg), [prepare data](docs/assets/action-flows/prepare-data.svg), [fit](docs/assets/action-flows/fit.svg), [simulate](docs/assets/action-flows/simulate.svg), [compare data](docs/assets/action-flows/data-diff.svg), and [compare models](docs/assets/action-flows/model-diff.svg). Every study starts by setting its question, whose outcome and queries the model is built to answer. Constructs, causal structure, measurements, fixed or estimable parameters, and probability laws can be revised together or incrementally. Fitting conditions the selected continuous-time nonlinear state-space model; simulation samples its current laws before or after fitting. [Numeric causal claims](docs/assumptions.md#numeric-causal-claims) require identification and matching evidence from a converged production fit.

In practice, the framework is designed for longitudinal consumer datasets that are easily accessible via data subject access requests (DSARs), from more domain specific ones like Apple Health, Oura, 23andMe, Strava (cardiometabolic health, chronic conditions, performance & adaptation), Anki, Duolingo, YouTube (education & deliberate practice) to more cross cutting ones like Google Takeout, WhatsApp, ChatGPT/Claude logs (mental health, cognition & attention, habit & behavior change) - and, most interestingly, their intersections!

## Features and Goals

- **Methodological rigor without friction** - An user should be simply able to provide a dataset and a question, and the software should provide the most rigorous possible answer without pushing any methodological decision onto the user.
- **Interpretability and interactivity** - Every model revision, check and result is recorded and can be inspected in the read-only web UI; users steer the agent in conversation through its CLI harness.
- **Support for large datasets, irregular timestamps and semantic heterogeneity** - via a continuous-discrete nonlinear state-space model — continuous-time latent dynamics observed at discrete, irregular timestamps — with non-Gaussian indicator-specific likelihoods (Poisson, Bernoulli, Beta, etc.).
- **Incremental numerical modeling and prior elicitation** - model revisions can be small and interleaved; the [model-edit chart](docs/assets/action-flows/edit-model.svg) shows the checks performed on each specification.
- **Fast and accurate parameter and state estimation in `jax`** - Exact inference in minutes using [parallel-in-time particle smoothing](https://arxiv.org/pdf/2401.14868) on GPU. Efficient caching ensures that we never waste time waiting for compilation.
- **Compatible with `codex` and `claude-code`** - Use your existing agent subscription to drive the study through its public actions.

## Demo

[Open the HEALTHDEMO model](https://project-n98yx.vercel.app/v2/HEALTHDEMO)

## Modeling

Causal identification on the SSM is achieved by temporal unrolling the DAG as per [Jahn et al. (2025)](https://proceedings.mlr.press/v275/jahn25a.html) then running the [ID algorithm](https://doi.org/10.1016/j.artint.2008.12.006) on the unrolled segment for each treatment-outcome pair.

The system is a continuous-discrete nonlinear state-space model: the latent constructs evolve in continuous time as a stochastic differential equation, observed at discrete and possibly irregular times. The drift is a composite vector field — per-construct decay and intercepts plus linear, saturating (Hill), and bilinear edges — so the dynamics are nonlinear in general:

<!-- docs-latex:start eyJkaXNwbGF5Ijp0cnVlLCJsYXRleCI6ImRcXGJvbGRzeW1ib2x7XFxldGF9KHQpID0gXFxtYXRoYmZ7Zn1cXGJpZ2woXFxib2xkc3ltYm9se1xcZXRhfSh0KSwgdDsgXFxib2xkc3ltYm9se1xcdGhldGF9XFxiaWdyKVxcLGR0ICsgXFxtYXRoYmZ7R31cXCxkXFxtYXRoYmZ7V30odCkifQ -->
<p align="center">
  <img src="docs/assets/generated/latex/display-a10ec81048d5b964206d.svg" alt="LaTeX: d\boldsymbol{\eta}(t) = \mathbf{f}\bigl(\boldsymbol{\eta}(t), t; \boldsymbol{\theta}\bigr)\,dt + \mathbf{G}\,d\mathbf..." width="400">
</p>
<!-- docs-latex:end -->

Observations follow indicator-specific [observation laws](packages/grassmoth/src/grassmoth/artifacts/model/likelihood.py):

<!-- docs-latex:start eyJkaXNwbGF5Ijp0cnVlLCJsYXRleCI6InlfaSh0KSBcXG1pZCBcXGJvbGRzeW1ib2x7XFxldGF9KHQpIFxcc2ltIEZfaVxcIVxcbGVmdChnX2leey0xfVxcbGVmdCgoXFxib2xkc3ltYm9se1xcTGFtYmRhfVxcYm9sZHN5bWJvbHtcXGV0YX0odCkrXFxib2xkc3ltYm9se1xcbXV9KV9pXFxyaWdodCk7IFxcdGhldGFfaVxccmlnaHQpIn0 -->
<p align="center">
  <img src="docs/assets/generated/latex/display-30c67cbe14177fd65ee8.svg" alt="LaTeX: y_i(t) \mid \boldsymbol{\eta}(t) \sim F_i\!\left(g_i^{-1}\left((\boldsymbol{\Lambda}\boldsymbol{\eta}(t)+\boldsymbol{..." width="461">
</p>
<!-- docs-latex:end -->

See [assumptions and limits](docs/assumptions.md) for the structural assumptions baked into the modeling framework and how far its results can be read.

## Quick Start

```bash
bun install --frozen-lockfile
uv sync --frozen --all-packages --group dev

# Environment — set OPENROUTER_API_KEY at minimum
cp .env.example.dev .env

# Start the web viewer and the tool server
# (bun run integration:start runs the full stack, including Temporal and the worker)
bun run dev
```

See the [dev setup guide](docs/guides/dev_setup.md) for full details including environment variables and optional dependencies.

## Documentation

- **How each action works:** [`edit_question`](docs/assets/action-flows/edit-question.svg), [edit_model](docs/assets/action-flows/edit-model.svg), [prepare_data](docs/assets/action-flows/prepare-data.svg), [fit](docs/assets/action-flows/fit.svg), [simulate](docs/assets/action-flows/simulate.svg), [data_diff](docs/assets/action-flows/data-diff.svg) and [model_diff](docs/assets/action-flows/model-diff.svg).
- **[Bayesian workflow](docs/assets/bayesian-workflow.svg):** the book's workflow mapped onto the seven actions, including what is still missing.
- **[Assumptions and limits](docs/assumptions.md):** the modeling commitments behind every result.
- **Supported laws:** the library's [observation laws](packages/grassmoth/src/grassmoth/artifacts/model/likelihood.py) and [prior authoring definitions](packages/grassmoth/src/grassmoth/artifacts/probability/native.py).
- **API:** one POST per action returns running progress or the complete saved result; the remaining routes serve the slim timeline, workspaces and uploads. See the Swagger UI at `http://localhost:8100/api/docs` while the tool server runs, and the generated [study API reference](docs/reference/study-api.md).
- **Guides:** [dev setup](docs/guides/dev_setup.md), [integration testing](docs/guides/agentic_integration_testing.md), [code generation](docs/guides/codegen.md) and [benchmarks](docs/guides/running_evals.md).
