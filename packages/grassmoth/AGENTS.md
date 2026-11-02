# Scientific library architecture

- Scientific actions receive resolved values and return complete scientific outputs through their [owned contracts](src/grassmoth/actions/contracts/io.py). Keep the library independent of the service, including type-only imports, as required by the [library/service boundary](../../docs/guides/public_study_library.md).

- [grassmoth.public](src/grassmoth/public.py) re-exports the agreed actions, boundary values and helpers from their implementation owners; it contains no logic. Keep internal implementation behind that surface.

## Scientific model

- Never switch GPU computation to float64; GPU work stays in float32.

- Represent structural assumptions as DAGs with explicit latent confounders. ADMGs are only for internal projection into y0's identification algorithm, never user-facing.

- The latent SSM is continuous-time **nonlinear**, and every target is the exact nonlinear model. Never replace a nonlinear drift, emission or recurrence with a linear or Gaussian one in a target, a likelihood, a diagnostic or a reported quantity.
- Linearizations and Gaussian approximations may shape proposals at any stage of sampling, as long as the move stays exact:
  - its correction (Metropolis–Hastings ratio or particle weights) uses the true densities;
  - the approximation is built only from what the move holds fixed (the auxiliary point u, or the parameters during a path update), unless the acceptance ratio also accounts for rebuilding it around the proposed state;
  - adapting it from the chain's history happens only during warmup;
  - each such proposal documents where it loses efficiency (poor fit, multiple modes). A bad approximation may cost mixing, never correctness.
- Production posteriors, all diagnostics, posterior-predictive checks, and counterfactual/predictive outputs must use the exact engines in [docs/assumptions.md](../../docs/assumptions.md#model-class).
- [test_no_linearized_likelihood.py](tests/models/ssm/test_no_linearized_likelihood.py) enforces the first rule: it fails if a linearized likelihood backend appears anywhere in production source.
