# Likelihoods

Defines the observation-model vocabulary for `LikelihoodSpec` entries in a `DynamicalModelSpec`.

## Conditional Expressions

`LikelihoodSpec.law` is a [closed union of law specifications](../../../packages/grassmoth/src/grassmoth/artifacts/model/likelihood.py), each with a `distribution` tag and required `Expression` fields directly on the law. Expressions use scientific state references, coefficient operands, arithmetic, and bounded functions. In the table, `p = a + Σ bᵢ state(xᵢ)` is an affine predictor; each loading refers to its actual construct. `a`, `bᵢ`, and the auxiliary operands are fixed coefficients or references to `DynamicalModelSpec.parameters`.

| Native law | Argument expressions |
|---|---|
| `Delta` | `v=state(x)` or `v=p` |
| `Normal` | `loc=p`, `scale=s` |
| `StudentT` | `df=ν`, `loc=p`, `scale=s` |
| `Poisson` | `rate=exp(p)` |
| `Gamma` | `concentration=k`, `rate=k / exp(p)` or `k / (1 / p)` |
| `BernoulliLogits` | `logits=p` |
| `BernoulliProbs` | `probs=normal_cdf(p)` |
| `NegativeBinomial2` | `mean=exp(p)`, `concentration=r` |
| `Beta` | `concentration1=m * c`, `concentration0=(1 - m) * c`, where `m=sigmoid(p)` or `normal_cdf(p)` |
| `OrderedLogistic` | `predictor=p`, `cutpoints=ordered_cutpoints(base, gaps)` |
| `Categorical` | `logits=category_logits(p, intercepts, slopes)` |

`ordered_cutpoints` starts at the base threshold and cumulatively adds positive gaps in declared ordinal-level order. For two levels, its gap argument is literal zero. `category_logits` prepends the reference category's zero logit to `intercepts + slopes * p`; the indicator supplies category order and the existing anchor rules apply. These structured arguments retain the existing category-specific parameter identities.

`Delta(v=state(x))` declares an exact measurement, with no loading, intercept or noise parameter to author. Its numerical view derives a unit loading and a zero intercept. The indicator's [aggregation and support](../../assumptions.md#time) determine what is exact: a point value for `first` or `last`, or the declared window summary for interval observations. An exact window mean does not force a constant trajectory, and missing observations impose no equality.

For endogenous constructs, particle inference supports direct point bindings `Delta(v=state(x))`. Observed coordinates are fixed at their recorded times, missing coordinates remain sampled, and the initial-state and transition densities still inform the parameters. Conflicting exact readings of the same state at the same model time are rejected. Affine Delta bindings and endogenous interval-summary constraints are supported in densities and predictive draws, but not yet in fitting. Exact observations of [deterministic inputs](../../assumptions.md#a3-markov-dynamics) are checked against their model-owned trajectories, including interval summaries. PSIS-LOO is omitted when the fitted data contain exact observations, because removing an equality changes the posterior's support.

The numerical backend derives its family and response from these expressions, and it rejects unsupported formulas before fitting.

For the supported families and links, read the [law specifications](../../../packages/grassmoth/src/grassmoth/artifacts/model/likelihood.py). The [measurement dtype definitions](../../../packages/grassmoth/src/grassmoth/artifacts/observations/dtypes.py) own the allowed observation-family combinations.
