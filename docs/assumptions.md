# Assumptions and limits

These are the scientific commitments behind every result: what the model can express, what identification assumes, and how far each output can be read. The [action charts](../README.md#documentation) show where each commitment is checked. The library defines the supported [observation laws](../packages/grassmoth/src/grassmoth/artifacts/model/likelihood.py) and [prior authoring vocabulary](../packages/grassmoth/src/grassmoth/artifacts/probability/native.py).

## Model class

- Latent constructs evolve as a continuous-time stochastic differential equation. The drift sums expressions over states and coefficients. These include per-construct dynamics such as decay and restoring potentials, and per-edge mechanisms such as linear and saturating (Hill) effects. Products express moderation, so the drift is nonlinear in general. Diffusion is additive, and fitting and simulation support only Gaussian diffusion.
- Indicators are observed at discrete, possibly irregular times through indicator-specific [observation laws](reference/statistical-model-spec/likelihoods.md).
- `fit` targets this model with Euler–Maruyama transitions between consecutive prepared time points, using the true nonlinear drift and the true emission density. Interval observations use the [shared response aggregation](../packages/grassmoth/src/grassmoth/models/ssm/execution/observation_operator.py) on that grid, before applying their observation law. The [particle tree](../packages/grassmoth/src/grassmoth/models/ssm/compile/interval_tree.py) includes each interval factor once, after its complete window is available. The [compiled state support](../packages/grassmoth/src/grassmoth/models/ssm/compile/state_space.py) preserves zero process noise and fixed initial values. The dSMC path update under marginalized PGibbs requires deterministic recurrences to be affine in the deterministic coordinates; dependence on stochastic coordinates may remain nonlinear. It rejects other deterministic recurrences; joint NUTS reconstructs them sequentially through the exact model. Exact parallel prefixes reconstruct supported accumulators, including their downstream effects. With fixed state dimension, particle counts and bounded interval overlap, the dSMC trajectory update has logarithmic parallel depth; this is not a wall-time guarantee on fixed hardware. Linearizations and Gaussian approximations appear only in sampler proposals and the optional auxiliary reference below, never in the target, and every move they shape is corrected exactly against the true model, so their quality affects mixing, not the posterior; time discretization and interval quadrature remain numerical approximations.

- [Particle-aGRAD and twisted Particle-aGRAD](../packages/grassmoth/src/grassmoth/models/ssm/inference/samplers/particle/guided.py) provide sequential conditional-filter path updates under the same marginalized parameter ensemble. They preserve the nonlinear target, including exact point constraints, noisy interval readings and deterministic recurrences, and currently exclude coordinate blocks and tempering. Their forward pass has linear depth. Models with deterministic recurrence stages select a terminal particle and trace its ancestors, with exact per-label recurrence states and weights. This supports nonlinear stages feeding later drift without quadratic backward replay; work is linear in the number of times for fixed particle, label and memory sizes. Its cost is weaker early-time movement from genealogical degeneracy. Nearly uniform twisted weights may limit coalescence but do not guarantee movement; native particle-identity updates, frozen fractions and latent chain diagnostics must assess it. Other models retain backward selection, replaying crossing windows and exact auxiliary factors; its work is linear for bounded window spans and overlap. The twisted guide's information scan has logarithmic depth for fixed augmented dimension. Recurrences and window statistics inherit shared state innovations with no added process noise. The single Gaussian guide is shared symmetrically across the parameter ensemble; its coefficients are built at the refreshed auxiliary path, including the guide's own recurrence replay. True densities and full proposal corrections determine all weights. Strong nonlinear drift, long memory, informative future readings and separated modes can reduce mixing. These methods follow [Corenflos and Finke](https://arxiv.org/abs/2401.14868), except that twisted Particle-aGRAD with κ = 1 builds its local proposal and twist from a Gaussian approximation of the readings times the auxiliary path's actual law, not the paper's unshifted pseudo-observation; causal auxiliary scores vary the current independent coordinates with the prefix fixed. Marginalizing the auxiliary variable for Particle-mGRAD is deferred.
- Guided MPGibbs automatically uses [StudentT precision augmentation](../packages/grassmoth/src/grassmoth/models/ssm/inference/targets/student_precision.py): each observed point or completed window receives an independent Gamma precision conditional on the current exact reading residual. Given these precisions, the path weights, auxiliary scores and guide use the exact conditional Gaussian reading factors and the labels include their Gamma prior densities. Missing readings contribute nothing; overlapping windows keep one precision per emitted reading. The parameter ensemble and its fixed-pilot proposal remain independent of these precisions. After the conditional path update, precisions are discarded and the final parameter label is drawn from the original marginal StudentT density; optional companions and returned caches also use that marginal target. This ordered partial collapse avoids a label move conditioned on thousands of precisions when learning shared degrees of freedom, without requiring a companion ([van Dyk and Park](https://www.ma.imperial.ac.uk/~dvandyk/Research/08-jasa-pcg.pdf), [StudentT augmentation analysis](https://arxiv.org/abs/2109.01726)). Adapted κ = 1 point proposals cap their frozen base step using conditional curvature and the unordered ensemble; fixed steps remain literal, while general window/replayed geometry stays frozen. Precisions and effective caps are temporary and add no checkpoint state. dSMC remains marginal. The augmentation preserves the StudentT model for every positive degree of freedom; nonlinear dynamics, label-dependent reading geometry, nonlinear window summaries, finite ensembles and separated modes can still limit mixing.
- The [MPGibbs parameter proposal](../packages/grassmoth/src/grassmoth/models/ssm/inference/samplers/mpgibbs/kernel.py) uses either a symmetric random walk or a bounded pseudo-Langevin drift from a fixed construction pilot. The drift is bounded in the proposal covariance metric and its actual Gaussian asymmetry is paid in every label weight; the nonlinear target is unchanged. Undefined pilot scores define zero proposal drift, while undefined or infinite candidate densities receive zero selection mass. The pilot remains fixed through warmup, retention and continuation. Stale directions, poor covariance estimates, concentrated path conditionals and separated modes can still reduce efficiency.
- Optional [parallel tempering](../packages/grassmoth/src/grassmoth/artifacts/sampling.py) connects the exact posterior to a [frozen Gaussian reference](../packages/grassmoth/src/grassmoth/models/ssm/inference/warmup/gaussian_reference.py), following the [geometric reference path of Syed et al.](https://arxiv.org/abs/1905.02939). The reference is Gaussian in unconstrained parameters and independent, unfixed path coordinates; deterministic states are reconstructed through the exact model. It defines auxiliary replicas, not an approximate posterior. All posterior draws, convergence diagnostics, observation likelihoods and predictions use only the exact cold replica. Reference quality and communication between replicas still limit exploration: tempering does not guarantee discovery of distant modes, and a Gaussian built from initialization may miss them.
- `simulate` and the automatic predictive checks integrate the same equation with Diffrax Heun. Stochastic solver steps end at every prepared time and subdivide further when the resolution heuristic requires it; the step budget never coarsens that schedule. Each step gets an independent Brownian increment, replayed across paired intervention arms. Stochastic readings therefore use solver endpoints. Heun and the fitted Euler–Maruyama target still carry different discretization error: fine-grid resolution checks compare standardized innovations with unit variance and zero lag-one correlation, without making the two schemes identical. Authored trajectory clamps and deterministic ODE dense output retain their interpolation semantics. <!-- cspell:ignore Heun -- proper name of the numerical method. -->
- Missing observations are masked out of the likelihood, which assumes the missing-data mechanism is ignorable.
- Each study is fitted on its own, with no pooling across subjects or studies. Static subject-level states ([A5](#a5-time-invariant-constructs-are-static-subject-level-states)) are therefore not random effects in the multilevel sense.

## Causal graph

The model is one connected DAG over constructs. A theorized common cause appears as an explicit latent confounder node; the graph has no bidirected edges. Indicators belong to their construct and never cause it.

Endogenous means modeled, with or without causal parents. Every latent construct is endogenous. Exogenous means supplied by a deterministic trajectory law in the model, with no endogenous dynamics, diffusion, initial-state coefficients or noise coupling; it cannot be an effect. Its indicators are optional; any indicators use direct exact readings, `Delta(v=state(x))`, which are evidence about the supplied trajectory.

| Role | Temporal status | Example |
| --- | --- | --- |
| Exogenous | Time-varying | Recorded dose, daily stressor count |
| Exogenous | Time-invariant | Exactly recorded baseline age |
| Endogenous | Time-varying | Mood, stress, sleep quality |
| Endogenous | Time-invariant | Latent person intercept, stable susceptibility |

Declared edges specify direct causal parents. Their [temporal interpretation](../packages/grassmoth/src/grassmoth/models/causal/identification.py) follows the constructs’ temporal status.

The model is built to answer the study [question](../packages/grassmoth/src/grassmoth/artifacts/model/question.py), which every study sets first. The question names its outcome and intervention targets by construct identity before any model exists, and every edited model must define them. When the question names an outcome, every other construct must have a directed causal path to it; undirected connectivity alone is insufficient. Upstream causes and explicit confounders need not lie on a selected intervention's path to the outcome. Its outcome scopes execution after structural projection: only the retained connected component is compiled, fitted and simulated. An unmeasured root is projected out as correlated noise or a shared baseline factor, except a confounder that blocks identification of an effect on the outcome, which must be measured or modeled before the model runs.

### A3. Markov dynamics

Time-varying endogenous states follow first-order Markov dynamics: the current state summarizes earlier history. Explicit delay equations and higher-order lags are not modeled. Residual autocorrelation can indicate missing state dynamics or confounders. First-order within-person dynamics are the standard starting point in [dynamic SEM](https://doi.org/10.1080/10705511.2017.1406803).

An exogenous construct's native `Delta` law supplies its trajectory on the [joint law's time coordinates](../packages/grassmoth/src/grassmoth/artifacts/probability/joint_layout.py). Each value holds until the next point, including beyond the final point; requesting time before the first point is an error. Time-invariant inputs require a constant path. Fitting pins that path on every grid point, including input changes between observations, with no initial or transition density. Exact input readings must agree with the trajectory under their declared measurement support; they never define it. Simulation uses the model's path in both arms; an intervention overrides it from its date until the next intervention. Deterministic inputs are omitted from simulation checks and data comparisons involving a simulation. Stochastic exogenous laws are currently unsupported; uncertainty must be modeled as an endogenous construct.

### A4. Acyclic within a time slice

Static relationships form a DAG. Relationships between evolving states connect successive slices in the identification graph, allowing dynamic feedback. Every evolving state has carryover, independently of its causal role or observation status.

### A4b. Effects between evolving states act through the drift

Edge mechanisms evaluate the current state in the continuous-time drift. A cross-slice arrow expresses state evolution; it does not impose a measurement-clock delay on those equations. Identification and the v2 graph retain the declared direct relationships. Same-time co-movement can arise from explicit latent confounders or diffusion correlation.

### A5. Time-invariant constructs are static subject-level states

A time-invariant construct keeps its initial value over the modeled window. Its uncertain value is sampled once and reused; numerical noise must not turn it into a changing trajectory. It may be exogenous or endogenous, but its causes must also be time-invariant, because a changing cause cannot determine a fixed child. Static states separate stable between-person differences from within-person dynamics; failing to separate them [biases lagged effects](https://doi.org/10.1037/a0038889).

## Measurement

### A1. Reflective indicators

Each construct causes its indicators. Formative measurement, where indicators define the construct, is not supported. Indicators of one construct should therefore correlate, and removing one does not change the construct's meaning. Formative models suit composite indices rather than theoretical constructs; see [Diamantopoulos & Siguaw (2006)](https://doi.org/10.1111/j.1467-8551.2006.00500.x).

### A6. Indicator count sets how measurement error is handled

With two or more indicators, shared variance separates measurement error from construct variance; see [Bollen (1989)](https://doi.org/10.1002/9781118619179). With one indicator, [A9](#a9-a-single-indicator-absorbs-measurement-error) applies. The distinction affects precision, not identifiability.

### A8. Indicator errors are independent over time

All temporal dependence belongs to the constructs, so indicators get no autoregressive structure. Serially correlated residuals signal misspecification, such as a clock that is too coarse, missing cross-loadings or dynamics in the measurement itself; see [Asparouhov, Hamaker & Muthén (2018)](https://doi.org/10.1080/10705511.2017.1406803). Correlated indicator residuals are not supported.

### A9. A single indicator absorbs measurement error

A construct with one indicator is identified with that indicator: its loading is fixed, and measurement error merges into the structural noise. This is the standard fallback described by [Bollen (1989)](https://doi.org/10.1002/9781118619179). Its coefficients can be attenuated toward zero. Prefer several indicators when attenuation matters.

## Time

Each prepared history has one origin for model day zero. For files, it is the explicit source-span start, or otherwise the earliest support boundary across the full panel. Selecting a different set of model indicators does not move it. The initial-state law applies at day zero, and fitting includes that point even when the first observation is later. The conditioned trajectory law retains the known calendar instant with its coordinates. Relative input laws acquire this same binding when conditioned.

A simulation replicate keeps its calendar timestamps and starts a new history: its origin is the source report's origin plus its first model time, and its model days restart at zero. Calendar-free histories remain explicitly calendar-free; any synthetic epoch is only a serialization convention. The [simulate chart](assets/action-flows/simulate.svg) defines how requested starts use the available initial states.

| Concept | Meaning |
| --- | --- |
| `measurement_clock` | The shared measurement interval and default authoring interval for persistence and effect priors. |
| `observation_window` | The [indicator-owned support interval](../packages/grassmoth/src/grassmoth/artifacts/observations/spec.py), resolved to exact UTC boundaries. Calendar months and years retain their actual lengths. |
| `aggregation` | How a window is summarized, which fixes the value's support and anchor (next table). |
| `anchor_time` | Where the value attaches to the latent path. |
| `dt` | The time between consecutive prepared time points, including window boundaries. It scales each Euler–Maruyama step in `fit`. |
| Simulation window | See the [simulation contract](../packages/grassmoth/src/grassmoth/artifacts/simulation.py) and [simulate chart](assets/action-flows/simulate.svg). |

| `aggregation` | What matters | Support | Anchor |
| --- | --- | --- | --- |
| `mean` | Average level over the window | Interval | Window end |
| `sum` | Cumulative amount | Interval | Window end |
| `count` | Event frequency | Interval | Window end |
| `std` | Within-window instability | Interval | Window end |
| `last` | Most recent state | Point | Window end |
| `first` | Earliest state | Point | Window start |

These choices are substantive: a daily mean mood and an end-of-day mood encode different theories of what matters. Neither a point summary nor an exact (`Delta`) observation implies that a value persists between observations.

A simulation window, like each question query, has one calendar date, its `start`; the window lasts a `horizon`, and interventions sit `after` offsets from the start. A retained endogenous trajectory law's calendar origin places the start in model days. Otherwise the start is day zero, where the authored initial-state law applies. Input paths use their own dated coordinates or relative model days; simulation takes no panel reference.

A historical simulation start still uses the selected revision's current joint law, including later observations used to fit it. Changing `start` does not undo conditioning or restrict the simulation to information available at that date.

## Causal identification

Identification is checked separately for each treatment's effect on the question's outcome, using nonparametric do-calculus and the ID algorithm of [Shpitser & Pearl (2006)](https://aaai.org/Papers/AAAI/2006/AAAI06-191.pdf). One unidentified effect does not affect the others. Linear instrumental-variable arguments are not accepted, because the nonlinear model does not make the linearity assumptions they need.

### A3a. Scope of the two-slice check

Identification concerns a specified intervention–outcome query. The temporal segment used to decide identification and the effect's response horizon are distinct. For periodic causal graphs with finite maximum direct and latent lag, [Jahn, Karnik & Schulman (2025), Theorem 1 and Algorithm 2](https://arxiv.org/html/2504.20172v1) bound the required past and give a finite procedure for deciding identification across all future outcome shifts. Their bounds depend on graph width and maximum lag; they do not establish universal two-slice sufficiency.

Identification of an appropriate interventional transition law can also support propagation to later outcomes: see [Dynamic Causal Networks, Theorem 1 and Section 4.1](https://www.cs.upc.edu/~gavalda/papers/gilles2017.pdf#page=8), under confounding confined within individual slices. That paper's “static hidden confounders” are slice-local; a persistent time-invariant confounder in this framework does not automatically satisfy that condition.

The [current identifier](../packages/grassmoth/src/grassmoth/models/causal/identification.py) uses a fixed two-slice graph. It does not verify a sufficient-window bound or identify the interventional transition law required for the propagation result above. Its finding establishes identification within that truncated graph; extending it to the full temporal process requires assumptions or a proof that the implementation does not verify. A bounded check can support longer-horizon effects when that justification is supplied.

### A7. Identified measurement lets constructs count as observed

Under A1, A6 and A9 the construct covariance is identified from the indicators, so graph identification can treat constructs as observed. This is the two-step logic of [Anderson & Gerbing (1988)](https://doi.org/10.1037/0033-2909.103.3.411). It requires the graph and the measurements to come from the same model revision.

The graph users see stays a DAG with explicit latent confounders. Internally, the check unrolls it to two slices, projects it to an ADMG as in [Richardson & Spirtes (2002)](https://doi.org/10.1214/aos/1031689015), and runs y0's identification algorithm. The projection never leaves the check.

## Parameter anchors

Every retained construct has exactly one location anchor and one scale anchor. Any shift or rescaling of a latent state that the anchors leave free must be absorbed by exactly one free parameter group. Two such groups would create an exact likelihood ridge, and none would over-constrain the model. A model whose complete retained constructs break this invariant is refused.

A model whose free coefficients enter only through a sum, or only through a product, in every use cannot be constructed: the observed-data law depends only on that combination, so no data can separate them. Two linear mechanisms on one edge, `a·X + b·X`, are the common case. The [check](../packages/grassmoth/src/grassmoth/artifacts/model/coefficient_redundancy.py) expands each equation into terms and tests the rank of the linear and multiplicative forms exactly; it misses redundancies it cannot see and never reports one that is absent. Passing it is not an identifiability proof. Redundancy that depends on the data, such as a cause the record holds constant, is reported against the question instead, because an intervention can break it.

- **Location** is anchored by the first of these that applies:
  1. A standardized channel: a mean-centered Gaussian or Student-t identity-link indicator whose intercept is fixed at zero.
  2. A direct exact state observation, `Delta(v=state(x))`.
  3. For a time-invariant construct, a fixed initial mean.
  4. For an evolving construct, a complete restoring expression with a fixed center.

  Fixed means and centers need not be zero. Observation intercepts and thresholds are identified relative to the anchor; free latent-side locations require a standardized or exact state-observation anchor. The need to constrain competing location parameters follows the ctsem discussion in [Driver, Oud & Voelkle (2017)](https://doi.org/10.18637/jss.v077.i05).
- **Scale and sign:** the reference indicator's loading is fixed to ±1 by its polarity, the marker-variable convention of [Bollen (1989)](https://doi.org/10.1002/9781118619179). A construct measured only by categorical indicators instead pins its first non-baseline slope to +1, following the nominal-response model of [Bock (1972)](https://doi.org/10.1007/BF02291411). Reference indicators are chosen in this order: continuous, ordinal, binary or count, then categorical.

Some parameter pairs are only weakly separated, and the invariant deliberately doesn't gate them. No diagnostic currently flags them either:

- transient versus equilibrium location when dynamics are slow;
- static baseline-factor SDs in a single-subject fit;
- observation dispersion versus latent volatility;
- diffusion correlation versus fast reciprocal edges under coarse sampling;
- Hill `Emax` versus `EC50` when the input never spans the half-saturation point.

## Numeric causal claims

`simulate` reports a numeric effect only when all of these hold:

- The treatment's effect on the question's outcome is identified within the [implemented graph check's scope](#a3a-scope-of-the-two-slice-check).
- The simulated model was conditioned by a committed production fit, using an exact joint sampler with the nonlinear Euler–Maruyama target, and it matches that fit's record exactly.
- That fit passed its [convergence checks](assets/action-flows/fit.svg), within the [sampler diagnostics' limits](#reading-results). A fit with failed or unavailable required criteria still saves, but its effects are withheld. Non-blocking R-hat warnings do not withhold an effect.
- The simulated paths are finite.

Otherwise it still returns the generated histories and records why no effect is reported. Interventions are dated state assignments, after which the model's dynamics resume. Effects are reported for the question's outcome only. Certification covers the identified treatment and outcome; it does not add a separate identification proof for a general timed joint intervention.

Certification does not look at predictive checks or LOO. Read those before trusting a reported effect.

## Reading results

- **LOO**, when configured, estimates how well each measurement row is predicted from all other rows, including later ones; see [Vehtari, Gelman & Gabry (2017)](https://doi.org/10.1007/s11222-016-9696-4). It measures interpolation, not forecasting; forecasting needs [leave-future-out validation](https://doi.org/10.1080/00949655.2020.1783262), which is not implemented. LOO is omitted for exact observations of uncertain latent coordinates; known inputs and fixed accounting origins do not disable it. LOO-PIT is not computed.
- **Predictive checks** of fitted laws against the panel they were fitted on are in-sample. A different panel revision does not prove the observations were held out. Neither case establishes held-out calibration.
- **Sampler diagnostics** cover parameters and every independent, unobserved latent coordinate: rank-normalized R-hat, bulk and tail ESS, and MCSE. Latent reports retain the worst time-coordinate result per construct; frozen independent chains fail. Fixed inputs and deterministic copies are excluded. Without retained latent chains, latent convergence is explicitly unevaluated. The gate requires R-hat < 1.05 on every parameter and free latent coordinate, following [RStan’s practical cutoff](https://mc-stan.org/rstan/reference/Rhat.html), plus bulk and tail ESS ≥ 100 per chain. R-hat in [1.01, 1.05) is a non-blocking warning. These diagnostics cannot establish that every posterior mode was explored or provide family-wise error control.
- **Calibration and recovery** are not automated. A `simulate` → `prepare_data` → `fit` round trip mixes sampler error with the gap between the Diffrax solver and Euler–Maruyama.
- **Prior sensitivity** is not measured: there is no power-scaling or posterior-contraction check.
- **The DT-to-CT diagnostic** compares the elementwise conversion of a linear reference matrix, built from the priors, with its matrix logarithm. Its warning says nothing about the full nonlinear system.
- **C3 resolvability** is a screen that compares `1 / decay` with the observation gaps and span. It is not a nonlinear relaxation analysis or an identification test.
- **Data pattern warnings**, such as dominant duplicate values or arithmetic sequences, flag possible fabrication; they do not prove it.
- **Question checks** compare each query with the model and the record. An intervention value outside an exactly read input's recorded range, or a record that holds the input constant, means the answer there comes from the model's assumptions, not from this person's data.
