# Inference benchmarks

This is the only README under `benchmarks/`. It owns the suite, the compute
schedule and the current decisions. Benchmarks run as studies through the
server's public actions ([Running](#running)). Agents don't edit this file.
Each run folder holds the JSON request bodies it sends and a `notes.md` for run
records and observations; [Decisions](#decisions) keeps the current decisions.

## Scorecard and tournament

One suite serves two purposes:

- A **scorecard** runs one pinned inference configuration across the
  [cases](#cases).
- A **tournament** compares the current policy with a declared change on the
  case that motivated it. Both sides use the same model, data, seeds and full
  compute schedule.

A configuration joins the policy only if it wins its tournament and the
scorecard does not regress. Previously passing cases must still pass, with no
material loss of efficiency or posterior agreement under the comparison's
predeclared criteria. Close or noisy rankings need a targeted confirmation run.

Accepted choices carry forward. They are reopened only when new evidence
calls for it. What carries forward is the adaptation procedure: every fit learns
its own warmup values and freezes them before the retained draws.

## Compute schedule

The schedule is part of the test. Every fit in every stage uses it:

| Setting | Value |
| --- | ---: |
| Latent particles | 128 |
| Chains | 4 |
| Warmup sweeps per chain | 2,000 |
| Retained draws per chain | 4,000 |
| Hardware | One Modal A10 per fit |

Each fit has a **$2 maximum**. A `fit` request sets the hardware and the time
limit; the server adds 8 CPU cores and 32 GiB of host memory to every GPU fit
([`remote.py`](../apps/grassmoth-server/src/grassmoth_server/actions/fit/remote.py)).
At current Modal rates that is about $1.75 per hour, so keep the time limit at
3,600 seconds or less. Failed attempts keep their cost; save `modal billing
report` beside a stage's results to reconcile the actual charges.

A fit that fails with fewer particles or sweeps says nothing about the sampler,
so:

- Never lower particles, chains, warmup or draws to fit a wall limit, finish
  sooner or get a quick answer. Size the wall limit and memory allowance from
  the schedule, never the reverse.
- A fit that runs out of time or memory is reported as a timeout or rejection.
  Changing the schedule needs the user's approval.
- A smaller run may check that code executes or measure time per sweep. Its
  numbers are never stage results.
- A run below the schedule doesn't count, whatever it shows.
- **The retained-draw schedule is a ceiling.** A fit may declare
  `stop_when_converged`. The production convergence check then runs only at
  multiples of `checkpoint_draws`, never before `min_draws` per chain, and
  sampling stops at the first pass. Warmup is unchanged. Report the stopping
  draw, the passing checkpoint and the time to reach it, and label comparisons
  with fixed-length runs. A passing checkpoint is a finite-chain diagnostic, not
  a guarantee against optional-stopping bias.

## Cases

**The control.** C0 has two nonlinear stochastic paths with cubic drift and a
known alternating input. Both paths get a Normal reading at every interval,
over 16 days on a 2,048-interval grid. Diffusion scales are fixed, and one
input effect is unknown.

**The other cases.** Each changes C0 in declared ways and may combine
difficulties. Per-construct and per-time diagnostics locate a failure. When
they and CPU kernel traces can't attribute it, the case's conditional contrast
does: one counterfactual fit, run only then.

| Case | Difficulty | Conditional contrast |
| --- | --- | --- |
| C0 | Control | Last comparable passing policy |
| NL | Strong smooth single-well drift on x, measured before fitting | Moderate strength on the same nonlinearity scale |
| MODE\* | Small fixture with demonstrated competing posterior well histories | Informative readings in the ambiguous window |
| HD | Eight coupled states, two never read, fixed couplings | Restore their readings at the same dimension |
| OBS | Tight dense and Poisson readings on x, sparse noisy readings on y, and one window with no readings | Remove the limiting channel |
| PRIOR | 4,096 intervals over the same span, and one small positive diffusion | Restore that diffusion at the fine grid |
| PARAM | Learned diffusion scales, cross effect and reading noise, with two independent indicators on x | Fix the implicated parameter group |
| TAIL | StudentT readings on both paths, sharing one learned degrees of freedom | C0's Gaussian readings at matched central precision |

**What each case pins and checks.** Each declaration pins its model, data,
priors, seeds, observation masks, validity evidence and contrast. The rules:

- **NL** must meet a preregistered criterion before any fit. The exact
  one-step transition-mean remainder, in process-noise units, is summed over
  one local relaxation time, `ceil(1 / (dt · |∂f/∂x|))` steps. Its median over
  fixed perturbations must exceed 1 at no fewer than 10% of eligible starts, on
  the truth path and on at least five of nine prior paths. If a fixture misses,
  change the drift, never the bar.
- **MODE\*** runs only after cheap checks show posterior ambiguity. A prior
  switch probability between 0.2 and 0.8 is necessary but not sufficient. If
  MODE can't be validated, report multimodality as untested.
- **HD** first gets a timing-only probe at the full chain and particle counts.
- **TAIL** must meet its
  [preregistration](cases/suite/TAIL/preregistration.json) before any fit. On
  the truth path, each channel needs 2–20% of its readings where the log density
  is convex, at least half near the centre, and at least three near the largest
  convex curvature. The same data must show both curvature signs on at least six
  of nine prior paths.
- **C7** (an interval-mean reading channel) and **C8** (observed deterministic
  stages) are CPU checks. They confirm what the current policy supports or
  refuses.
- **France** is the end-to-end study, run after the cases covering its
  difficulties pass.
- **Untested interactions stay explicit**, such as learned diffusion under
  sparse readings.

**Model documents.** C0's are in [`cases/stage00`](cases/stage00):
[`truth.json`](cases/stage00/truth.json) sets every coefficient,
[`fixed.json`](cases/stage00/fixed.json) frees only the input effect and
[`learned.json`](cases/stage00/learned.json) also frees both diffusion scales.
The other cases' documents are in [`cases/suite`](cases/suite), and the scorecard
requests and run notes are in [`suite`](suite). Their status:
- **NL and MODE are valid** after a redesign. NL uses a bounded-slope drift.
  MODE's two same-well readings leave about 0.70 of the posterior mass on
  staying and 0.30 on visiting the other well and returning, under the exact
  Euler–Maruyama law. The records are in [`suite/notes.md`](suite/notes.md).
- **TAIL is valid, and both of its fits fail.** About a quarter of the
  latent time points never move in retention; see
  [suite/notes.md](suite/notes.md).
- **C7** passes a zero-sweep public fit with its interval-mean channel intact;
  this checks admission and compilation, not convergence. **C8** remains refused
  by the current policy; supported model limits are in
  [the model assumptions](../docs/assumptions.md#model-class).

**Data.** In the study, `edit_model` with the truth model and one `simulate`
(seed 721, 16 days on a 675-second clock) give C0's history. No action thins a
simulation, so sparse or missing channels are uploaded through `prepare_data`.
A `simulate` option for an observation schedule would change the HTTP contract,
which needs the user's approval.

## Policy

The current policy is MPGibbs with these settings:
- two parameter particles;
- random-walk parameter proposals;
- Gaussian initialization at scale 0.1, with its proposal covariance adapted in
  warmup windows;
- twisted Particle-aGRAD path updates with κ = 1 and 128 particles.

The latent step adapts in warmup under a per-coordinate observation-curvature
bound and freezes before the retained draws. The `fit` requests pin every setting.

Tournament candidates include:
- dSMC with Particle-RWM;
- Particle-aGRAD;
- κ = 0;
- the MPGibbs `companion` moves, `centred_mh` and
  `centred_and_scale_noncentred_mh`, owned by the
  [sampler spec](../packages/grassmoth/src/grassmoth/artifacts/sampling.py).
  Their exactness checks are in
  [`test_mpgibbs_companion.py`](../packages/grassmoth/tests/models/ssm/inference/test_mpgibbs_companion.py).

## Judging

- A case passes when the library's
  [production convergence check](../packages/grassmoth/src/grassmoth/models/ssm/inference/diagnostics/convergence.py)
  passes. Finishing, acceptance rates and jump sizes don't count.
- Arms that both pass are compared on bulk and tail effective samples per total
  execution second, including initialization, compilation and warmup. Their
  posteriors must agree within Monte Carlo uncertainty.
- Per-construct and per-time diagnostics locate failures. Saved evidence and CPU
  kernel traces come before conditional contrasts or more GPU work. Forward
  ESS, particle movement and smooth-mode refresh never replace the production
  check.
- Rejected, timed-out, errored, refused, invalid and unrun cases are reported as
  such, not as sampling results.
- Truth recovery is reported, not gated: each case is one simulated history.
- NUTS runs only as a declared diagnostic, at most once per case, when no arm of
  that case converges at the schedule. It isn't a reference fit.

## Running

All benchmark fits go through one study on the server, calling the public
actions over HTTP as the
[study API reference](../docs/reference/study-api.md)
describes. Start the stack as the
[integration testing guide](../docs/guides/agentic_integration_testing.md)
says. The server saves every result with its logs, phase timings and code hash,
keeps a fit running if the caller stops, and shows the study in the viewer at
`/v2/{workspaceId}`. There is no benchmark runner or harness of our own.

- **Each case is a model revision** made by `edit_model`. Its data comes from
  `simulate`, or from `prepare_data` for thinned readings.
- **Each arm is a `fit`** carrying the schedule, `hardware: "A10"` and the time
  limit. Arms of the same case run at the same time on separate GPUs.
- **A scorecard runs C0 first.** If C0 fails, no further GPU fits run.
  Otherwise the independent, valid cases run concurrently on separate A10s,
  preserving their preparation and initialization dependencies.
- **Reserve costs before launching.** Count each fit against the run's total
  budget and the $2-per-fit limit, including overhead and retries. Launch in
  waves when the reservations don't fit, and never shorten the schedule to fit a
  budget.
- **Every run must have diagnostics enabled,** including timing checks and
  repeats.
- **Identical requests return the saved result,** success or failure. Use
  `repeat` for a deliberate timing repeat. A call's identity ignores the code
  version, so a patched sampler runs in its own checkout with its own server and
  study, per the isolated-checkout rule in [AGENTS.md](../AGENTS.md).
- **Call the library directly** only to measure what no action exposes, such as
  kernel timing or initializer scoring.

The [model assumptions](../docs/assumptions.md#model-class) own the
scientific limits.

## Decisions

Run records go in the run folder's `notes.md`: date, study workspace, the calls'
IDs, outcome per case and the proposed decision. This section keeps the current
decisions, newest first.

### Second scorecard — 2026-10-24

This scorecard ran on master `4572dc7b` at the full schedule, with early
stopping at 1,000-draw checkpoints. It is recorded in
[suite/notes.md](suite/notes.md).
- **Passed:** C0 and NL at 1,000 draws per chain; OBS and HD at 2,000.
- **Failed at the 4,000-draw ceiling:**
  - MODE: x and y ESS, while every chain switches between both histories;
  - PRIOR: y ESS only;
  - PARAM: all six parameters, and y.

The latent step stays as it is. Its rule is documented in `_curvature_limits`
in [adaptation.py](../packages/grassmoth/src/grassmoth/models/ssm/inference/samplers/particle/adaptation.py),
and the matrix-step and dynamics-floor experiment was not adopted. Each
failing case gets one change at a time. A change must be derived from that rule
or from saved evidence, predict its effect, and be tested on at least two seeds.

### Suite — 2026-10-24

The user replaced the incremental ladder with the scorecard and tournament
suite above. The user also accepted twisted Particle-aGRAD with κ = 1 as the
policy's path update. Three commits make it work:
- `ff4f0e47` builds the guide from the readings and the auxiliary path's actual
  law;
- `7e7062e8` bounds the warmup step by observation curvature;
- `e8ab964c` gives each free coordinate its own curvature bound.

**The tournament that motivated it** ran on C0 on 2026-10-23 at the full
schedule, on the code before those commits.
- The dSMC policy failed on y's effective sample size, at 181 against 400.
- Both κ = 0 guided methods passed.
- Both κ = 1 methods failed when their step adapted to the cap.

**Diagnostic fits after the fixes.** These used a shorter schedule, so they are
not results under this README's schedule. Twisted κ = 1 reached about 15
worst-latent samples per second on C0 and on three one-change variants, against
about 2–3 for κ = 0. Run records are in
[00_current_sampler/notes.md](00_current_sampler/notes.md).

**First scorecard.** It is recorded in [suite/notes.md](suite/notes.md). C0 and
NL passed, and OBS, MODE and PARAM failed. PRIOR and HD finished sampling but
lost their results to a transport fault, fixed in `dcfb3698`.

### Workflow — 2026-10-22

Benchmarks run as one study through the server's public actions, as described in
[Running](#running). The library runner (`stage.py` with its declarations,
runtimes, fixtures and tests) is retired once stage 00's C0 work is finished; the
C0 runs it made stand. In the same change this folder moves out of the library to
a top-level `benchmarks/`: the study needs the server running, and the folder
holds no code, so it isn't a package.

### 00 — blocked at C0, superseded by the suite on 2026-10-24

At the full schedule on an A10, C0 fails only on latent effective sample size:
worst-coordinate bulk ESS 338 for x and 230 for y against the required 400. Every
R̂ is under 1.05, and the input effect is 0.60 [0.44, 0.77] against a truth of
0.7. This run predates covariance adaptation in A0. Later stages stay blocked
while C0's latent-chain dependence is diagnosed. The schedule stays fixed, and
C0's one NUTS diagnostic is available. Runs, costs and the earlier
below-schedule run are in [00_current_sampler/notes.md](00_current_sampler/notes.md).

### 01 — adaptation accepted into A0, 2026-10-23

The user accepted
[covariance adaptation](../packages/grassmoth/src/grassmoth/models/ssm/inference/warmup/parameter_adaptation.py),
always on since 868ef169, into the policy and dropped the fixed-versus-adapted
comparison. The one C0 comparison, run on 2026-10-22 at 256 particles, failed in
both arms with mixed ESS changes ([notes](00_current_sampler/notes.md)). C6 moved
to stage 03.
