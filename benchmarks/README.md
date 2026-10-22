# Incremental inference benchmarks

This is the only README under `benchmarks/`. It owns the ladder, the cases, the
compute schedule and every stage's decision. Benchmarks run as one study through
the server's public actions ([Running](#running)). Agents don't edit this file.
Each stage folder holds the JSON request bodies it sends and a `notes.md` for run
records and observations; [Decisions](#decisions) keeps each stage's current
decision.

## How the ladder works

The ladder builds one sampler policy a step at a time, so every result can be
attributed to one change.

1. **A stage starts from its parent's accepted policy and changes one thing.**
   A sampler stage compares the parent policy with the parent policy plus one
   capability, on the same cases, seeds and schedule. A model stage runs the
   accepted policy on cases that each add one difficulty to a case that passed.
2. **Nothing runs on top of a failure.** Cases run in the listed order, control
   first. The first case where no arm passes production convergence stops the
   stage, and later cases and stages wait until that failure is diagnosed and
   fixed. The fix becomes the next comparison, even when it isn't the next
   planned stage.
3. **Accepted choices carry forward and are not re-tested.** Later stages don't
   cross an accepted choice with its alternatives. A few passed cases stay as
   regression checks. An earlier choice is reopened only when a later failure
   points to it.
4. **Each fit tunes itself.** Warmup adapts every step size and preconditioner
   for that fit and freezes them before the retained draws. What carries forward
   is the adaptation procedure, never values estimated on another case.

The result is the value of each addition on top of the ones before it, in this
order. It is not a search for the best combination of all methods.

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
([`remote.py`](../../../apps/grassmoth-server/src/grassmoth_server/actions/fit/remote.py)).
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

## Cases

All cases share one model family: two nonlinear stochastic paths with cubic
drift, a known alternating input and Normal readings, over 16 days on a
2,048-interval grid. Each case differs from its parent in one declared way.

| Case | Parent | Change | Stages |
| --- | --- | --- | --- |
| C0 | — | Control: fixed diffusion scales, one unknown input effect, a reading at every interval | 00, then regression everywhere |
| C1 | C0 | Learn both diffusion scales; same truth and readings | 00, 02 |
| C2 | C0 | Keep every 32nd reading of both paths plus the endpoints; the input stays fully known | 00 |
| C3 | C1 | Apply C2's thinning | 00, 02 |
| C4 | C3 | Twice the intervals (4,096) at the same span, readings, priors and input | 03 |
| C5 | C1 | σy 0.35 → 0.0225 with regenerated readings; this also changes signal-to-noise | 03 |
| C6 | C0 | Also learn the cross effect, at its existing truth; same readings | 01 |
| C7 | C0 | Add one interval-mean reading channel | 03 |
| C8 | C0 | Add observed deterministic stages: nonlinear between stages, affine within, no feedback into the paths | 03 |
| France | — | FRANCE1940V2's pinned model and observations | 04 |

C0–C3's model documents are in [`cases/stage00`](cases/stage00):
[`truth.json`](cases/stage00/truth.json) sets every coefficient,
[`fixed.json`](cases/stage00/fixed.json) frees only the input effect and
[`learned.json`](cases/stage00/learned.json) also frees both diffusion scales;
nothing else differs. The other cases still need theirs.

In the study, `edit_model` with the truth model and one `simulate` (seed 721,
16 days on a 675-second clock) give the shared history. C0 and C1 fit that
`simulate` result directly. No action thins a simulation, so C2 and C3 fit
readings uploaded through `prepare_data`: the same simulated values, keeping
every 32nd reading of both paths plus the endpoints, and every input reading. A
`simulate` option for an observation schedule would change the HTTP contract,
which needs the user's approval.

## Stages

| Stage | Parent | Change tested | Cases, in order | Arms |
| --- | --- | --- | --- | --- |
| 00 Current sampler | — | None: find where today's sampler first fails | C0, C1, C2, C3 | A0 |
| 01 Parameter adaptation | 00 | Proposal covariance learned during MPGibbs warmup instead of the fixed 0.1·I | C0, C6, then 00's failing case | A0 with fixed vs adapted covariance |
| 02 Companion moves | 01 | Extra parameter moves after each sweep | C1, C3 | A0, A1, A2 |
| 03 Model stress | 02 | One added difficulty per case | C4, C5, C7, C8 | A0 |
| 04 Real study | 03 | The real study | France | A0 |

- **A0** is the parent's accepted policy. In stage 00 that is today's sampler:
  MPGibbs with dSMC, two parameter particles, one latent path redrawn per sweep,
  random-walk parameter proposals, Gaussian initialization at scale 0.1, an
  adapted latent step, no aMALA and no companion. The stage's `fit` requests pin
  every setting.
- **A1** adds a centred MH move on the parameters after each sweep.
- **A2** follows A1's move with a scale-only non-centred MH move: the diffusion
  scales change while the path is rebuilt from fixed innovations. A2 − A1 is the
  value of the non-centred move, including its cost.

Each companion adapts its own step size during warmup. Covariance adaptation and
both companions still need implementing in the sampler, and their exactness
checks belong with that code. Stage 01 keeps the initializer and starting values
identical in both arms and compares one regularized dense covariance with the
fixed one; there is no separate diagonal step. Stage 04 starts with a short
timing run at the schedule's particle and chain counts to size the wall limit,
then runs the full fit.

**Latent-path branch.** When paths stop mixing under the current policy, such as
one path barely moving under the shared step or both stuck along the grid, the
next comparison is the current dSMC proposal against a changed one, on the case
that showed it. A C5 failure alone doesn't qualify, because C5 also changes
signal-to-noise. An accepted change joins the policy, and the ladder resumes
where it stopped.

## Judging

- A case passes when the library's
  [production convergence check](../src/grassmoth/models/ssm/inference/diagnostics/convergence.py)
  passes. Finishing, acceptance rates and jump sizes don't count.
- Arms that both pass are compared on effective samples per total execution
  second, including initialization, compilation and warmup, and must agree with
  each other.
- Rejected, timed-out and errored fits are reported as such, not as sampling results.
- Truth recovery is reported, not gated: each case is one simulated history.
- NUTS runs only as a declared diagnostic, at most once per case, when no arm of
  that case converges at the schedule. It isn't a reference fit.

## Running

All benchmark fits go through one study on the server, calling the public
actions over HTTP as the
[study API skill](../../../.agents/skills/grassmoth-study-api/SKILL.md)
describes. Start the stack as the
[integration testing guide](../../../docs/guides/agentic_integration_testing.md)
says. The server saves every result with its logs, phase timings and code hash,
keeps a fit running if the caller stops, and shows the study in the viewer at
`/v2/{workspaceId}`. There is no benchmark runner or harness of our own.

- **Each case is a model revision** made by `edit_model`. Its data comes from
  `simulate`, or from `prepare_data` for thinned readings.
- **Each arm is a `fit`** carrying the schedule, `hardware: "A10"` and the time
  limit. Arms of the same case run at the same time on separate GPUs; cases
  follow the ladder order and stop at the first failure.
- **Identical requests return the saved result,** success or failure. Use
  `repeat` for a deliberate timing repeat. A call's identity ignores the code
  version, so a patched sampler runs in its own checkout with its own server and
  study, per the isolated-checkout rule in [AGENTS.md](../../../AGENTS.md).
- **Call the library directly** only to measure what no action exposes, such as
  kernel timing or initializer scoring.

The [model assumptions](../../../docs/assumptions.md#model-class) own the
scientific limits.

## Decisions

Run records go in the stage folder's `notes.md`: date, study workspace, the calls'
IDs, outcome per case and the proposed decision. This section keeps each stage's
current decision.

### Workflow — 2026-10-22

Benchmarks run as one study through the server's public actions, as described in
[Running](#running). The library runner (`stage.py` with its declarations,
runtimes, fixtures and tests) is retired once stage 00's C0 work is finished; the
C0 runs it made stand. In the same change this folder moves out of the library to
a top-level `benchmarks/`: the study needs the server running, and the folder
holds no code, so it isn't a package.

### 00 — blocked at C0

At the full schedule on an A10, C0 fails only on latent effective sample size:
worst-coordinate bulk ESS 338 for x and 230 for y against the required 400. Every
R̂ is under 1.05, and the input effect is 0.60 [0.44, 0.77] against a truth of
0.7. Stage 01 stays blocked while C0's latent-chain dependence is diagnosed. The
schedule stays fixed, and C0's one NUTS diagnostic is available. Runs, costs and
the earlier below-schedule run are in [00_current_sampler/notes.md](00_current_sampler/notes.md).

### 01–04 — not started
