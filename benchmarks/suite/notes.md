# Suite records — 2026-10-24

## Final first scorecard

**The policy does not pass the suite: two native passes, three convergence failures, two infrastructure errors without saved statistical verdicts.** All seven valid fixtures were attempted at the unchanged full schedule on separate A10s, with C0 first, reservations and a passing HD timing forecast. [Complete scorecard](../../scratchpad/suite-scorecard-20261024/findings.md) contains worst latent ESS/second, R̂, every parameter's ESS, truth coverage, frozen-coordinate checks, phase timings, costs and failure localization.

| Case | Native verdict / disposition | Fit-app cost |
|---|---|---:|
| C0 | passed | $0.43489352 |
| NL | passed | $0.43421398 |
| MODE | failed | $0.07927063 |
| HD | infrastructure error; no verdict | $0.70628578 |
| OBS | failed | $0.49272463 |
| PRIOR | infrastructure error; no verdict | $1.18228225 |
| PARAM | failed | $0.44913184 |

Total **$3.86216469 / $10**, including the HD timing probe, first MODE pre-sampling refusal, automatic PRIOR/HD repeats and all Modal CPU work. [Billing reconciliation](../../scratchpad/suite-scorecard-20261024/cost-reconciliation.json). All suite jobs are stopped. PRIOR and HD completed sampling before SIGKILL/137 during result handling; their automatic replacements were stopped to protect reservations. They are not statistical convergence failures. [Infrastructure evidence](../../scratchpad/suite-scorecard-20261024/PRIOR-HD-infrastructure.md).

NL's bounded-slope drift and MODE's exact-EM conditioned-history fixture are valid; their [redesign record](../../scratchpad/nl-mode-redesign-20261024/findings.md) supersedes the rejected initial candidates below. MODE's corrected prepared data retains one null y row at an existing anchor, with no added observation. The selected requests and revisions are in [the manifest](fixture-manifest.json).

Numerical analysis and decoding ran on Modal CPU; subsequent saved-result analyses use one Volume and cached data. New uploads are serialized and paced to ≤2 MB/s. No sampler/library changes or commits. Local tests/lint were deferred at the user's request. The historical record below describes the initial preparation stage only and is superseded by this final record.

## Initial fixture record (before scorecard and redesign)

Fixture preparation only, on master `7e7062e8b85abc76943fefb56308766ef80cc428` in study `SUITE_20261024`. **No scorecard or HD timing request has run.** C0, HD, OBS, PRIOR and PARAM are fixture-valid. **NL is invalid; MODE multimodality is untested. The suite is incomplete.** C7/C8 confirm policy refusals on CPU. Incremental Modal charge: **$0** ([before](billing-before.json), [after](billing-after.json)). No sampler/library changes or commits.

The [full validity report](../../scratchpad/suite-build-20261024/findings.md), [pinned protocol](fixture-protocol.json) and [manifest of revisions/call IDs](fixture-manifest.json) are the review record. All declarations are in [cases/suite](../cases/suite); C0 remains in [stage00](../cases/stage00). NL's saved root documents are rejected q=100 declarations, not a valid selected case.

| Case | Pre-GPU evidence | Disposition |
| --- | --- | --- |
| C0 | Native model/finite simulation; identical NL probe control has zero times above 1 on truth and all 9 prior paths | Fixture valid; first full-schedule request unrun |
| NL | q=100,300,1000,3000 all produce non-finite native simulated paths; seeds and bar unchanged | Invalid, no request; core coverage blocker |
| MODE | 22/128 prior switch events, p=0.171875, Wilson 95% [0.116331,0.246540]; full posterior separated mass not established | Untested, no fit data/request |
| HD | Eight stochastic states retained; z3/z6 channels have zero readings, six others have 2,049 each | Fixture valid; timing required before full fit |
| OBS | 2,049 grid points; each x channel 1,793 readings, y 57; days [6,8) unobserved; active x curvature 631.02–775.22, y 6.25 | Fixture valid |
| PRIOR | 4,097 point times; positive sigma_y=0.0225; stochastic readings unchanged at 2,049 times | Fixture valid |
| PARAM | Native anchors/redundancy pass; independent aligned x indicators; all six priors pinned | Fixture valid, not a general identifiability proof |
| C7 | Public local zero-sweep fit refuses interval observations | Refused by policy |
| C8 | Corrected retained deterministic stages; public local zero-sweep fit refuses recurrence replay | Refused by policy |

HD uses declared potential indicators with every hidden reading masked; deleting those indicators caused native projection refusal. PRIOR uses exact known-input half-step timestamps while retaining the 675s reading clock. C8 places the study outcome at the final deterministic stage so authoring does not prune it. These are declaration/data changes, not framework workarounds in inference.

Requests: [C0](fit-C0-twisted-k1.json), [HD](fit-HD-twisted-k1.json), [OBS](fit-OBS-twisted-k1.json), [PRIOR](fit-PRIOR-twisted-k1.json), [PARAM](fit-PARAM-twisted-k1.json). Every full request pins the [approved policy and schedule](../README.md#policy), seed 19, A10, 3,600s and both supported movement/identity traces. All request bodies validate against the published OpenAPI schema.

[HD timing only](timing-HD-twisted-k1.json): 50 warmup/100 draws, 4 chains/128 particles, A10, 600s, diagnostics enabled. It is unrun and has no scorecard standing.

Review before spend. Run C0 first; stop GPU work if it fails. Then run valid independent cases on separate A10s, with HD full fit after timing. Reserve actual spend plus outstanding jobs, overhead and retries against $10 and $2/fit. Five $2 reservations plus timing already exceed the batch allowance: use waves, never a reduced scorecard schedule. Do not label the resulting partial coverage a completed suite while NL remains invalid.

Validation: root lint passed (7/7 tasks), scratch public-boundary lint passed, native fixture actions and saved-data assertions passed for the five valid cases. C7/C8 were CPU refusal checks only. The browser viewer remained at “Reading the journal…” and browser captures timed out; UI rendering is unverified. The full report records the safe stale-stack restart, persisted Temporal recovery and all failed draft attempts.

## Second scorecard

**Four native passes (C0, NL, OBS, HD); three convergence failures (MODE, PRIOR, PARAM). All seven statistical results are retained.** [Full report](../../scratchpad/suite-scorecard-2-20261024/findings.md) contains stopping checkpoints, ESS and ESS/second, R-hat, parameter truth coverage, native frequency/regional movement, freezing, timings, costs and saved-evidence localization.

The server and fit workers ran master `4572dc7b`. Requests preserve the existing schedule/policy and seed 19, adding only early stopping and the two native movement reports. MODE keeps its own 900-second limit, as pane 5.2 confirmed. [Exact request audit](../../scratchpad/suite-scorecard-2-20261024/request-audit.json), [calls and provenance](../../scratchpad/suite-scorecard-2-20261024/scorecard-manifest.json).

| Case | Native verdict | Retained draws/chain | GPU | Fit app, including host resources |
|---|---|---:|---:|---:|
| C0 | passed | 1000 | $0.15461125 | $0.25506090 |
| NL | passed | 1000 | $0.14727794 | $0.24352920 |
| MODE | failed | 4000 | $0.05530872 | $0.09034894 |
| HD | passed | 2000 | $0.29945626 | $0.50038737 |
| OBS | passed | 2000 | $0.20288886 | $0.33644607 |
| PRIOR | failed | 4000 | $0.57690077 | $0.95732784 |
| PARAM | failed | 4000 | $0.29822205 | $0.49730924 |

Total **GPU $1.76890255 / $10; all-in $3.84615101** including the HD timing probe, CPU services/analysis and failed pre-allocation submissions. All owned apps are stopped and no reservation remains. [Billing reconciliation](../../scratchpad/suite-scorecard-2-20261024/cost-reconciliation.json).

No free coordinate was completely frozen in any fit. MODE explores both preregistered histories but fails x/y ESS; PRIOR fails only y ESS; PARAM fails all six parameter ESS checks and y bulk ESS. Localization uses saved evidence on Modal CPU, with no new GPU contrast after failure.

Comparison with the **first fixed-length scorecard**: C0/NL remain passes, OBS now passes, and HD/PRIOR now retain results (HD passes; PRIOR fails statistically). MODE and PARAM remain failures; MODE also uses the repaired fixture without its prior-driven input-effect parameter. Different stopping lengths change timing amortization, and these are not controlled one-change causal comparisons.

**TAIL is blocked before data generation.** Pinned master refuses literal StudentT df=5 and also refuses its exact Delta-parameter representation. Pane 5.2 chose to record the blocker and finish the suite. No seed search, gate change or TAIL fit occurred. [Declaration/build status](../cases/suite/TAIL/build-status.json), [execution-owner evidence](../../scratchpad/suite-scorecard-2-20261024/TAIL-blocker.md).

Saved results remain on one Modal Volume; no result bodies were downloaded locally or baked into analysis images. All numerical analysis and checks ran on Modal CPU. No sampler/library changes, commits or README edits. [Volume audit and handoff](../../scratchpad/suite-scorecard-2-20261024/handoff.md).

## TAIL — 2026-10-24

**Valid fixture; both fits fail with frozen latent coordinates.** The fixture was built on master `362bb0be`, which accepts the truth's literal df = 5, through the public actions with the unchanged [preregistration](../cases/suite/TAIL/preregistration.json).

**Validity gate on the truth path:**

| Channel | Convex readings | Central readings | Readings near the largest convex curvature |
|---|---:|---:|---:|
| x | 7.6% | 68.1% | 41 |
| y | 6.9% | 70.2% | 36 |

8 of 9 prior paths pass, against 6 required.

**Fits.** Seeds 19 and 29 used the second scorecard's policy, schedule and early stopping, with an 1,800-second limit that pane 5.2 approved. Both reached the 4,000-draw ceiling and failed the native check:
- x and y ESS are 0, with no R̂ available;
- about 550 of each construct's 2,049 chain-time coordinates stay exactly unchanged through retention;
- the lowest-band refresh is 0.15–0.20;
- the input effect passes at both seeds, and df passes at seed 29 only (R̂ 1.10 at seed 19).

Requests: [seed 19](fit-TAIL-twisted-k1-seed19.json), [seed 29](fit-TAIL-twisted-k1-seed29.json), [build status](../cases/suite/TAIL/build-status.json). GPU $0.57.

## Diagnoses of the failing cases — 2026-10-24

**Method.** These come from the second scorecard's saved retained states, measured on Modal CPU with paired single-sweep kernel traces at keys 19 and 29. No sampler change was made.

**The step rule.** For κ = 1, the auxiliary adds K = (1 + A·H)²/A of information about the current path. The per-coordinate step cancels it for concave readings, which is the 2R step for Gaussian ones, and minimizes it for convex ones; see `_curvature_limits` in [adaptation.py](../../packages/grassmoth/src/grassmoth/models/ssm/inference/samplers/particle/adaptation.py). On C0, a fixed-step sweep peaks at 2R: δ = R, 2R, 3R and 4R give 9.2, 15.8, 12.8 and 9.2 worst-latent ESS per second.

**PRIOR (y fails).**
- Leftover auxiliary memory at the unread half-step times is 0.6%, so it isn't the cause.
- The twist linearizes y's nearly deterministic dynamics at the noisy auxiliary path, up to 119 process SDs from the path.
- The filter's minimum ESS over time falls to 1.4–18 of 128.
- As a diagnostic only, anchoring the twist at the current path raises the minimum ESS to about 128 and multiplies y's level movement by 8.7. That anchor isn't valid as a production move.

**MODE (x and y fail).** Two mechanisms:
- **Leftover memory.** At unread times the auxiliary keeps about 80% of y's level: its precision is 51 against a target precision of 12. Raising the unread variance 100-fold raises level jumps about 40-fold.
- **The twist anchor.** Its errors reach 206 process SDs.

Histories switch about every 10 draws, but within a history y's level has a lag-1 correlation of 0.98.

**PARAM (all six parameters fail).**
- The particle filter is healthy: minimum ESS at least 127.7 of 128.
- The step's one-label curvature agrees with the mixture's, so that lead is ruled out.
- σ_y sits in a narrow funnel with the path. Moving it ±10% costs 12–27 log-density units, and a naive non-centred move costs 22–40.
- The effect parameters' disagreement between chains is unexplained.

**TAIL.** The failure is recorded above. Its mechanism hasn't been diagnosed yet.

**Where the second half of the study lives.** The second scorecard and TAIL ran on a Modal-hosted copy of `SUITE_20261024`, keeping the local machine free. Their journal and results are on the Modal volume `grassmoth-suite-scorecard-2-20261024`, at `data/SUITE_20261024/study/history.git`. The local study holds the fixtures and the first scorecard. The two copies can't be merged by fetching refs. Attempts are numbered refs (`refs/attempts/<n>`), and the volume copy kept numbering from its fork point, so its later attempts collide with the local ones. A merge would have to renumber and re-parent the volume's attempt chain. Until a tool does that, the volume copy is the record for the second scorecard, TAIL and the PARAM screen.

## Anchor check and PARAM companion screen — 2026-10-25

**Smoothed twist anchor: not adopted.**
- **What was tested:** linearizing the dynamics at a u-only path, from three fixed Gaussian smoothing passes, instead of at the raw auxiliary path. It ran as paired single-sweep CPU traces at saved states, keys 19 and 29, against a rule declared in advance.
- **PRIOR passed at both keys.** Minimum filter ESS rose from 4.5 and 11.5 to 128, and y's level movement rose 3.3–7.1-fold.
- **MODE failed.** The smoother lands between the two wells, with transition errors up to thousands of process SDs.
- **C0 dipped.** Level-jump ratios of 0.86 and 0.76 fell below the 0.90 tolerance.
- **TAIL partly responded.** 85–93% of its frozen coordinates moved, but the minimum filter ESS stayed at 1. Most frozen readings are central, not in the tails.

**PARAM companion screen.** This is a mechanism check, not a suite result.
- **Design:** the existing exact `centred_nuts` companion against the policy, at 200 warmup and 200 draws, 4 chains, seeds 19 and 29. Requests: [policy 19](screen-PARAM-policy-seed19.json), [policy 29](screen-PARAM-policy-seed29.json), [NUTS 19](screen-PARAM-nuts-seed19.json), [NUTS 29](screen-PARAM-nuts-seed29.json).
- **The effects and both reading SDs mix:** lag-1 correlation falls from 0.97–0.99 to 0.03–0.31, with 3.6–550× larger jumps.
- **σ_y stays slow** (lag-1 0.94–0.95), and σ_x is mixed.
- **Cost:** GPU $0.14.

## PARAM tournament — 2026-10-25

**The `centred_nuts` companion passes PARAM's native gate at both seeds, and the policy alone fails both.**

**Setup.** Master `d3589088`, full schedule, early stopping at 1,000-draw checkpoints, one A10 per fit. Within a seed, the only difference between arms is the companion.

| Arm | Seed 19 | Seed 29 | Numerical seconds |
|---|---|---|---:|
| Policy | failed at the 4,000 ceiling | failed at the 4,000 ceiling | 849, 853 |
| Policy + `centred_nuts` | **passed** at 4,000 | **passed** at 4,000 | 936, 918 |

**What changed with the companion.**

| Parameter | Bulk ESS, companion (seeds 19, 29) | Bulk ESS, policy alone | R̂, companion |
|---|---|---|---|
| Input and cross effects | 8,685–9,363 | 5–204 | below 1.001 |
| Both reading SDs | 9,728–10,362 | — | — |
| σ_x | 963–970 | — | ≤ 1.0065 |
| σ_y (limiting) | 461, 450 | — | ≤ 1.0065 |

The companion adds 7.5–10.3% to the numerical time. GPU cost was $1.28 for the four fits.

Requests: [policy 19](tournament-PARAM-policy-seed19.json), [policy 29](tournament-PARAM-policy-seed29.json), [companion 19](tournament-PARAM-nuts-seed19.json), [companion 29](tournament-PARAM-nuts-seed29.json).

**Same round, on CPU.**
- **Anchor v2** (a u-only smoother with a bounded line search) fails its preregistered rule. PRIOR passes all six checks, but MODE's minimum ESS ratio is 0.92 at key 19, and one C0 bootstrap lower bound is 0.898 against 0.90. It is not adopted.
- **TAIL.** After the anchor, the remaining collapse comes from the gap between the guide and the exact StudentT × auxiliary factor (median 23–26 log units).
- **NegativeBinomial2.** The float32 guide failure reproduces in 12 of 108 extreme scalar cases. All of France's restricted annual-grid probes stay finite.

## PARAM on regenerated data — 2026-10-25

**Regeneration.** After the `simulate` fix (`9986d2fe`), PARAM's history was regenerated with the same truth, seed 721 and reading clock. The old saved result would otherwise have been returned, so a fresh call was forced through an unchanged model revision. The new data revision is `34e8b5a9`, in the volume copy of `SUITE_20261024`.

**Innovation check.** The innovations standardized under the fit's Euler–Maruyama law now look right:

| | SD | Lag-1 correlation |
|---|---|---|
| Before the fix | 0.30 | 0.94 |
| After the fix | 0.98–1.00 | about 0.01 |

**Short screen.** This compared the random walk against random walk + `centred_nuts`: 200 warmup and 200 draws, seeds 19 and 29, with the data revision as the only change from the earlier screen.
- **The effects and both reading SDs** mix with the companion: lag-1 correlations fall from 0.89–0.99 to 0.13–0.33, and jumps grow 3–1,370 times.
- **σ_x and σ_y** stay slow in both arms, at lag-1 0.91–0.94.
- **Truth recovery:** every truth now lies inside the companion's marginal intervals. The joint truth sits at the 58–74% radius rank, against the 99th percentile on the old data. The earlier misses came from the generation defect.

These short runs are not convergence verdicts. GPU cost was $0.12.

**Third arm: pseudo-Langevin.** This used master `b11c87e6`, with the fix that bounds its drift and rejects invalid candidates, on the same regenerated data and schedule; the time limit was 450 s.
- Both seeds now run cleanly, with finite and moving chains.
- It doesn't improve on the random walk. The effects' lag-1 correlations are 0.98–0.99, and their mean squared jumps are 0.08–0.74× the random walk's.
- Of the three parameter moves, only `centred_nuts` mixes PARAM's effects and reading SDs at this schedule. The process scales stay slow under all three.

GPU cost was $0.06.

## SV (stochastic volatility) — 2026-10-29

**New case. The twisted aGRAD policy fails it at the full schedule on one H200.**

**Model.** The declarations are in [cases/suite/SV](../cases/suite/SV). It is the Gaussian stochastic volatility model of Kim, Shephard & Chib (1998) and the Stan User's Guide:
- daily percent returns r_t ~ Normal(0, exp((μ + h_t)/2)), with h an Ornstein–Uhlenbeck log-variance and h₀ ~ N(0, 1);
- readings are squared returns, r² ~ Gamma(½, rate ½·exp(−(μ + h))). |r| depends on no parameter, so the posterior is the Gaussian SV posterior;
- on the 1-day clock, Euler–Maruyama gives h_t = (1 − θ)·h_{t−1} + σ·ε_t;
- truth μ = −0.2, θ = 0.02 (persistence 0.98), σ = 0.2; priors μ ~ Normal(0, 2), θ ~ HalfNormal(0.1), σ ~ HalfNormal(0.5);
- 3,000 daily readings from one `simulate`: seed 721, start 2020-01-01, horizon 2999d.

The StudentT-returns version (Hoffman & Gelman 2014) is not expressible, because a reading's scale cannot depend on a state.

**Fixture validity.**
- The simulated h has lag-1 autocorrelation 0.9845 (truth 0.98) and innovation SD 0.2011 (truth 0.2); its innovations are uncorrelated (0.03).
- All 3,000 readings are positive (minimum 2.8e-8). r²/exp(μ + h) has mean 1.006 and variance 1.94, against χ²₁'s 1 and 2.
- The engine's log joint equals an independent hand-written SV density, plus the native log-Jacobian, to within 0.01 at all four starting and all four final chain states.

**Framework patch.** Fitting any model with a construct that has no edges raised "Cannot replace constructs absent from the graph" after sampling, in `condition_model`. The run used master `615557c4` plus `scratchpad/sv-benchmark-20261029/conditioning-edgeless.patch`, in an isolated checkout with its own server and study.

**Run.**
- Study `SV_20261029` on that isolated stack; its data root is `/Users/ma9o/Desktop/nof1-nuts-da-20261029/data`.
- Call `call:d580a9e7c929235553ee2e9ac6bac086dc15fcc83cb0a5727e43864c11124626`, journal commit `6c39639a`, conditioned model `fa3f569f`.
- Request: [fit-SV-twisted-k1-h200.json](fit-SV-twisted-k1-h200.json). It pins the policy (twisted Particle-aGRAD with κ = 1 and 128 particles, an adapted latent step at 0.5, random-walk parameter proposals with a 0.2 acceptance target, initialization scale 0.1) and the full schedule, with early stopping at 1,000-draw checkpoints, profiling on, one H200 and a 1,200 s limit.

**Outcome: failed at the 4,000-draw ceiling.** No checkpoint passed.

| Quantity | Posterior mean [94% HDI] | Truth | R̂ | Bulk ESS | Tail ESS |
| --- | --- | ---: | ---: | ---: | ---: |
| level μ | −11.0 [−13.3, −8.8] | −0.2 | 3.69 | 4.3 | 4.6 |
| decay θ | 0.0011 [0.0007, 0.0019] | 0.02 | 1.82 | 5.8 | 14.7 |
| vol-of-vol σ | 0.172 [0.159, 0.187] | 0.2 | 2.45 | 4.8 | 4.8 |
| latent h | — | — | 3.01 | 4.5 | 5.8 |

Parameter acceptance was 0.054 against the 0.2 target; the latent update fraction was 0.57.

**Mechanism.** The readings pin only μ + h_t.
- The chains started from the pilot point (θ ≈ 0.003, μ ≈ 0.7–1.2, σ ≈ 0.26–0.28), with log joints of −1,592 to −1,787.
- They climbed to −793 to −1,052, above the truth's −1,235, by smoothing their paths to σ ≈ 0.17–0.19.
- With θ near 0.001, the pull of h toward zero vanished. The chains slid along the μ + h ridge to μ ≈ −8 to −12, with final paths averaging +7.9 to +11.6 and h₀ ≈ 7–11 against its N(0, 1) law.
- Three of the four chains' log joints were still falling during retention.

This is the coupled-parameter weakness seen in PARAM's σ funnel, plus the level ridge. It is not a target error.

**Timing on the H200.** 76 ms per warmup sweep and 87 ms per retained draw; the profiled full sweep took 79.9 ms. Sweep compilation took 22.5 s. Of 578 s of numerical work, warmup took 152 s, profiling 46 s and sampling 350 s. The app ran 13 min 4 s.

**Cost.** $0.972 in total: the fit app $0.961 (H200 $0.835, CPU $0.078, memory $0.047, egress $0.001) and three Modal CPU analyses $0.011. Billing snapshots are in `scratchpad/sv-benchmark-20261029`.

**Results.** The complete body is on the `grassmoth-fit-analysis` Volume, with a receipt in the study. Summaries are in `scratchpad/sv-benchmark-20261029/results` and at `/analysis-cache/sv-benchmark-20261029` on the Volume.

**Proposed next step.** Land the conditioning patch on master first. Then change one move at a time, on two seeds: the `centred_nuts` companion, then `centred_and_scale_noncentred_mh`. Neither directly moves the μ + h ridge, so a level-shift move may also be needed.

## SV: Particle-mGRAD with translation and scale charts — 2026-10-30

**Passes all 12 checks at the 4,000-draw checkpoint on one H200.** It was the first grassmoth arm of the [README comparison](../../README.md#stochastic-volatility), until the [optimized master fit](#sv-particle-mgrad-on-master-with-fused-gpu-kernels--2026-10-30) replaced it. [Findings](../../scratchpad/sv-mgrad-scale-20261030/findings.md).

- **Source:** `77798eb9` (structural and learned translation charts) plus the scale-chart patch, in an isolated checkout, before either was reviewed. Master has since landed the scale chart (`b3c15788`) and replaced the structural chart with the computed chart (`355138e5`).
- **Run:** study `SV_MGRAD_SCALE_20261030`, call `call:a55af53f91d1bfc8f5b6ba7ec3c2cbb546927ca121390df58bdb205fbbf1550c`, journal commit `570c3cd8`. Its inputs match the original SV fixtures exactly.
- **Request:** [fit-SV-mgrad-scale-h200.json](fit-SV-mgrad-scale-h200.json). It pins Particle-mGRAD with 128 particles, κ = 1, scan unroll 8 and the latent step adapted to 0.5; two parameter labels with random-walk proposals at a 0.2 acceptance target; compiler reparameterization (the structural translation chart) and the scale chart; initialization scale 0.1; the full schedule with early stopping at 1,000-draw checkpoints; profiling on; one H200; a 1,200 s limit.
- **Attempts:** the first, at `repeat: 0`, raised a GPU fault in profiling: a cuSolver internal error, then `CUDA_ERROR_ILLEGAL_INSTRUCTION`. The retry at `repeat: 1` passed.

| Quantity | Posterior mean [94% HDI] | Truth | R̂ | Bulk ESS | Tail ESS |
| --- | --- | ---: | ---: | ---: | ---: |
| level μ | −0.195 [−0.687, 0.315] | −0.2 | 1.010 | 617 | 742 |
| decay θ | 0.0146 [0.0067, 0.0222] | 0.02 | 1.007 | 514 | 672 |
| vol-of-vol σ | 0.198 [0.168, 0.229] | 0.2 | 1.012 | 458 | 653 |
| latent h | — | — | 1.013 | 785 | 1,122 |

**Timing on the H200.** 85.8 ms per warmup sweep and 111.4 ms per retained draw, including checkpoints. Sweep compilation took 18.8 s and profiling 33.1 s. The native fit duration was 689.0 s.

**Cost.** $1.489 in total: the passing fit $1.130, the raised attempt $0.353 and a Modal CPU analysis $0.006.

**On master's computed chart,** the same seed narrowly fails one check, decay bulk ESS at 396.8 against 400. That fit predates the IEKS initializer: `call:3491f68d64dff6c2d32573e33305d91c85edea3a7235e691eafc7ed1112488fc`, study `SV_MGRAD_COMPUTED_20261030`. [Findings](../../scratchpad/sv-mgrad-computed-20261030/findings.md).

## SV: NUTS on an M1 Max CPU — 2026-10-30

**Passes all 12 checks at the first checkpoint, 1,000 draws per chain, in 1,664 s.** It was the NUTS arm of the [README comparison](../../README.md#stochastic-volatility) until the [cubic-drift comparison](#sv-with-a-cubic-drift-particle-mgrad-with-80-chains--2026-10-31) replaced it. [Findings](../../scratchpad/sv-nuts-20261030/findings.md).

- **Source:** master `bb38704d`, unpatched, in an isolated checkout with its own server and study.
- **Run:** study `SV_NUTS_20261030`, call `call:bf8fa4ff97b66f67d98c66dc6a4d9ffa934d0ca350d94f07f1ae8a58f111903d`, journal commit `22c9b53e`. Model, question, truth and readings are identical to the mGRAD study's and the original SV fixtures.
- **Request:** [fit-SV-nuts-cpu.json](fit-SV-nuts-cpu.json). It is mGRAD's request with two changes: `nuts_da` in non-centred coordinates with NumPyro's defaults (target acceptance 0.8, tree depth at most 10), and `hardware: local`. Local fits enforce no time limit.
- **Hardware:** Apple M1 Max, 10 cores, 64 GiB, shared with other work at load averages of 5–12.

| Quantity | Posterior mean [94% HDI] | Truth | R̂ | Bulk ESS | Tail ESS |
| --- | --- | ---: | ---: | ---: | ---: |
| level μ | −0.193 [−0.684, 0.311] | −0.2 | 1.004 | 1,182 | 1,574 |
| decay θ | 0.0145 [0.0071, 0.0223] | 0.02 | 1.002 | 970 | 1,209 |
| vol-of-vol σ | 0.198 [0.167, 0.231] | 0.2 | 1.002 | 1,266 | 1,889 |
| latent h | — | — | 1.008 | 1,644 | 1,800 |

Every retained transition had tree depth 6, or 63 leapfrog steps. Mean acceptance was 0.89, with one divergence in retention and 160 in warmup, and BFMI was 0.89–0.95. Warmup took 1,339 s, 670 ms per sweep: each sweep waits for its slowest chain, which averaged 127 steps. Sampling took 317 s, 317 ms per draw. The posterior means agree with both passing mGRAD fits within 0.6 combined MCSEs.

**Why not the H200.** Non-centred NUTS rebuilds the path in a sequential scan at every gradient. Fixed 8-step HMC sweeps on the same source took 521 ms on an H200 and 41 ms on this CPU, or 65 and 5.1 ms per leapfrog step. A full H200 fit projected to 6–11 h and $33–55, so the user chose the CPU. The CPU's per-step cost stayed at 4.9–5.1 ms across load levels, so contention moved the timing by a few percent at most.

**Unrolled scan.** A probe that unrolled the path scan 8×, like mGRAD's `scan_unroll`, ran 3.4× faster on the H200, but every transition diverged. The unrolled decode is exact on CPU, so the fault is in the GPU program. Its patch is in the findings folder, and neither arm uses it.

**Cost.** The fit has no Modal charge. The two H200 probes cost $0.210.

## SV: Particle-mGRAD on master with fused GPU kernels — 2026-10-30

**Passes all 12 checks at the 4,000-draw checkpoint on one H200, in 458.6 s.** It was the grassmoth arm of the [README comparison](../../README.md#stochastic-volatility), against the NUTS fit above, until the [cubic-drift comparison](#sv-with-a-cubic-drift-particle-mgrad-with-80-chains--2026-10-31) replaced it. [Findings](../../scratchpad/sv-mgrad-optimized-20261030/findings.md).

- **Source:** master `a7cb357b`, which fuses GPU resampling and backward selection, in an isolated checkout. It runs the production IEKS initializer and the computed translation chart.
- **Run:** study `SV_MGRAD_OPTIMIZED_20261030`, call `call:cc47bb41febb026cd59d5ff8ad28d87d09e5b1373b7bfc2d4ebc650ef4fb82db`, journal commit `023a12bd`. Model, question, truth and readings are identical to the NUTS study's.
- **Request:** [fit-SV-mgrad-optimized-h200.json](fit-SV-mgrad-optimized-h200.json). It is the first mGRAD request with profiling off, a 900 s limit and no `learned_translation` field, which master removed; compiler reparameterization now builds the computed chart. Against the NUTS request, only the algorithm, hardware, profiling and time limit differ. The NUTS code is the same at `bb38704d` and `a7cb357b`, whose only change is MPGibbs-only.
- **Checkpoints:** 1,000, 2,000 and 3,000 draws failed, and 4,000 passed. Decay's bulk ESS of 408.8 leaves little margin.

| Quantity | Posterior mean [94% HDI] | Truth | R̂ | Bulk ESS | Tail ESS |
| --- | --- | ---: | ---: | ---: | ---: |
| level μ | −0.189 [−0.691, 0.304] | −0.2 | 1.009 | 837 | 995 |
| decay θ | 0.0145 [0.0069, 0.0220] | 0.02 | 1.007 | 409 | 567 |
| vol-of-vol σ | 0.198 [0.167, 0.231] | 0.2 | 1.006 | 446 | 761 |
| latent h | — | — | 1.006 | 438 | 616 |

**Timing on the H200.** 56.1 s to the start of warmup, including 22.8 s of sweep compilation; 111.3 s of warmup at 55.6 ms per sweep; 286.2 s of sampling at 71.6 ms per draw, including checkpoints. The native fit duration was 458.6 s. Against the computed-chart mGRAD control, which had profiling on, numerical time fell 34.4% from 693.1 s.

**Against NUTS.** Worst bulk ESS per second is 0.891 against 0.583, or 1.53×, and worst tail ESS per second 1.235 against 0.726, or 1.70×. The posterior means agree within 0.5 combined MCSEs. [Comparison](../../scratchpad/sv-nuts-20261030/results/comparison-optimized.json).

**Cost.** $0.792: the fit $0.791 and a Modal CPU analysis $0.001.

## SV with a cubic drift: Particle-mGRAD with 80 chains — 2026-10-31

**R̂ passes at the 4,000-draw ceiling on one H200, but bulk ESS for decay and vol-of-vol and tail ESS for vol-of-vol fall short of the 8,000 that 80 chains require.** It is the grassmoth arm of the [README comparison](../../README.md#stochastic-volatility). [Findings](../../scratchpad/sv-cubic-p128-20261031/findings.md).

- **Model:** [cases/suite/SV-cubic](../cases/suite/SV-cubic) is the SV case with its cubic drift coefficient raised from 0 to 0.03, so the log-variance drift is −θh − 0.03h³. The 3,000 readings come from one `simulate` of that truth with seed 721.
- **Source:** master `a7cb357b` plus the 80-chain experiment patch, in an isolated checkout: sweep memory planning for 80 chains, a binary-search resampling kernel, parallel CPU convergence diagnostics, GPU telemetry, a 192 GiB host-memory request and a serialization fix. [Patch](../../scratchpad/sv-cubic-20261030/mgrad-final-source.patch).
- **Run:** study `SV_CUBIC_P128_20261031`, call `call:54cf93588096debce6cd4460ac99ab6e09d5fb62912e10eca61dc3dcb0baecd9`, journal commit `e56793aa`.
- **Request:** [fit-SV-cubic-mgrad-h200.json](fit-SV-cubic-mgrad-h200.json). Against the optimized master request above, it runs 80 chains, retains no optional diagnostic metrics and allows 3,600 s.

| Quantity | Posterior mean [94% HDI] | Truth | R̂ | Bulk ESS | Tail ESS |
| --- | --- | ---: | ---: | ---: | ---: |
| level μ | −0.180 [−0.322, −0.037] | −0.2 | 1.008 | 10,662 | 16,042 |
| decay θ | 0.0150 [0.0000, 0.0301] | 0.02 | 1.011 | 6,883 | 8,669 |
| vol-of-vol σ | 0.194 [0.150, 0.237] | 0.2 | 1.016 | 4,800 | 7,040 |
| latent h | — | — | 1.003 | 26,573 | 26,617 |

**Timing on the H200.** 42.7 s to the start of warmup, including 20.2 s of sweep compilation; 136.5 s of warmup at 68 ms per sweep; and 275.9 s of sampling sweeps at 68.9 ms each. The README counts warmup and sampling, 412.0 s, and lists compilation separately. The convergence checks added 419.9 s, taking 41, 84, 126 and 169 s at the four checkpoints because each recomputes every diagnostic from scratch on the CPU. The native fit took 880.4 s, and reporting and saving another 241.5 s.

**Cost.** $2.015 billed: the fit $2.003 (H200 $1.409, memory $0.477, CPU $0.117) and a Modal CPU analysis $0.012. The README's $0.59 prices its 412.0 s of warmup and sampling at Modal's rates for the H200, 8 CPU cores and 32 GiB.

## SV with a cubic drift: NUTS with 32 chains on an M1 Max — 2026-10-31

**Passes all 12 checks at the first checkpoint, 1,000 draws per chain.** It is the NUTS arm of the [README comparison](../../README.md#stochastic-volatility), against the mGRAD fit above. [Findings](../../scratchpad/sv-nuts-local32-20261031/findings.md).

- **Model and data:** identical to the mGRAD fit's.
- **Source:** the mGRAD fit's source plus a patch that gives each local CPU device a group of four vectorized chains, with no communication between devices inside a sweep block. From a shared adapted state, groups of one, two and four chains per device reproduce the single-device draws, random streams, leapfrog counts and divergences. [Patch](../../scratchpad/sv-nuts-local32-20261031/chain-mesh.patch).
- **Run:** study `SV_NUTS_LOCAL32_20261031`, call `call:151c784c52e9bdddbcb86a473204b0b6f2f696fe526a1d9a782f91e98bc043a6`, journal commit `85226638`.
- **Request:** [fit-SV-cubic-nuts-cpu.json](fit-SV-cubic-nuts-cpu.json). It runs non-centred `nuts_da` with NumPyro's defaults (target acceptance 0.8, tree depth at most 10), 32 chains and `hardware: local`.
- **Hardware:** Apple M1 Max, 8 performance and 2 efficiency cores, 64 GiB. The worker ran 8 XLA CPU devices at a median 780–790% CPU, while other work kept load averages at 9–12.

| Quantity | Posterior mean [94% HDI] | Truth | R̂ | Bulk ESS | Tail ESS |
| --- | --- | ---: | ---: | ---: | ---: |
| level μ | −0.180 [−0.321, −0.037] | −0.2 | 1.001 | 19,295 | 21,214 |
| decay θ | 0.0152 [0.0000, 0.0305] | 0.02 | 1.003 | 9,785 | 13,050 |
| vol-of-vol σ | 0.195 [0.151, 0.238] | 0.2 | 1.002 | 9,395 | 15,131 |
| latent h | — | — | 1.002 | 33,680 | 22,536 |

Every retained transition took 31 leapfrog steps. Mean acceptance was 0.87, with no divergence in retention and 682 in warmup, 17–25 per chain.

**Timing.** 3.0 s to the start of warmup, 975.7 s of warmup, and 179.4 s of sampling sweeps at 178.2 ms each, against 156.0 ms for the 4-chain fit on one device. The README counts warmup and sampling, 1,155.1 s. The convergence check added 5.7 s, so the native fit took 1,164.3 s; reporting and saving took another 26.5 s.

**Against mGRAD.** Over warmup and sampling, worst bulk ESS per second is 8.1 against mGRAD's 11.7, and worst tail ESS per second 11.3 against 17.1. On sampling sweeps alone, NUTS leads 52.4 against 17.4. The posterior means agree with mGRAD's within 1.2 combined MCSEs for level and decay and 2.0 for vol-of-vol, and with the other cubic-drift NUTS fits within 0.9. [Comparison](../../scratchpad/sv-nuts-local32-20261031/comparison.json).

**Cost.** The fit has no Modal charge. The Modal CPU analysis cost $0.006.

## SV with a cubic drift: auxiliary Kalman and centred NUTS on the H200 — 2026-10-31

**Three more arms of the [README comparison](../../README.md#stochastic-volatility). None passes the check.** The model and data are the mGRAD fit's above. All three use seed 19, 2,000 warmup sweeps and float32 on one H200, with profiling, optional diagnostics and LOO off.

| Fit | Study and call | Request | Source |
| --- | --- | --- | --- |
| Auxiliary Kalman, whole path, 80 × 4,000 | `SV_AUX_KALMAN80_20261031`, `call:35b457af242a9fe737b00c78dfb945c506f4cfc18ddc618ffdf65cbd6e1b4f1e` | [fit-SV-cubic-aux-kalman-h200.json](fit-SV-cubic-aux-kalman-h200.json) | `ca79f464` plus the uncommitted auxiliary Kalman candidate and a fix for a per-time trace saved with optional diagnostics off. [Patch](../../scratchpad/sv-aux-kalman80-20261031/candidate.patch) |
| Auxiliary Kalman, 300-day blocks, 80 × 10,000 | `SV_AUX_KALMAN80_20261031`, `call:93439ebbd2b948329e7c16410590e1c4b70c02ebb19401e61b82382656db6a50` | [fit-SV-cubic-blocked-kalman-h200.json](fit-SV-cubic-blocked-kalman-h200.json) | `8efe81d8` plus the candidate with blocks. [Patch](../../scratchpad/blocked-kalman-20261031/candidate.patch) |
| NUTS, centred, 80 × 4,000 | `SV_NUTS80_CENTRED_20261031`, `call:7eaa16ced5a9a4844864a001a6e6cba02cba68e10c9d6944a7164e07282c876e` | [fit-SV-cubic-nuts-centred-h200.json](fit-SV-cubic-nuts-centred-h200.json) | `a7cb357b` plus the 80-chain experiment patch of the mGRAD fit above |

| R̂ / bulk ESS / tail ESS | Kalman, whole path | Kalman, blocked | NUTS, centred |
| --- | --- | --- | --- |
| level μ | 1.019 / 4,523 / 8,723 | 1.007 / 13,975 / 24,861 | 1.003 / 26,543 / 51,854 |
| decay θ | 1.073 / 761 / 1,163 | 1.023 / 3,134 / 6,850 | 1.014 / 5,814 / 11,344 |
| vol-of-vol σ | 1.160 / 368 / 644 | 1.038 / 1,492 / 3,336 | 1.041 / 1,869 / 3,194 |
| latent h | 1.049 / 1,183 / 2,184 | 1.011 / 6,993 / 13,911 | 1.003 / 26,952 / 57,703 |

- **Whole-path Kalman:**
  - Timing: 16.8 s of warmup and 41.3 s of sampling sweeps at 10.3 ms each.
  - Movement: the physical path changed in 54.7% of sweeps, a figure that also counts moves made by parameter updates. Parameter acceptance was 18.6%. The step tuning shrinks every time step's scale together, because all 3,000 days share one accept/reject.
  - Means: all three agree with the mGRAD and NUTS fits within one combined MCSE.
  - Earlier attempts: three failed after sampling. `call:192e05c7` ran out of its 32 GiB of host memory, `call:d411a03e` failed to save a 5.76 GB per-time trace, and `call:25ec0fc6` saved remotely but was never published when its local coordinator was interrupted. The successful fit used 192 GiB. [Findings](../../scratchpad/sv-aux-kalman80-20261031/findings.md).
- **Blocked Kalman:**
  - Timing: 17.4 s of warmup and 97.2 s of sampling at 9.7 ms per sweep.
  - Block acceptance: 50.3%.
  - Blocks: each sweep first draws the parameter label from its exact conditional given the path, then accepts or rejects alternating 300-day blocks in parallel under that label.
  - Block length: chosen from 4-chain CPU probes of 30, 100 and 300 days. Those probes had only 300 warmup sweeps, and the 30- and 100-day blocks were still below the 50% acceptance target. [Findings](../../scratchpad/blocked-kalman-20261031/findings.md).
- **Centred NUTS:**
  - Timing: 932.3 s of warmup and about 884 s of sampling sweeps at 221 ms each.
  - Mean acceptance: 0.84, with no divergence in retention and 1,897 across the chains in warmup.
  - Means: agree with the other fits within 0.9 combined MCSEs.
  - Cost: Modal billed $4.09 including analysis. [Findings](../../scratchpad/sv-nuts80-centred-20261031/findings.md).
- **Two other GPU NUTS attempts left no result:**
  - A non-centred run (`call:d5f21bd8` in `SV_NUTS_H200_20261031`) reached 101 of 2,000 warmup sweeps in 2,655 s. That averages 26 s per sweep; its last 25-sweep block took 23 s per sweep. It was lost when the Mac running the client slept. Its chains advanced independently, with parallel U-turn checks. [Findings](../../scratchpad/sv-nuts-cpu80-20261031/findings.md).
  - An earlier centred run (`call:e7d560dd` in `SV_CUBIC_NUTS80_20261031`) was stopped at warmup sweep 126.
- **One more auxiliary Kalman fit was lost:** `call:8ab3cb8b`, with 40 chains × 40,000 draws, sampled in 234 s and computed its diagnostics in 93 s, then hit a 480 s time limit while saving. [Findings](../../scratchpad/sv-aux-kalman80-long-20261031/findings.md).

## Lorenz-96 — 2026-10-31

**Behind the [README comparison](../../README.md#lorenz-96). No fit converges.**

- **Model and data:** [cases/suite/L96](../cases/suite/L96) is the stochastic Lorenz-96 system, with 40 cyclic states, forcing 8 and a time spacing of 0.01 over 300 time points. Alternating states are observed with noise scale 1, giving 6,000 readings. The truth was simulated with seed 721 by the Heun simulator.
- **Target:** fits use the exact Euler–Maruyama model. Forcing, process and observation scales are known, so this is smoothing of 12,000 latent coordinates per chain.
- **Settings:** seed 19, float32 and 2,000 warmup sweeps throughout.
- **Original attempt:** a 3,000-step model with three shared unknown parameters was rejected by the compiler before sampling. [Findings](../../scratchpad/lorenz96-20261031/findings.md).

| Fit | Study and call | Request |
| --- | --- | --- |
| mGRAD, 256 particles, 2 × 4,000, H200 | `LORENZ96_FIXED300_20261031`, `call:6d009a37f30e05c6c95bef4148b72831bc6cf40b8558ce2c383c84463173d069` | [fit-L96-mgrad-h200.json](fit-L96-mgrad-h200.json) |
| Auxiliary Kalman, whole path, 2 × 4,000, H200 | `AUX_KALMAN_MASTER_20261031`, `call:274fdac6d9fad58c12b6bbb996c613f5720e1aaa01255a136aadf41ca43109bc` | [fit-L96-aux-kalman-h200.json](fit-L96-aux-kalman-h200.json) |
| Auxiliary Kalman, 25-step blocks, 2 × 10,000, H200 | `AUX_KALMAN_MASTER_20261031`, `call:19cedc78860a0353cc9bb5a6a3ebfe59a5580f18966038b001f169da0b95ffe0` | [fit-L96-blocked-kalman-h200.json](fit-L96-blocked-kalman-h200.json) |
| NUTS, non-centred, 32 chains on the M1 Max, 51 warmup and 50 retained sweeps requested | `LORENZ96_FIXED300_20261031`, stopped in warmup | [fit-L96-nuts-cpu-probe.json](fit-L96-nuts-cpu-probe.json) |

| | mGRAD | Kalman, whole path | Kalman, blocked |
| --- | ---: | ---: | ---: |
| Warmup | 534.0 s | 17.0 s | 15.8 s |
| Sampling sweeps | 1,064.4 s | 33.9 s | 76.7 s |
| One sweep of both chains | 266.1 ms | 8.5 ms | 7.7 ms |
| Worst bulk / tail ESS | 2.43 / 4.77 | 63.0 / 99.3 | 31.1 / 49.4 |
| Worst R̂ | 2.507 | 1.059 | 1.076 |
| Acceptance | 50.7% of time steps change | 49% of paths | 51.1% of blocks |

- **mGRAD:**
  - Source: `a7cb357b` plus a Lorenz patch. It keeps unobserved coupled states and handles a model with no free parameters in the charts. It also bounds backward-pair scoring to one time point, which removes a 467 GiB allocation at 80 chains. The fit runs on CUDA 12 with a Jacobi eigensolver. [Patch](../../scratchpad/lorenz96-20261031/source.patch).
  - Chains: 80-chain fits failed with CUDA errors, so the user allowed fewer chains.
  - Timing: sweep compilation took 155 s and the four convergence checks 77 s, both excluded above.
  - Mixing: every state's worst bulk ESS lies between 2.4 and 3.0, with R̂ between 1.83 and 2.51.
  - Cost: $6.29 for all Lorenz attempts, primitive checks and analyses.
- **Kalman:**
  - Source: the auxiliary Kalman candidate, on `ca79f464` for the whole-path fit and `8efe81d8` for the blocked one, with the Lorenz support and H200 workarounds above and XLA autotuning off.
  - Matched profile: on identical source and settings, a ten-sweep profile took 279.9 ms per compiled sweep for mGRAD against 13.5 ms for Kalman. Ten real sweeps took 2.82 s against 0.235 s (calls `call:3f6403c197a3b7f1604d9bd06fc833675d20dc47777fa25a4a6529f9f63c5dc2` and `call:eef0ebf4a92423214de17f2337cff578a52a3fc381c3b815b4b101b3f186816c`). [Findings](../../scratchpad/aux-kalman-profile-20261031/master/findings.md).
  - Block length: 25 steps, from CPU probes of 25, 75 and 300. Shorter blocks moved the path more per sweep but mixed worse over the whole path. [Findings](../../scratchpad/blocked-kalman-20261031/findings.md).
- **NUTS:** the probe reported 26 of its 51 warmup sweeps after 48 min 55 s and was stopped at 78 minutes, with no retained draw.
