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
