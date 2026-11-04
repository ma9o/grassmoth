# Suite records

Run records for the [suite](../README.md#cases) and for SV-cubic and Lorenz-96, the root README's comparison cases, oldest first. The [README](../README.md#decisions) keeps the current decisions. Records that later runs replaced, among them the first scorecard and the SV runs before the cubic drift, are in the tree at [`f6eaba3d`](https://github.com/ma9o/grassmoth/blob/f6eaba3df8398ae4395969662484e5d3a15af533/benchmarks/suite/notes.md).

## Fixtures — 2026-10-24

The cases were built through the public actions in study `SUITE_20261024`. The [protocol](fixture-protocol.json) pins the seeds and validity rules, registered before any study action, and the [manifest](fixture-manifest.json) holds each case's revisions and calls. All declarations are in [cases/suite](../cases/suite).

- **NL and MODE** were redesigned before the first scorecard. NL's cubic candidates produced non-finite simulated paths, and MODE's first candidate didn't establish separated posterior mass. NL now uses a bounded-slope drift and MODE an exact-EM conditioned-history fixture; the manifest keeps the rejected NL candidates' calls. [Redesign record](../../scratchpad/nl-mode-redesign-20261024/findings.md).
- **MODE** holds its unobserved input effect fixed at 0.7, with no learned nuisance parameter. Its corrected prepared data retains one null y row at an existing anchor, with no added observation.
- **HD** uses declared potential indicators with every hidden reading masked; deleting those indicators caused native projection refusal.
- **PRIOR** uses exact known-input half-step timestamps while retaining the 675s reading clock.
- **C8** places the study outcome at the final deterministic stage so authoring does not prune it.

These are declaration and data changes, not framework workarounds in inference.

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

Requests: [C0](fit-C0-twisted-k1-sc2.json), [NL](fit-NL-twisted-k1-sc2-retry1.json), [MODE](fit-MODE-twisted-k1-sc2.json), [HD](fit-HD-twisted-k1-sc2.json), [OBS](fit-OBS-twisted-k1-sc2-retry1.json), [PRIOR](fit-PRIOR-twisted-k1-sc2-retry1.json), [PARAM](fit-PARAM-twisted-k1-sc2.json) and the [HD timing probe](timing-HD-twisted-k1-sc2-retry1.json). The `retry1` bodies resubmit at `repeat: 1` after failed pre-allocation attempts.

**These histories predate the `simulate` fix** in `9986d2fe`, which made readings sample the diffusion instead of a smooth interpolant between solver steps. So do TAIL's and the PARAM tournament's below. Only PARAM's history has been [regenerated](#param-on-regenerated-data--2026-10-25) since.

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

**The `centred_nuts` companion passes PARAM's native gate at both seeds, and the policy alone fails both.** Master removed the companion moves on 2026-10-29 (`2d003603`).

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

## SV with a cubic drift: the case

**Model.** [cases/suite/SV-cubic](../cases/suite/SV-cubic) is the Gaussian stochastic volatility model of Kim, Shephard & Chib (1998) and the Stan User's Guide, with a cubic drift on the log-variance:
- daily percent returns r_t ~ Normal(0, exp((μ + h_t)/2)), where the log-variance h has drift −θh − 0.03h³, diffusion scale σ and h₀ ~ N(0, 1);
- readings are squared returns, r² ~ Gamma(½, rate ½·exp(−(μ + h))). |r| depends on no parameter, so the posterior is the Gaussian SV posterior;
- truth μ = −0.2, θ = 0.02 (persistence 0.98), σ = 0.2; priors μ ~ Normal(0, 2), θ ~ HalfNormal(0.1), σ ~ HalfNormal(0.5);
- 3,000 daily readings from one `simulate` with seed 721, start 2020-01-01 and horizon 2999d.

Fits use the exact Euler–Maruyama model on the 1-day clock.

**Before the cubic drift.** The case replaced the same model with a zero cubic coefficient, first fitted on 2026-10-29. There the policy's twisted Particle-aGRAD failed at the 4,000-draw ceiling, with worst R̂ 3.69 (`call:d580a9e7`): the readings pin only μ + h_t, and the chains slid along that ridge. Particle-mGRAD with translation and scale charts, and NUTS, later passed it. NUTS runs on the CPU because non-centred NUTS rebuilds the path in a sequential scan at every gradient: fixed 8-step HMC sweeps on that case took 521 ms on an H200 against 41 ms on the M1 Max, or 65 and 5.1 ms per leapfrog step.

## SV with a cubic drift: Particle-mGRAD with 80 chains — 2026-10-31

**R̂ passes at the 4,000-draw ceiling on one H200, but bulk ESS for decay and vol-of-vol and tail ESS for vol-of-vol fall short of the 8,000 that 80 chains require.** It is the grassmoth arm of the [README comparison](../../docs/README.md#stochastic-volatility-with-cubic-drift). [Findings](../../scratchpad/sv-cubic-p128-20261031/findings.md).

- **Model and data:** [the case above](#sv-with-a-cubic-drift-the-case).
- **Source:** master `a7cb357b` plus the 80-chain experiment patch, in an isolated checkout: sweep memory planning for 80 chains, a binary-search resampling kernel, parallel CPU convergence diagnostics, GPU telemetry, a 192 GiB host-memory request and a serialization fix. [Patch](../../scratchpad/sv-cubic-20261030/mgrad-final-source.patch).
- **Run:** study `SV_CUBIC_P128_20261031`, call `call:54cf93588096debce6cd4460ac99ab6e09d5fb62912e10eca61dc3dcb0baecd9`, journal commit `e56793aa`.
- **Request:** [fit-SV-cubic-mgrad-h200.json](fit-SV-cubic-mgrad-h200.json). It pins Particle-mGRAD with 128 particles, κ = 1, scan unroll 8 and the latent step adapted to 0.5; two parameter labels with random-walk proposals at a 0.2 acceptance target; compiler reparameterization and the scale chart; initialization scale 0.1; 80 chains, 2,000 warmup sweeps and 4,000 draws, with early stopping at 1,000-draw checkpoints; no optional diagnostic metrics; profiling off; one H200; a 3,600 s limit.

| Quantity | Posterior mean [94% HDI] | Truth | R̂ | Bulk ESS | Tail ESS |
| --- | --- | ---: | ---: | ---: | ---: |
| level μ | −0.180 [−0.322, −0.037] | −0.2 | 1.008 | 10,662 | 16,042 |
| decay θ | 0.0150 [0.0000, 0.0301] | 0.02 | 1.011 | 6,883 | 8,669 |
| vol-of-vol σ | 0.194 [0.150, 0.237] | 0.2 | 1.016 | 4,800 | 7,040 |
| latent h | — | — | 1.003 | 26,573 | 26,617 |

**Timing on the H200.** 42.7 s to the start of warmup, including 20.2 s of sweep compilation; 136.5 s of warmup at 68 ms per sweep; and 275.9 s of sampling sweeps at 68.9 ms each. Warmup and sampling took 412.0 s. The convergence checks added 419.9 s, taking 41, 84, 126 and 169 s at the four checkpoints because each recomputes every diagnostic from scratch on the CPU. The native fit took 880.4 s, and reporting and saving another 241.5 s.

**Cost.** $2.015 billed: the fit $2.003 (H200 $1.409, memory $0.477, CPU $0.117) and a Modal CPU analysis $0.012. At Modal's rates for the H200, 8 CPU cores and 32 GiB, its 412.0 s of warmup and sampling alone cost $0.59.

## SV with a cubic drift: NUTS with 32 chains on an M1 Max — 2026-10-31

**Passes all 12 checks at the first checkpoint, 1,000 draws per chain.** It is the NUTS arm of the [README comparison](../../docs/README.md#stochastic-volatility-with-cubic-drift), against the mGRAD fit above. [Findings](../../scratchpad/sv-nuts-local32-20261031/findings.md).

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

**Timing.** 3.0 s to the start of warmup, 975.7 s of warmup, and 179.4 s of sampling sweeps at 178.2 ms each, against 156.0 ms for the 4-chain fit on one device. Warmup and sampling took 1,155.1 s. The convergence check added 5.7 s, so the native fit took 1,164.3 s; reporting and saving took another 26.5 s.

**Against mGRAD.** Over warmup and sampling, worst bulk ESS per second is 8.1 against mGRAD's 11.7, and worst tail ESS per second 11.3 against 17.1. On sampling sweeps alone, NUTS leads 52.4 against 17.4. The posterior means agree with mGRAD's within 1.2 combined MCSEs for level and decay and 2.0 for vol-of-vol, and with the other cubic-drift NUTS fits within 0.9. [Comparison](../../scratchpad/sv-nuts-local32-20261031/comparison.json).

**Cost.** The fit has no Modal charge. The Modal CPU analysis cost $0.006.

## SV with a cubic drift: auxiliary Kalman and centred NUTS on the H200 — 2026-10-31

**More arms of the [README comparison](../../docs/README.md#stochastic-volatility-with-cubic-drift). None passes the check.** The model and data are the mGRAD fit's above. All four fits use seed 19, 2,000 warmup sweeps and float32 on one H200, with profiling, optional diagnostics and LOO off. The README's blocked column is the rerun on master.

| Fit | Study and call | Request | Source |
| --- | --- | --- | --- |
| Auxiliary Kalman, whole path, 80 × 4,000 | `SV_AUX_KALMAN80_20261031`, `call:35b457af242a9fe737b00c78dfb945c506f4cfc18ddc618ffdf65cbd6e1b4f1e` | [fit-SV-cubic-aux-kalman-h200.json](fit-SV-cubic-aux-kalman-h200.json) | `ca79f464` plus the uncommitted auxiliary Kalman candidate and a fix for a per-time trace saved with optional diagnostics off, on CUDA 12 with XLA autotuning off. [Patch](../../scratchpad/sv-aux-kalman80-20261031/candidate.patch) |
| Auxiliary Kalman, 300-day blocks, 80 × 10,000 | `SV_AUX_KALMAN80_20261031`, `call:93439ebbd2b948329e7c16410590e1c4b70c02ebb19401e61b82382656db6a50` | [fit-SV-cubic-blocked-kalman-h200.json](fit-SV-cubic-blocked-kalman-h200.json) | `8efe81d8` plus the candidate with blocks, on CUDA 12 with XLA autotuning off. [Patch](../../scratchpad/blocked-kalman-20261031/candidate-run.patch) |
| Auxiliary Kalman, 300-day blocks, 80 × 10,000, on master | A copy of `SV_AUX_KALMAN80_20261031` in its own checkout, `call:531125b7b6d074cab2995e9f33a0db0a0d99722d9f983909ef9ca099dd999359` | [fit-SV-cubic-blocked-kalman-master-h200.json](fit-SV-cubic-blocked-kalman-master-h200.json) | Master `6aebcf79` plus 192 GiB of fit host memory, on master's CUDA 13 with XLA autotuning on. [Patch](../../scratchpad/blocked-rerun-20261031/host-memory.patch) |
| NUTS, centred, 80 × 4,000 | `SV_NUTS80_CENTRED_20261031`, `call:7eaa16ced5a9a4844864a001a6e6cba02cba68e10c9d6944a7164e07282c876e` | [fit-SV-cubic-nuts-centred-h200.json](fit-SV-cubic-nuts-centred-h200.json) | `a7cb357b` plus the 80-chain experiment patch of the mGRAD fit above |

| R̂ / bulk ESS / tail ESS | Kalman, whole path | Kalman, blocked | Kalman, blocked, on master | NUTS, centred |
| --- | --- | --- | --- | --- |
| level μ | 1.019 / 4,523 / 8,723 | 1.007 / 13,975 / 24,861 | 1.006 / 12,295 / 22,730 | 1.003 / 26,543 / 51,854 |
| decay θ | 1.073 / 761 / 1,163 | 1.023 / 3,134 / 6,850 | 1.021 / 2,861 / 6,400 | 1.014 / 5,814 / 11,344 |
| vol-of-vol σ | 1.160 / 368 / 644 | 1.038 / 1,492 / 3,336 | 1.037 / 2,020 / 3,364 | 1.041 / 1,869 / 3,194 |
| latent h | 1.049 / 1,183 / 2,184 | 1.011 / 6,993 / 13,911 | 1.011 / 7,211 / 13,845 | 1.003 / 26,952 / 57,703 |

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
- **Blocked Kalman on master:**
  - Why: the fit above ran before the speedups that landed with the sampler. This rerun replaces it in the README.
  - Timing: 16.4 s of compilation from an empty cache, then 12.2 s of warmup and 76.4 s of sampling at 7.6 ms per sweep. Warmup is 30% shorter and sampling 21% shorter.
  - Same sampler: the speedups factor each block's Gaussian law once per sweep, propose only the blocks being updated, and cache density terms. The proposals and random keys are unchanged, so mixing changes only through rounding.
  - ESS: vol-of-vol bulk ESS rose from 1,492 to 2,020 while decay and level fell 9–12%, which is run-to-run variation.
  - Block acceptance: 48.7%.
  - Runtime: master's CUDA 13 with autotuning ran without the CUDA errors that the earlier Kalman fits worked around.
  - Cost: the fit app ran 310 s, at most $0.57, and the analysis about $0.01. [Findings](../../scratchpad/blocked-rerun-20261031/findings.md).
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

**Behind the [README comparison](../../docs/README.md#lorenz-96). No fit converges.**

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
  - Speedups: the blocked fit predates the blocked-sampler speedups that landed with the sampler in `6aebcf79`. A separate H200 test of the speedups cut Lorenz-96 sweep time by 27% and lengthened compilation by 19%. [Results](../../scratchpad/blocked-speed-20261031/results.json).
  - Master since `05bc217d` keeps unobserved coupled states and builds charts for a model with no free parameters. It lacks the bounded pair scoring, CUDA 12 and autotuning off.
- **NUTS:** the probe reported 26 of its 51 warmup sweeps after 48 min 55 s and was stopped at 78 minutes, with no retained draw.

## TAIL with blocked auxiliary Kalman — 2026-11-04

**Both seeds pass the production convergence check, with no Student-t precision augmentation.** The fits use TAIL's schedule of 4 chains, 2,000 warmup sweeps and 4,000 draws, with 300-step blocks and an adapted step. This is a quick look, not a scorecard result: it ran on an L40S rather than an A10 and skipped the prior-bank gate. The truth-path gate passed.

| Seed | Worst R̂ | Worst bulk ESS | Worst tail ESS | Total | Sampling | Input effect (truth 0.7) | df (truth 5) |
|---|---:|---:|---:|---:|---:|---|---|
| 19 | 1.0095 | 799 | 1,240 | 24.8 s | 8.4 s | 0.73 [0.57, 0.89] | 5.58 [4.92, 6.20] |
| 29 | 1.0074 | 726 | 1,183 | 88.9 s | 9.1 s | 0.73 [0.55, 0.89] | 5.58 [4.95, 6.26] |

- **Slowest times are block edges, not outliers.** Of each seed's 20 slowest times by movement, 20 and 19 lie within one step of a multiple of 150, which covers 2% of times. Three of each 20 have a convex reading, against 14% of all times. Movement RMS at convex readings matches central ones: 0.0467 against 0.0463 at seed 19, 0.0475 against 0.0470 at seed 29. Acceptance is per block and the step adapts to it, so this doesn't show that augmentation would add nothing.
- **Data.** `simulate` with seed 721 on `35234163` gives latent paths that differ from the October history, because of `9986d2fe` (RMS difference 0.57–0.68, path RMS 0.54–0.82). The measurement noise matches to 3e-7, so the gate's numbers are unchanged. The [October fits](#tail--2026-10-24) predate that fix and the precision augmentation in `52807ada`, so they aren't a comparison. No current guided fit exists on this history.
- **Source.** Seed 19 ran on clean `35234163` (digest `18fa4d7f`) with the study service hosted on Modal CPU. Seed 29 ran from a local stack on the shared checkout (digest `a0bca1c8`), whose uncommitted edits change only the default adaptation scheme; the request pins `simple`. Seed 29's longer total is almost all setup and compilation.
- **Failed attempt.** Seed 29's first attempt, `call:2b93521b870e9331b3c5dfacccca135d8438fbd597a64019388fb92059f9aa60`, lost its runner to Modal preemption. It isn't a sampling result.

Study: `TAIL_KALMAN_20261104`, local and gitignored. Calls: seed 19 `call:81265585311ff9934d2c6ff16bb900a588ff310e6043ea178a17f6badd6042ed`, seed 29 `call:0afa481a1f1e9ee9223de0e96e5879ad43b8df1048e1a67a2ab4cbad7c4af68c`. Modal cost $0.52, including restarts. No decision is proposed; the next comparison is a guided fit on the same history.
