# Stage 00 notes

## 2026-10-22 A10 run at the full schedule

The [2026-10-22 A10 run](../../scratchpad/benchmarks/stage00-a10-20261022-03/results.json)
used the declared 128 particles, four chains, 2,000 warmup sweeps and 4,000 retained
draws per chain. [C0's complete report](../../scratchpad/benchmarks/stage00-a10-20261022-03/C0/summary.json)
fails production convergence. C1–C3 are blocked and were not launched.

This completed run used the retired direct-library runner. It has no HTTP study
workspace, action call ID or study commit ID. It stands under the
[workflow decision](../README.md#workflow--2026-10-22); migration did not launch
another fit.

| Quantity | R̂ | Bulk ESS | Tail ESS |
| --- | ---: | ---: | ---: |
| Input effect | 1.0016 | 1,705.7 | 2,556.2 |
| Latent x, worst coordinate | 1.0152 | 338.4 | 371.5 |
| Latent y, worst coordinate | 1.0320 | 229.7 | 194.9 |

All reported R̂ values pass the production limit of 1.05. The latent bulk and tail
ESS findings fail the required 400. The input-effect parameter passes; its saved
mean is 0.599 with 94% HDI [0.437, 0.765], against truth 0.7. Truth recovery does
not override the failed latent findings.

The public fit took 415.3 seconds, including 195.5 seconds of sweep compilation,
57.2 seconds of warmup and 113.8 seconds of retained sampling. Total dispatch time
was 621.5 seconds, including image setup, startup and result transport. The
[provider billing snapshot](../../scratchpad/benchmarks/stage00-a10-20261022-03/billing-final-reconciliation.json)
reports $0.2173 for this execution, plus $0.0162 for two earlier setup failures:
**$0.2335 total** including two CPU metadata audits, within the $2 per-fit cap.
All five Apps are stopped. The
earlier attempts failed before inference because of dependency-group selection
and source mounting; both runner faults were corrected.

Decision: keep stage 01 blocked and diagnose the remaining latent-chain dependence
in C0. The schedule stays fixed. These results alone do not select a preconditioner
or another sampler capability. C0's single declared NUTS diagnostic remains an
available next comparison.

### Execution provenance

- Modal App: `ap-bWWE3F5fV9drQtW5FY2Hdj`.
- Modal function call: `fc-01M5HYG0DGWQJ6N5GZFQHGH2P1`.
- Function: `fu-SUniy1cY14365ghBMREcf2`; image: `im-3cOOIoInSd6SuTUpk4KTd8`.
- Device: `gpu:NVIDIA A10G:0`; arithmetic: float32; sampler seed: 19.
- Executed source SHA-256: `7e7444d00ef7d27d13976948012ddb15a682c05cf0c843e43d2baf4268867a7b`.

The [source audit](../../scratchpad/benchmarks/stage00-a10-20261022-03/executed-source.json)
matches the saved runtime digest. The exact library source, runner source and
lockfile recovered from the immutable execution image and mounts are retained
under `scratchpad/benchmarks/stage00-a10-20261022-03/executed-source/`.
The full public fit output is retained as `C0/fit.msgpack` beside the summary,
resolved settings, observations and phase timings. Two later CPU-only metadata
audits did no inference and cost $0.00003615 combined. The retired runner,
declarations, fixture inputs and tests are archived
under `retired-library-runner/` in that run directory, with a SHA-256 manifest.

The latent frozen fraction was 0.749, with latent update rate 0.503. Redrawing one
of two path coordinates per sweep at roughly half acceptance already implies
roughly three quarters of coordinates remain unchanged. That fraction alone does
not diagnose a broken path update; the failed ESS findings establish the problem.

## 2026-10-22 run below the schedule

The 2026-10-22 run
([outputs](../../scratchpad/benchmarks/stage00-library-20261022/results.json))
used 16 latent particles and 1,000 warmup plus 1,000 retained draws per chain,
so it doesn't count. It showed C0's paths not mixing: worst-coordinate R̂ 3.60
for x and 3.92 for y, bulk effective sample size about 4, and an input effect of
0.41 [0.25, 0.57] against a truth of 0.7. C1–C3 also ran, before the runner
stopped at the first failure, and failed.

## HTTP request files

The numbered JSON files describe a future run through the
[study API skill](../../.agents/skills/grassmoth-study-api/SKILL.md).
They are request templates: replace `${..._REVISION}` strings with the successful
prerequisite revisions before submitting them. For questions, models and prepared
data, find the required `artifact_id` in each call's timeline
`record.attempt.outcome.effects.produced` and use its `revision`. A simulation
produces no panel artifact: select its successful response's journal `commit_id`
as the data revision. Journal commits cannot serve as question or model artifact selectors. These are
not IDs for the completed legacy run. Save concrete bodies and responses under
the new run's `scratchpad/` directory; record its workspace and call IDs here.

| Files | Prerequisite revisions to pin |
| --- | --- |
| `00-edit_question.json` | None; start one study |
| `01-edit_model-truth.json` | `QUESTION_REVISION` from 00 |
| `02-simulate.json` | `TRUTH_MODEL_REVISION` from 01 |
| `03-edit_model-C0.json`, `04-fit-C0-A0.json` | Truth parent, then `C0_MODEL_REVISION`; `SIMULATION_REVISION` from 02 |
| `05-edit_model-C1.json`, `06-fit-C1-A0.json` | C0 parent, then `C1_MODEL_REVISION`; same simulation |
| `07-edit_model-C2.json`, `08-prepare_data-C2.json`, `09-fit-C2-A0.json` | C0 parent, then `C2_MODEL_REVISION` and `C2_DATA_REVISION` |
| `10-edit_model-C3.json`, `11-prepare_data-C3.json`, `12-fit-C3-A0.json` | C1 parent, then `C3_MODEL_REVISION` and `C3_DATA_REVISION` |

Case model parents are explicit. C2 and C3 create separate revisions without
changing the fixed and learned scientific definitions. Simulation selects the
truth revision, seed 721 and one 16-day history; dense fits select its explicit
journal commit and replicate 0. A simulation must not be selected with data `latest`.
The complete A0 algorithm is pinned in each fit body, with the
[canonical schedule](../README.md#compute-schedule). The HTTP settings contract
does not expose `retain_latent_paths`; the
[server's fit settings owner](../../apps/grassmoth-server/src/grassmoth_server/actions/fit/execution.py)
sets it to true so production checks include the latent paths.

For C2/C3, export replicate 0's recorded readings from the saved simulation to
one CSV, retaining their actual anchors and support bounds. Keep every 32nd
x/y reading plus each channel's endpoints; retain every u reading. Use columns
`timestamp`, `x`, `x_anchor`, `x_support_start`, `x_support_end`, and the same
four channel columns for y and u. `timestamp` indexes the union of retained
anchors. Leave a channel's anchor and other fields empty where it has no retained
reading. The
[recorded importer](../../apps/grassmoth-server/src/grassmoth_server/actions/prepare_data/materialization.py)
selects rows by each channel's non-null anchor; a non-null anchor with a null
value would instead record an unknown reading. Preserve retained values and
support exactly; do not synthesize or re-bucket observations.

Upload that CSV with `POST /api/upload` to the study's otherwise empty `input`
folder, then submit C2's `prepare_data` body. Reuse the identical uploaded bytes
for C3's preparation under its own model revision. Record the upload hash and
verify prepared readings against the retained simulation before fitting. No
upload or preparation was executed during this migration.

Stop at the first case where no arm passes production convergence. C1–C3's
files are declarations, not authorization to bypass C0. For timing repeats,
change `input.repeat`; identical resolved requests return their saved outcome.
For a sampler patch, use its own checkout, server and study, as required by the
[workflow](../README.md#running).

## 2026-10-22 C0 with 256 particles

The user explicitly requested this particle-count comparison. C0 alone uses 256
latent particles; the four chains, 2,000 warmup sweeps, 4,000 retained draws per
chain, seed 19 and complete A0 algorithm are unchanged. This does not change the
ladder's declared 128-particle schedule or authorize C1–C3.

The [submitted request](../../scratchpad/benchmarks/stage00-c0-256-a10-20261022/13-fit-C0-A0-256-particles.json) ran through HTTP in study
`BENCH00C0256_20261022`, served at `http://localhost:19400`. Its fit call is
`call:1ec2ffc7d6673a9b767aefd66ddadf1bbd575df3de24c58e7b259ebac1c2b0c9`.
C0's model artifact is `cfba07b77d22cf5e2f17180e253021f90a383e96`; the selected
simulation journal commit is `3004cd2463e50b9e6342da0ca7c7acb71b0cc8a6`, replicate 0.
The [data comparison](../../scratchpad/benchmarks/stage00-c0-256-a10-20261022/data-comparison.json)
confirms all 2,049 readings of each of x, y and u equal those used for the
128-particle fit, with zero numerical difference. Library source differs only in
public re-exports and whitespace; inference implementation is unchanged.

Status: completed; **production convergence failed**. The saved
[summary](../../scratchpad/benchmarks/stage00-c0-256-a10-20261022/summary.json)
contains the library's findings and diagnostics; none were recomputed by the
caller. All 16,000 draws were retained.

| Quantity | Maximum R̂, 256 | Bulk ESS, 128 → 256 | Tail ESS, 128 → 256 |
| --- | ---: | ---: | ---: |
| Input effect | 1.0019 | 1,705.7 → 1,766.1 | 2,556.2 → 2,536.5 |
| Latent x, minimum ESS across coordinates | 1.0095 | 338.4 → 506.8 | 371.5 → 342.6 |
| Latent y, minimum ESS across coordinates | 1.0192 | 229.7 → 366.0 | 194.9 → 355.7 |

Every R̂ passes 1.05. Three ESS findings fail the required 400: x tail, y bulk
and y tail. The input-effect mean is 0.600, with 94% HDI [0.441, 0.773] and
standard deviation 0.0884, against truth 0.7. Latent acceptance increased to
0.6185; parameter acceptance was 0.3516. These rates do not override the ESS findings.

The retained numerical duration is 678.3 seconds: sweep compilation took 64.0
seconds, warmup 200.1 seconds and sampling 398.1 seconds. The HTTP action took
935.8 seconds including image preparation, dispatch, result transport and
publication. The server allocated eight CPU cores and 32 GiB host memory,
compared with two cores and 16 GiB in the earlier library runner; compilation
cache conditions also differ. Total wall-time differences therefore do not
isolate the particle-count change.

The [billing reconciliation](../../scratchpad/benchmarks/stage00-c0-256-a10-20261022/billing-reconciliation.json)
reports **$0.3901**, below the $2 per-fit cap. Modal App
`ap-kNrm34tfhs4zFYrFmrFYSo` is stopped. The actual device was
`gpu:NVIDIA A10G:0`; the fit retained its 3,600-second limit and float32 arithmetic.
The saved runtime source digest is
`477eb5acacf5c657633725aca985867035427e78d7c00f8e77645620eb4057b5`.
Fit journal commit: `862cedbf60baa1de68fd30e1b4acdccd88a3c1a7`; produced model:
`85cb973f8af7bc4fbe50ba371464f274f0456fd9`.

Requests, complete responses, timeline, source snapshots and runtime logs are
retained in `scratchpad/benchmarks/stage00-c0-256-a10-20261022/`. An initial model
edit was rejected because it selected a journal commit as its parent; selecting
the question artifact revision corrected it before any fit. Resuming the local
stack also required its original Temporal queue names. Neither issue launched
an additional GPU fit.

Decision: increasing to 256 improves both paths' bulk ESS but does not clear C0's
production gate. C1–C3 and later stages remain blocked. No additional fits ran,
and the ladder's declared schedule remains unchanged.

## 2026-10-22 aMALA+ CPU compilation investigation

At the user's request, four CPU compile-only HTTP probes used the same C0 inputs,
four chains and 256 particles, with zero warmup and draws. Full profiles, requests,
responses, limitations and proposed owner changes are in the
[scratchpad notes](../../scratchpad/benchmarks/c0-amala-plus-cpu-compile-20261022/notes.md).

The cold `amala_plus` sweep compilation took 192.8 seconds and compiled four node
batch widths with XLA memory estimates of 107.89, 99.53, 50.09 and 25.37 GiB before
the separate availability check rejected it. The `drift: none` control compiled
in 18.2 seconds with a 6.28 GiB estimate. Pairwise seam-gradient calculations
dominate the aMALA+ graph. A warm executable cache still left 145.8 seconds of
sweep compilation, including repeated frontend preparation; a same-worker control
repeat also missed the staged graph cache with unchanged source.

These are instrumented CPU measurements of one-sweep blocks, not the production
25-sweep blocks or a convergence result. No sampler code changed, no GPU cost was
incurred, and C0's production gate and later-stage status are unchanged.

## 2026-10-22 C0 with parameter self-preconditioning

The user requested implementing parameter self-preconditioning before continuing
the ladder, then rerunning C0. This comparison matches the preceding 256-particle
run: four chains, 2,000 warmup sweeps, 4,000 retained draws per chain, seed 19 and
the same A10 settings. The [data comparison](../../scratchpad/self-preconditioning-20261022/data-comparison.json)
confirms all 2,049 readings in each channel are identical. The
[request](../../scratchpad/self-preconditioning-20261022/04-fit-C0.json) ran through
HTTP in its own checkout, server and study, `C0SELFPRECONDITION_20261022`.

The [implementation](../../packages/grassmoth/src/grassmoth/models/ssm/inference/warmup/parameter_adaptation.py)
learns a regularized dense parameter covariance from within-chain warmup moments.
The [runner](../../packages/grassmoth/src/grassmoth/models/ssm/inference/samplers/mpgibbs/runner.py)
owns its use between complete sweeps and freezing before retention. Initialization,
latent proposal policy and the exact target are unchanged. The public request and
response schemas are unchanged. The complete patch, experiment declaration and
submitted source manifest are retained under `scratchpad/self-preconditioning-20261022/`.

The saved [comparison](../../scratchpad/self-preconditioning-20261022/comparison.json)
uses the library's production diagnostics, without recomputing them in the caller:

| Quantity | Bulk ESS, fixed → adapted | Tail ESS, fixed → adapted | Maximum R̂, adapted |
| --- | ---: | ---: | ---: |
| Input effect | 1,766.1 → 1,901.4 | 2,536.5 → 2,753.5 | 1.0034 |
| Latent x, minimum across coordinates | 506.8 → 488.0 | 342.6 → 454.4 | 1.0123 |
| Latent y, minimum across coordinates | 366.0 → 267.1 | 355.7 → 205.7 | 1.0128 |

**C0 still fails production convergence:** y's bulk and tail ESS remain below 400.
All R̂ values pass 1.05. The input-effect mean is 0.599, with 94% HDI
[0.425, 0.760] and SD 0.0884. The checkpoint retains the learned parameter
Cholesky factor 0.08423, replacing the initial 0.1. All 16,000 draws were retained.
These values and the failed findings are in the [summary](../../scratchpad/self-preconditioning-20261022/summary.json).

This is one seed per policy, with mixed ESS changes. C0 has only one parameter;
the runner's scalar-step compensation preserves proposal variance when updating
its covariance. Floating-point perturbations can change discrete selections and
subsequent joint paths despite using the same random stream. This comparison
does not establish that self-preconditioning systematically improves x or harms y,
or test its ability to learn correlations between multiple parameters.

Actual sweep compilation took 50.6 seconds, warmup 200.4 seconds and retained
sampling 398.6 seconds; the preceding sampling time was 398.1 seconds. Numerical
duration was 656.5 seconds and action wall time 793.6 seconds. The first CLI log
stream stopped updating during compilation; a refreshed log recovered the complete
run. Its stale output did not indicate an inference stall. The
[billing reconciliation](../../scratchpad/self-preconditioning-20261022/billing-reconciliation.json)
reports **$0.3701**, below the $2 cap. Modal App `ap-8aVgUYWsn9ExlXd4EmqmgV` is stopped.

Fit call: `call:828e4d290d60ac3447df3d4f26deb08a96005d9de71670618a7f777117aca674`.
Fit journal commit: `785ee0e6b124f29bd704669150bb66b7de0e8abf`.
Runtime source digest: `bc3a1c115644f5ba2933cf84d73d47c4724c4ef105bf9e097eeba128a5933938`.
The [validation record](../../scratchpad/self-preconditioning-20261022/validation.json)
retains 38 focused passing checks, including analytic covariance recovery and
continuation through the checkpoint codec, plus successful root and workspace lint.

Decision: parameter self-preconditioning is implemented; this C0 run does not
clear the latent convergence gate or establish an efficiency benefit. No C1–C3,
C6 or additional GPU fits were launched. The ladder's 128-particle schedule remains
unchanged.

## 2026-10-22 C0 diagnostics retention failure and fix

The [all-diagnostics request](../../scratchpad/benchmarks/c0-all-diagnostics-20261022-01/00-fit-C0.json)
repeated the self-preconditioned C0 configuration with all diagnostic groups
enabled in study `C0SELFPRECONDITION_20261022`. Call
`call:4d4e4f8909b04f0cc371f57239084ffb865bdc98655cf85e917dade8d2b57cc6`
completed sampling but failed during report assembly. The
[runner log](../../scratchpad/benchmarks/c0-all-diagnostics-20261022-01/modal-latest.log)
records 16,000 retained draws followed by SIGKILL, exit 137, consistent with host
memory exhaustion. No scientific result or diagnostic payload was saved. Its
automatic retry was stopped; the
[reconciled cost](../../scratchpad/benchmarks/c0-all-diagnostics-20261022-01/billing-reconciliation.json)
was $0.52418051. This attempt provides no convergence result.

The fix in the shared checkout removes repeated encoding in the
[fit action](../../packages/grassmoth/src/grassmoth/actions/fit/action.py), shares
already immutable storage at the
[ownership boundary](../../packages/grassmoth/src/grassmoth/artifacts/immutability.py),
and keeps restored telemetry on the host during reporting. No diagnostic fields
or histories are removed. A bounded
[allocation check](../../scratchpad/diagnostics-memory-fix-20261022/allocation-profile.json)
with 64 MiB of telemetry and three report owners reduced additional peak
allocation from 192 MiB to about 3.4 KiB; this is not a full C0 memory measurement.

Nine focused ownership, reporting and continuation checks passed, as did root
lint. A short CPU fit through HTTP in `DIAGNOSTICSMEMORYFIX_20261022`, call
`call:be839c117ddeb83119525673b6e28559c6b12af302d04402c7ef7344d96dbebe`,
[saved all three diagnostic groups and all eight weight metrics](../../scratchpad/diagnostics-memory-fix-20261022/http-validation.json),
including lossless warmup and sampling histories with shared repeated buffers.
This smaller fit checks execution only. The full C0 GPU run has not been
repeated with the fix, so C0 and later-stage decisions remain unchanged.

## 2026-10-22 C0 diagnostic run, 512 warmup and 512 samples

At the user's request, the [short run](../../scratchpad/benchmarks/c0-diagnostics-512-20261022/00-fit-C0.json)
used 512 warmup sweeps and 512 retained draws per chain. The C0 model, all 2,049
readings per channel, four chains, 256 particles, seed 19, A10 selection and
proposal settings match the preceding all-diagnostics attempt. It used the
shared checkout with the diagnostics memory fix and the existing 32 GiB host
allocation. All diagnostic groups were enabled.

The action succeeded in study `C0SELFPRECONDITION_20261022`, call
`call:c0957cce2a15bf4b760d98964d9b02ab26fcf30ffc881d02dbd9478e8a9ad98f`,
with journal commit `8c4b1162b40b0f830ea88ebae5ea405ef3c56bed`. The
[validation](../../scratchpad/benchmarks/c0-diagnostics-512-20261022/validation.json)
confirms all 2,048 retained draws and all eight weight histories at 4,108 tree
nodes plus terminal selection. Each warmup and sampling history has shape
`(4, 512, 4109)`; the combined history has shape `(4, 1024, 4109)`.
Repeated chain and sampling-phase views share their wire buffers. The saved
response is 1,222,379,923 bytes. Reporting and publication completed without a
memory failure at these requested counts.

The [production report](../../scratchpad/benchmarks/c0-diagnostics-512-20261022/summary.json)
records the following chain measurements:

| Quantity | Bulk ESS | Tail ESS | Maximum R̂ |
| --- | ---: | ---: | ---: |
| Input effect | 200.7 | 261.2 | 1.0232 |
| Latent x, worst across coordinates | 38.2 | 21.7 | 1.0825 |
| Latent y, worst across coordinates | 13.7 | 24.3 | 1.2160 |

Parameter acceptance was 0.401, latent update fraction 0.623 and coordinate
frozen fraction 0.688. The lowest mean pair-weight ESS fraction occurred at the
first merge level: the node spanning indices `[58, 60)` averaged ESS 51.7 out of
65,536 candidates, a fraction of 0.000789. Other first-level nodes show similar
concentration. Terminal weights were uniform, with ESS 256 of 256, and no tree
weight summary recorded nonfinite measurements. These findings point toward
early merge proposal mismatch as an investigation target; this short run does
not establish its cause or a remedy. The complete
[sampler diagnostics](../../scratchpad/benchmarks/c0-diagnostics-512-20261022/sampler-diagnostics.json)
retain each node's reported measurements.

Sweep compilation took 49.7 seconds, warmup 68.0 seconds and sampling 67.7 seconds;
numerical duration was 199.2 seconds. Actual hardware was `gpu:NVIDIA A10:0`.
Runtime source digest:
`7ce5dd286fab3ea0fbe78adfa052e1f12562f930948523ff5e5f350142bad722`.
The GPU app `ap-CQ69eFVKq7g98PoaU2xqIS` is stopped, and the
[billing reconciliation](../../scratchpad/benchmarks/c0-diagnostics-512-20261022/billing-reconciliation.json)
reports $0.15554454, including result egress.

Decision: diagnostic retention succeeded at the requested counts. This is a
short diagnostic run, with failed convergence checks, and does not replace a
full-schedule C0 result. No later cases were run.

## 2026-10-22 C0 aMALA+ retry after gradient reuse

The user requested a full-schedule C0 retry with `amala_plus` after gradient reuse
was promoted to master in `a52d656a`. The [request](14-fit-C0-amala-plus.json)
uses 128 particles, four chains, 2,000 warmup sweeps and 4,000 retained draws per
chain, seed 19, all diagnostics enabled, and one A10 with a 3,600-second limit
and a $2 per-fit cap. The only request-setting change from the declared A0
policy is `drift: amala_plus`; current master includes parameter adaptation.

The new study is `C0AMALAPLUS_20261022`, served from the shared master checkout
at `http://localhost:19600`. Existing diagnostics-retention edits are retained
in the source patch. Model editing, seed-721 simulation and fit submission use
the public HTTP actions. The fit call is `call:c1256d77b2e187ae52ea7c3208ce32e0ea15cf7f0b81c907af13c1dd5508fda5`.
Full requests, source manifest, source patch, logs and results are retained in
`scratchpad/benchmarks/c0-amala-plus-a10-20261022/`. The fit is running; later
cases remain blocked pending its production convergence report.

The user stopped the run during warmup. Sweep compilation succeeded in 187.4
seconds; the last logged warmup progress was 626/2,000 sweeps. No retained draws
or convergence result were produced. Modal App `ap-rZQ2AXOa6rR98SpaKbQOz9` is
confirmed stopped with zero containers, and this run's local stack is stopped.
The initial billing report attributes $0.17344567 to the app; billing may lag.
This stopped attempt does not change C0's decision or unblock later cases.

## 2026-10-22 C0 latent-step adaptation × pilot mixture

The user approved a four-arm diagnostic comparison after the short C0 run.
The [declaration](../../scratchpad/benchmarks/c0-step-pilot-factorial-20261022/experiment.json)
pins the original C0 model and simulation, four chains, 256 particles, seed 19,
512 warmup sweeps and 512 retained draws per chain. The factors are adaptive
latent steps versus fixed `delta: 0.01`, crossed with the fixed pilot mixture
enabled or disabled. All arms use random-walk parameter proposals, no latent
drift, one latent coordinate block, Gaussian initialization at scale 0.1 and all
diagnostics. Each fit ran concurrently on its own A10 with 32 GiB host memory,
through HTTP in `C0SELFPRECONDITION_20261022` from the shared checkout.

All four actions succeeded and retained complete diagnostic histories. Their
runtime source digest is identical:
`9845181cf413321ada74ad749d208edf37275940d3d09d4bb3f1b679fae8e528`.
The adaptive/pilot control reproduced the preceding seed-19 chain diagnostics
exactly. The [comparison](../../scratchpad/benchmarks/c0-step-pilot-factorial-20261022/comparison-primary.json)
reads the production reports without recomputing their chain diagnostics:

| Latent step | Pilot | x bulk ESS | y bulk ESS | Movement RMS | Lowest first-merge mean weight ESS | Scales at cap 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Adaptive | On | 38.2 | 13.7 | 0.0369 | 51.7 | 80.5% |
| Adaptive | Off | 61.6 | 29.0 | 0.0383 | 69.1 | 92.1% |
| Fixed 0.01 | On | 20.5 | 20.3 | 0.0314 | 9,991.6 | 0% |
| Fixed 0.01 | Off | 14.6 | 15.3 | 0.0316 | 14,037.5 | 0% |

Each first merge scores 65,536 pairs assembled from 256 particles per side;
categorical weight ESS does not count independent trajectories. Fixing the
small step improves these weight measurements greatly but does not consistently
improve chain mixing. Disabling the pilot does not prevent scale saturation.
All input-effect means are between 0.602 and 0.605, with overlapping intervals.
Every short-run convergence report fails. These results do not establish a
production remedy or change the full-schedule C0 gate.

Primary fit calls, in table order:

- `call:ff7c6cce63a668426c017f555f362a437468aeb3e020fa5354cae3848532c188`
- `call:a819e3a6199926a1a1312eafe3dd376906a47f7bd2cbea7722b4519a470c3906`
- `call:53e67d41b0ee3f211b75fb6a77418aca3b53cfad1032c3f9b8e459120cf02283`
- `call:6a3b5f06d521a19ed209b3f3596f596be4f9b680672926ad99180ac254d5490f`

Complete responses, scalar summaries, source patch and lossless-retention checks
are retained per arm under `scratchpad/benchmarks/c0-step-pilot-factorial-20261022/`.
The four GPU apps are stopped. The
[primary billing reconciliation](../../scratchpad/benchmarks/c0-step-pilot-factorial-20261022/billing-primary-reconciliation.json)
reports $0.58192606 total, with every fit below its $2 cap.

The predeclared follow-up repeated the adaptive pilot-on/off contrast with seed
20 on separate GPUs. Both actions succeeded with the same source digest and
complete diagnostics. The seed-19 improvement did not reproduce:

| Seed | Pilot | x bulk ESS | y bulk ESS | x maximum R̂ | y maximum R̂ |
| --- | --- | ---: | ---: | ---: | ---: |
| 19 | On | 38.2 | 13.7 | 1.0825 | 1.2160 |
| 19 | Off | 61.6 | 29.0 | 1.0634 | 1.1085 |
| 20 | On | 62.5 | 28.6 | 1.0630 | 1.1120 |
| 20 | Off | 58.3 | 18.7 | 1.0701 | 1.1650 |

The seed-20 calls are
`call:9d6662fc560780bde81aeec50c26448b349091c978e9288f9652c1c1c9811bc0`
and `call:bb99f1038d9d62017fc846651fcf84990dae4c18b120c913752373983d0afb4d`;
their journal commits are `ef88a879270d066c628f6acd3feb1660959cb8fd` and
`bbfa404893848e9d6e1d5b96dae78a27a0b5d39c`. Seed-20 scale saturation was 84.6%
with the pilot and 90.3% without it. Both convergence reports fail.

The [complete comparison](../../scratchpad/benchmarks/c0-step-pilot-factorial-20261022/comparison.json)
retains both seeds, native phase durations and the production findings. Numerical
timings include unequal compilation-cache reuse, so they are not a controlled
cold-compilation comparison. No chain diagnostics were recomputed by the caller.
All six GPU apps are stopped; the
[final billing reconciliation](../../scratchpad/benchmarks/c0-step-pilot-factorial-20261022/billing-reconciliation.json)
reports $0.86224803 total, with each fit below $2.

Decision: neither fixed `delta: 0.01` nor pilot removal provides a reproducible
mixing remedy in these runs. Low merge-weight ESS alone does not establish the
cause of slow MCMC mixing, and the pilot alone does not explain scale saturation.
The short-run findings do not select a production policy or change C0's gate.
No other experimental families or later benchmark cases were launched here.

## 2026-10-23 C0 latent-path branch: Kalman path move and smoother-marginal proposal

The user approved testing two dynamics-aware latent proposals against today's
sampler on C0. Per-frequency analysis of the earlier C0 draws showed the failure
sits in the smooth path components: periods above about 64 steps refreshed 5–14%
per sweep against about 40% for short wiggles. The dSMC paper (§4) and both
auxiliary-sampler papers name this regime as the weakness of time-separable
proposals.

Each arm ran from its own checkout, server and study at the full schedule: 128
particles, four chains, 2,000 warmup and 4,000 retained draws, seed 19, one A10.
All three checkouts are `a52d656a` plus the diagnostics memory fix used by the
2026-10-22 factorial. Diagnostics: production checks plus `particle_identity` and
`parameter_movement`. The `particle_weights` histories were omitted because they
would make each response about 3 GB at this schedule. Each study replayed
the seed-721 simulation; all ten arrays match the 2026-10-22 C0 data exactly.
Patches, declaration, requests, responses and analysis are in
`scratchpad/latent-proposals-20261023/`.

- **Base (A0):** unchanged. Study `LATENT_BASE_20261023`, call
  `call:3fb79e74d205c9066b133602c22894a39fa2d1a467f9027fb08b70c50be7ce34`.
- **Kalman:** A0 plus an auxiliary Kalman path move after every sweep
  ([Corenflos and Särkkä](https://arxiv.org/abs/2303.00301), Algorithm 2, sampled
  parallel in time). It draws u ~ N(x, δ/2), proposes from a Gaussian path model
  conditioned on u, and accepts against the exact target.
  - The Gaussian model expands the exact transitions and point densities around a
    path. It is built at initialization, rebuilt every 100 warmup sweeps at each
    chain's window-mean parameters, and frozen for the retained draws.
  - δ adapts during warmup towards 25% acceptance.
  - Study `LATENT_KALMAN_20261023`, call
    `call:5e422cdb76e0a11ab3c1567dfd22bca7ac5aef50a94aba34246ba7f297028bbf`.
- **Marginals:** the fixed pilot holds Gaussian-smoother marginals built at
  initialization ([dSMC §4.1](https://arxiv.org/abs/2202.02264)), as the 80%
  mixture component. The latent step is fixed at the cap, δ = 10; with an adapted δ
  the auxiliary factor vetoes non-local particles and δ collapses.
  - Study `LATENT_MARGINALS_20261023`, call
    `call:5bc3ef32003226a1e6ce1206891c633bd77649ec20d8af4167105e60f8a4e8cb`.

| Arm | Production gate | x bulk / tail ESS | y bulk / tail ESS | Max latent R̂ | Input-effect bulk ESS |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | **fails** | 434 / 333 | 194 / 268 | 1.037 | 1,779 |
| Kalman | **passes** | 12,105 / 11,775 | 12,210 / 11,984 | 1.001 | 1,899 |
| Marginals | **fails** (x tail) | 501 / 369 | 1,246 / 998 | 1.010 | 1,848 |

Input-effect means: 0.602, 0.599 and 0.602, all with SD about 0.088.

Per-sweep refresh of the path's frequency components (smoothest band → wiggliest
band), from the retained paths:

| Arm | x | y |
| --- | --- | --- |
| Base | 0.08 → 0.35 | 0.05 → 0.35 |
| Kalman | 0.93 → 0.95 | 0.93 → 0.95 |
| Marginals | 0.30 → 0.38 | 0.29 → 0.41 |

- **Kalman move:** retained acceptance was 0.92–0.93 in every chain, with δ at its
  1e4 cap. Each move is effectively an independent draw from a near-exact proposal.
- **Timings:** sweep compilation took 49 s (base), 56 s (Kalman) and 44 s (marginals).
  Warmup took 61, 66 and 60 s, and sampling 121, 130 and 121 s. The marginals arm
  also spent 11 s building the smoother at initialization.
- **Cost:** Modal billing reports $0.54 for the three fits, $0.17–$0.19 each.

Proposed decision: the Kalman path move passes C0 at about 7% extra sweep time,
with 28–63× the worst-coordinate ESS. The smoother-marginal proposal helps
smooth modes about 4–6× but still fails x's tail ESS.

Before the Kalman move joins the policy:
- **User decision:** whether its Gaussian model may be rebuilt every sweep at the
  current parameters. AGENTS.md and the model assumptions allow linearization only
  at initialization; this run froze it after warmup.
- **Framework work:** tests, lint and the fit chart. The move still rejects
  intervals, replay, exact readings on free coordinates and tempering, which
  FRANCE1940V2 needs.

## 2026-10-23 — C0 aGRAD comparison: timing exceeds the approved cap

Source: `4ed4c79205dbbaeb9d0cd1cdfb9cba5addcd95e4`. The standard stack was
restarted on that commit. Fresh format-30 study `C0GUIDED_20261023` replayed
the stage's question, truth model, seed-721 simulation and C0 model using the
[public study actions](../../.agents/skills/grassmoth-study-api/SKILL.md).
The run declaration, resolved requests, saved responses, Modal logs and billing
reports are in `scratchpad/benchmarks/c0-guided-20261023/`.

- Question: `5401314a4279b968937cabbcbb296cc2fd9914e4`.
- Truth model: `6c3548a436e0e7d81f202880a600e1cb52b6944d`.
- Simulation: `36e7d2a9d3ffbb3a6c97acddcd08a1d62c5e3e5f`, replicate 0.
- C0 model: `366c5db2fe534fbef7fc6fd217cbc6b0e73b30bb`.

Requests 15–19 declare A0 and both new methods at κ = 0 and 1. A0's request
differs from request 04 by omitting `particle_weights`; all five retain
`particle_identity`, `parameter_movement` and production diagnostics. Their full
schedule remains 128 particles, four chains, 2,000 warmup sweeps and 4,000
retained draws per chain, seed 19, A10, and a 3,600-second limit.

Requests 20–21 are timing probes only: 64 warmup and 64 retained sweeps per
chain, with the full particle and chain counts and κ = 1. They ran concurrently
on separate A10s and completed successfully. Both record numerical source digest
`7379d048cdd4507b386f66d05d3ec18a5facda399c88361e1de845d108ef137e`.

| Timing probe | Sweep compilation | Warmup, 64 sweeps | Sampling, 64 sweeps | Total execution | Projected full execution | Actual Modal cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Particle-aGRAD, κ = 1 | 8.27 s | 58.58 s | 56.68 s | 132.24 s | 5,389.93 s (89.83 min) | $0.09074948 |
| Twisted Particle-aGRAD, κ = 1 | 13.52 s | 59.02 s | 59.15 s | 141.26 s | 5,564.53 s (92.74 min) | $0.09352754 |

Projection keeps the measured non-sweep execution time and scales the two sweep
phases to 2,000 and 4,000 respectively. It is a timing estimate, not a completed
full-schedule fit. All recorded phase intervals and runtime settings are in
`timing-summary.json`; `billing-after-timing.json` records **$0.18427702 total**.

- Particle-aGRAD timing call:
  `call:9fbafb259163ed002407e5cd2a00fa769201f383de6020eb6256360da38344ea`;
  Modal app `ap-IoTzBxS9aiLC2xCkwUsyvq`.
- Twisted Particle-aGRAD timing call:
  `call:f6fb611398d6bee4bc99ad2c0d96b1e425c21e34170c7f80a98b771f24cefd7e`;
  Modal app `ap-YJuwfejpehVdxQAFnVsbf1`.

The timing estimates exceed the brief's 3,600-second/$2 cap. The user chose to
investigate optimizations first, and the five full arms have **not been submitted**.
A0 and both κ = 0 arms therefore have no new full-fit measurements. The two short probes
are not stage results: they establish no full-schedule gate, ESS comparison,
frequency-refresh comparison or agreement between passing arms.

The optimization investigation and isolated prototype are recorded in
[`scratchpad/guided-optimization-findings.md`](../../scratchpad/guided-optimization-findings.md).
Its kernel timings are not stage results. The main-checkout sampler, benchmark
README and full schedule are unchanged. No C1 work or commit was performed.

Validation: `bun run lint` passed (7 tasks); the saved-draw analysis script passed
`lint:public`. Its DCT-II transform was checked numerically against the previous
C0 frequency-refresh formula. No new tests were needed for request declarations.

## 2026-10-23 — C0 guided comparison on optimized master

At the user's approval, resumed `C0GUIDED_20261023` on master `b076b7ad` (including early inlining from `9108f6a2`). Requests **15–19** used their saved settings and exact C0/simulation revisions: **128 particles, four chains, 2,000 warmup, 4,000 retained draws per chain, seed 19, separate A10s**, a 3,600-second limit and $2 cap per fit. All had production checks plus `particle_identity` and `parameter_movement`; no timing runs were repeated. No schedule was reduced. The shared worker/API were restarted after verifying zero active actions; no other agent's run was killed.

**Both κ=0 methods pass; A0 and both κ=1 methods fail production convergence.** All five actions completed and retained the full draws. Detailed phase timings, input-effect summaries, per-arm bulk/tail ESS per total execution second, frequency-band refresh, mean agreement, raw results and billing are in the [run report](../../scratchpad/c0-guided-resume-20261023/findings.md) and [machine-readable comparison](../../scratchpad/c0-guided-resume-20261023/comparison.json).

| Arm | Gate | x bulk/tail ESS | y bulk/tail ESS | Largest latent R̂ | Total numerical execution | Worst-latent bulk ESS/s | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | failed | 460/463 | 181/298 | 1.0276 | 342.65 s | 0.527 | $0.22070 |
| aGRAD κ=0 | passed | 1427/1058 | 2200/2333 | 1.0030 | 677.28 s | 2.106 | $0.39590 |
| aGRAD κ=1 | failed | 32/62 | 30/63 | 1.1113 | 748.61 s | 0.041 | $0.43485 |
| Twisted κ=0 | passed | 1851/1373 | 2850/2154 | 1.0033 | 684.77 s | 2.703 | $0.41329 |
| Twisted κ=1 | failed | 0/0 | 0/0 | unavailable | 766.60 s | 0.000 | $0.43345 |

A0 fails y ESS. Both κ=1 methods fail latent convergence; twisted κ=1 triggers the frozen-coordinate failure (zero worst ESS, unavailable R̂). The κ=0 arms' input-effect means differ by 0.000187, only 0.068 combined MCSE, and their mean trajectories agree on the conservative saved worst-coordinate MCSE scale. Smoothest-band refresh is about 0.80–0.85 for κ=0, versus 0.046–0.078 for A0 and 0.0066–0.0078 for κ=1.

**Proposed decision:** prefer twisted Particle-aGRAD κ=0 as the C0 candidate: about 28% better worst-latent bulk and tail ESS per recorded execution second than ordinary aGRAD κ=0. Acceptance into the policy and C1 remain the user's decisions; this run stops at C0. One seed/history does not establish a general ranking.

Total actual Modal charge: **$1.89819050**. All five apps stopped, each below $2. ESS/s includes initialization, compilation, warmup and sampling; the report also gives complete action wall times and wall-time ESS/s. The large saved-result read caused shared API child exits; a temporary read-only single-worker API retrieved the original results without code changes or refits, then was stopped.

| Request | Call ID |
| --- | --- |
| 15 · A0 | `call:9256398a18f3c6521a4ab0cf41c3791c9419a2a0f23e89d3554a94ba29c6d23a` |
| 16 · aGRAD κ=0 | `call:3ea493b6d3c557ee0cddb487fbba1c94bdb3d11b85ac2d24de19262de075d2ab` |
| 17 · aGRAD κ=1 | `call:e0e2dfc6ae1d636e981b64f3875bcdad967d9d64c3a463988b1cd6b5e5cb98da` |
| 18 · Twisted κ=0 | `call:614c9782d0e5f6a09b0d1ffd9b39e110f15de9aa542ee070aaf3762e3f678785` |
| 19 · Twisted κ=1 | `call:b31e8479dcbea4b3dbb60b632323c839f07aa0b59fead6521cf7d95eb2affb37` |
