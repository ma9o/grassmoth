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

The [request](13-fit-C0-A0-256-particles.json) runs through HTTP in study
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
