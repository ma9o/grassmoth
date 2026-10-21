# Action-built inference experiments

This package schedules production [study actions](../../../../.agents/skills/nof1-study-api/SKILL.md).
It has no alternate model compiler, simulator, sampler, or convergence gate. Importing it
executes nothing. Run a declaration only after its compute budget and case are approved.

## Maintained pieces

- `cases/` contains immutable model documents and action recipes. `tiny.json` is a small
  nonlinear CPU debugging case; `nonlinear.json` replaces the former Python fixture.
  `t512.json` and `t512_unknown_diffusion.json` share a saved synthetic history.
  `france1940.json` reads its pinned model through the study API and its source CSV
  from local `data/FRANCE1940`; those assets are not copied into this package.
- `declarations.py` owns experiment inputs: case, resolved base controls, declared factor
  or interaction, paired generation/sampler seeds, repeats, optional frozen-checkpoint
  probes, and predeclared reference tolerances.
- `recipes.py` authors questions/models, generates or prepares data, and consumes the
  actions' structured specification, identification and data findings. An allowed
  failed finding needs a case-specific reason. Unavailable checks require an explicit
  unavailable-reason declaration; accepting an unavailable check never accepts a failed
  one with the same code. Neither bypasses production fit preflight.
- `sources.py` verifies every declared source hash before submitting any action. Saved
  models come from successful `edit_model` calls through the public client. Missing
  inputs and hash mismatches stop the recipe.
- `run.py` verifies model/data content, rejects undeclared control differences and
  retains complete action envelopes. Fresh fits use a workspace per arm and seed pair;
  probes share their anchor's workspace.
- `comparison.py` qualifies references using production convergence plus the declared
  R-hat/MCSE limits, compares marginal means and standard deviations with an MCSE
  uncertainty band, and compares each delivered parameter start and state path with
  a qualified reference in the same public scientific coordinates. It reports
  reference-scaled distances, ensemble spread and marginal reference coverage,
  separately from initialization preparation time; the initializer's own density
  is not a quality score.
- `report.py` writes one sortable HTML table. Assigned settings, adapted tuning,
  phase intervals, movement diagnostics, truth recovery and reference comparisons
  remain distinguishable. Summary equivalence does not prove equality of joint laws.

The [action charts](../../../../README.md#documentation) own execution flow;
[the sampler contract](../src/nof1_causal_lab/artifacts/sampling.py) owns option meanings;
[model assumptions](../../../../docs/assumptions.md) own the scientific limits.

## Declarations and runs

From `apps/data-pipeline`, after an authorized stack is available:

```bash
uv run python -m evaluation.run evaluation/experiments/cpu_compile.json \
  --endpoint http://localhost:8100 \
  --output ../../scratchpad/benchmarks/cpu-compile-01
```

The examples are declarations, not evidence that a case has run. `cpu_compile.json`
compares fixed trajectory lengths with zero warmup and zero retained draws.
`drift_probe.json` obtains one retained base checkpoint through an ordinary fit,
then submits each arm in that workspace with `ContinueFitTarget` pinned to the
anchor's revision, explicit `probe_warmup` and the declared drift. An optional
`probe_anchor` selects an already saved fit by workspace, call ID and case manifest;
its seed and chain count must match. The workspace lock serializes these arms,
so their wall-clock durations add up even though every fit gets its own GPU.
The runner never copies study storage. `initializer_score.json` records
starting values and preparation cost with no posterior draws. Add a qualified
`posterior.references` entry to score those starts; without one the report says
"not scored". Zero-draw arms skip posterior equivalence. These small counts are
for debugging; they do not certify a posterior or qualify a reference.

Every arm assigns exactly the declared paths. A whole algorithm object can be a
single treatment, or several paths require `interaction: true`. Code changes use
`runtime.source_digest`; compiler changes use `runtime.compiler.<name>` and may
need separate endpoints. The runtime must match the declared value. Undeclared
source, version, compiler, hardware, seed or sampler-control differences invalidate
a pair. Adapted scales, acceptance and movement are outcomes, not assigned controls.
Local arms run serially to avoid contention; remote arms in separate workspaces each
allocate their own fit device concurrently. Shared-workspace probes serialize.
A reused call is visibly marked and is not a new timing sample.

`repeats` advances fit's `repeat` identity without changing its random seed. For more
fresh timings of an already executed declaration, increase `repeats`; changing the
sampler seed asks a different stochastic question. Output directories must be new.
Stable workspace names permit saved calls to be reused, and every action is pinned
to its scientific input revisions. The client reads produced question, model and panel
artifact revisions from the public timeline; simulation and fit references select their
action commits. The output preserves case and attempt manifests and raw
MessagePack envelopes, including failures. Continued fits report the earlier and
new draws as one chain. The experiment reads the latest segment's saved diagnostics
and convergence from the fit output, and applies truth errors and reference comparison
to that segment. The [fit chart](../../../docs/assets/action-flows/fit.svg) owns when
segment and combined-chain diagnostics are produced. A failed equivalence
test is labelled inconclusive unless the mean uncertainty band is outside the
declared tolerance or a fixed coordinate disagrees; a standard-deviation tolerance
failure alone does not establish a precise discrepancy.

For posterior accuracy, declare `posterior.references` separately for every generation
seed and provide a saved reference's case manifest and call ID. The scientific model
and observations must match. Choose tolerances before looking at the experiment.
Truth recovery is separate from same-data posterior agreement. Predictive comparisons
use the production `simulate` and `data_diff` actions on the pinned fitted revisions;
no benchmark-local prediction engine is supplied.

## Accounting and provenance

Paid experiment runs require `--budget POLICY.json`. The [evaluation ledger](budget.py)
checks the review hold file and reserves each fresh fit before its public POST. It
also covers newly prepared anchors; saved anchors and reused calls do not
allocate again. A failed call retains its debit, and a lost response retains both
its debit and scheduling slot until recovered or reconciled. The policy directory
can be shared across stages and scheduler processes; admission uses a file lock.

For example, an operator-supplied policy is:

```json
{
  "directory": "/absolute/campaign/ledger",
  "hold_file": "/absolute/campaign/CLAUDE-HOLD",
  "rates": {"L40S": "0.001"},
  "max_usd": "10",
  "max_concurrent": 4,
  "termination_seconds": 60,
  "extra_usd_per_fit": "0.05"
}
```

Rates must cover GPU, CPU and memory; the example is illustrative, not a current
price quote. The resolved fit's `time_limit_seconds` is sent to production in its
settings. Reserve `rate × (2 × time_limit_seconds + termination_seconds)` plus the
extra per-fit allowance: Modal startup and computation each have that time bound.
Price image builds and other overhead into the extra allowance. Provider invoices,
not inference duration, determine actual cost. The [fit chart](../../../docs/assets/action-flows/fit.svg)
owns timeout/failure behavior. There is no benchmark hook or watchdog in production.

Save `modal billing report --for today --show-resources --json` and
`modal app list --json`, then provide reviewed allocation-to-app bindings:

```json
{
  "<allocation identity from ledger/allocations>": {
    "app_ids": ["ap-..."],
    "evidence": "provider trace tying this app to the recorded fit call"
  }
}
```

Production apps share a name. Match IDs from provider execution evidence; do not
infer identity from overlapping timestamps or the app name. Unmapped calls retain
their full reservation. Reconcile the saved exports without launching any work:

```bash
uv run python -m evaluation.budget --policy POLICY.json \
  --billing-report billing.json --apps apps.json --bindings bindings.json
```

The ledger retains the original exports, their SHA-256 hashes and every reconciliation
snapshot. Only stopped apps with reported charges settle to their invoices. Missing
billing is unknown, never zero; no bound app may settle two reservations. Supply the
complete binding set on each reconciliation. Successful or failed completion alone
never releases money. The report shows the ledger alongside measured phase times;
`--billing-report PATH` also retains the source billing export beside the experiment.
Importing the package or generating a report executes no scientific or paid job.

Exploratory declarations and outputs belong in gitignored `scratchpad/`. The
[storage configuration](../src/nof1_causal_service/storage/backend.py) owns
`NOF1_DATA_ROOT`, the general local-storage deployment override for an isolated stack.

## Saved T=512 data

Both T=512 recipes import the same `assets/t512/observations.csv` through `prepare_data`
using recorded readings and temporal supports. The unknown-diffusion variant frees only
the driver's diffusion scale. The literal truth model, seed 721, one-replicate simulate
request, returned truth revision and source hashes are retained beside the CSV.
`generated.csv` is the unmodified simulator output; `mask.json` declares the only
subsequent transformation. The mask removes readings without changing any retained
value, timestamp or interval support.

To regenerate into a new review directory on an explicitly authorized isolated stack:

```bash
uv run python -m evaluation.fixtures.recorded --recipe evaluation/cases/t512.json \
  --endpoint http://localhost:18100 --workspace BENCHT512REGEN \
  --output ../../scratchpad/t512-regenerated
```

This command calls `edit_question`, `edit_model` and `simulate`, exports the retained
history, and applies the committed mask. It runs no fits. Compare the exported files
and recorded findings before updating case hashes. Recipes use the committed CSV,
so changing an experiment's generation seed does not regenerate these recorded cases.

The [case note](cases/assets/t512/README.md) owns the differences from the historical
handwritten stress fixture. Its old posterior is not a reference for this changed case.

The [dense rung](cases/assets/dense/README.md) uses the same recorded-data machinery
for precise and noisy variants with coexisting dense and sparse components and
unknown diffusion. `parameterizations_dense_precise.json` and `parameterizations_dense_noisy.json` declare
`algorithm.parameterization` as the sole arm factor (`centred`, `non_centred`, `mixed`). The
latest segment's parameterization and actual mixed mask are retained in evaluation telemetry,
including compile-only calls. References and GPU budget approval belong to the
later pilot; these declarations do not run it.
