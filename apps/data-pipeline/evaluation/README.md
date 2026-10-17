# Action-built inference experiments

This package schedules production [study actions](../../../../.agents/skills/nof1-study-api/SKILL.md).
It has no alternate model compiler, simulator, sampler, or convergence gate. Importing it
executes nothing. Run a declaration only after its compute budget and case are approved.

## Maintained pieces

- `cases/` contains immutable model documents and action recipes. `tiny.json` is a small
  nonlinear CPU debugging case; `nonlinear.json` replaces the former Python fixture.
  `t512.json` and `t512_unknown_diffusion.json` share a saved synthetic history.
  `france1940.json` reads its pinned model and source CSV from local `data/FRANCE1940`
  at run time; those assets are not copied into this package.
- `declarations.py` owns experiment inputs: case, resolved base controls, declared factor
  or interaction, paired generation/sampler seeds, repeats, optional frozen-checkpoint
  probes, and predeclared reference tolerances.
- `recipes.py` authors questions/models, generates or prepares data, and consumes the
  actions' structured specification, identification and data findings. An allowed
  failed finding needs a case-specific reason. Unavailable checks require an explicit
  unavailable-reason declaration; accepting an unavailable check never accepts a failed
  one with the same code. Neither bypasses production fit preflight.
- `sources.py` verifies every declared source hash before submitting any action. A local
  study model is read from its pinned result tree without opening the study for writes
  or changing its storage format. Missing inputs and hash mismatches stop the recipe.
- `run.py` schedules one workspace per arm and seed pair, verifies model/data content,
  rejects undeclared control differences, and retains complete action envelopes.
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
`drift_probe.json` first obtains the same retained base checkpoint in each arm, then
continues it with zero warmup and the declared drift. `initializer_score.json` records
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
Local arms run serially to avoid contention; concurrent remote arms each allocate
their own fit device. A reused call is visibly marked and is not a new timing sample.

`repeats` advances fit's `repeat` identity without changing its random seed. For more
fresh timings of an already executed declaration, increase `repeats`; changing the
sampler seed asks a different stochastic question. Output directories must be new.
Stable workspace names permit saved calls to be reused, and every action is pinned
to its scientific input revisions. The client reads produced question, model and panel
artifact revisions from the public timeline; simulation and fit references select their
action commits. The output preserves case and attempt manifests and raw
MessagePack envelopes, including failures. Continued fits report the earlier and
new draws as one chain; segment bounds identify the new observations of the sampler.

For posterior accuracy, declare `posterior.references` separately for every generation
seed and provide a saved reference's case manifest and call ID. The scientific model
and observations must match. Choose tolerances before looking at the experiment.
Truth recovery is separate from same-data posterior agreement. Predictive comparisons
use the production `simulate` and `data_diff` actions on the pinned fitted revisions;
no benchmark-local prediction engine is supplied.

## Accounting and provenance

`reservation_usd_per_fit` is a planned upper budget, including base fits used by probes.
Recorded duration and the union of compile intervals measure execution, not GPU billing.
Supply saved output from `modal billing report` with `--billing-report PATH` to retain
its original text, filename and SHA-256 digest beside the measured results. Reconcile
its job allocations separately; the runner does not estimate an invoice from duration.
Missing billed cost is unknown, never zero. No billing command or paid job is launched
by importing this package or by generating its report.

Exploratory declarations and generated outputs belong in gitignored `scratchpad/`.

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
uv run python -m evaluation.fixtures.t512 \
  --endpoint http://localhost:18100 --workspace BENCHT512REGEN \
  --output ../../scratchpad/t512-regenerated
```

This command calls `edit_question`, `edit_model` and `simulate`, exports the retained
history, and applies the committed mask. It runs no fits. Compare the exported files
and recorded findings before updating case hashes. Recipes use the committed CSV,
so changing an experiment's generation seed does not regenerate these recorded cases.

The [case note](cases/assets/t512/README.md) owns the differences from the historical
handwritten stress fixture. Its old posterior is not a reference for this changed case.
