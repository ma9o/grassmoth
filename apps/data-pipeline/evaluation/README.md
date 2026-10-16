# Action-built inference experiments

This package schedules production [study actions](../../../../.agents/skills/nof1-study-api/SKILL.md).
It has no alternate model compiler, simulator, sampler, or convergence gate. Importing it
executes nothing. Run a declaration only after its compute budget and case are approved.

## Maintained pieces

- `cases/` contains immutable model documents and action recipes. `tiny.json` is a small
  nonlinear CPU debugging case; `nonlinear.json` replaces the former Python fixture.
  `france1940.json` imports a pinned model and hash-verified source CSV through actions.
- `declarations.py` owns experiment inputs: case, resolved base controls, declared factor
  or interaction, paired generation/sampler seeds, repeats, optional frozen-checkpoint
  probes, and predeclared reference tolerances.
- `recipes.py` authors questions/models, generates or prepares data, and consumes the
  actions' structured specification, identification and data findings. An allowed
  failed finding needs a case-specific reason. Unavailable checks require an explicit
  unavailable-reason declaration; accepting an unavailable check never accepts a failed
  one with the same code. Neither bypasses production fit preflight.
- `run.py` schedules one workspace per arm and seed pair, verifies model/data content,
  rejects undeclared control differences, and retains complete action envelopes.
- `comparison.py` qualifies references using production convergence plus the declared
  R-hat/MCSE limits, compares marginal means and standard deviations with an MCSE
  uncertainty band, and scores the initializer's retained starting values directly.
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
continues it with zero warmup and the declared drift. `initializer_score.json` scores
starting values and preparation cost with no posterior draws. These small counts are
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
to its returned revision. The output preserves case and attempt manifests and raw
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
The original T=512 data must enter through `prepare_data`, preserving its timestamps,
missing values and interval support. Its recipe is pending the documented DSL gap and
source-data decision; do not substitute a new simulation schedule or a hand-built target.
