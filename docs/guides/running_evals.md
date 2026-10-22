# Running inference experiments

The maintained [experiment package](../../apps/grassmoth-server/evaluation/README.md)
builds cases through production actions and schedules paired fits. Its declarations
pin the case, controls, treatment, seeds, repeats and any accuracy tolerances.

Read the [integration guide](agentic_integration_testing.md) before starting services.
Run a case or experiment only with an approved compute budget. From
`apps/grassmoth-server`, against an authorized study facade:

```bash
uv run python -m evaluation.run evaluation/experiments/cpu_compile.json \
  --endpoint http://localhost:8100 \
  --output ../../scratchpad/benchmarks/cpu-compile-01
```

The example uses local hardware and zero retained samples. It is intended to inspect
compilation, not establish posterior accuracy. The
[declaration owner](../../apps/grassmoth-server/evaluation/declarations.py) defines
supported contrasts; the [fit chart](../assets/action-flows/fit.svg) owns continuation
and compile-only behavior. The [sampler contract](../../packages/grassmoth/src/grassmoth/artifacts/sampling.py)
owns production choices and defaults.

Each run writes a sortable HTML report, raw action results and case/attempt manifests.
It rejects undeclared control differences and mismatched scientific inputs. Reserved
budget, recorded compute time and separately supplied billing output remain distinct.
See the [package guide](../../apps/grassmoth-server/evaluation/README.md#accounting-and-provenance)
for cost reconciliation and the declaration examples for initializer and movement probes.

The package never launches a case on import. Historical experiment scripts and
outputs belong in gitignored `scratchpad/`; maintained cases use the public actions.
