# Dense and sparse parameterization comparison

The two cases use the same 129-node six-hour grid over 32 days, two nonlinear
stochastic states, a known alternating input, and one seed (8029, one replicate).
The dense outcome is measured at every node. The sparse state is measured at
rows 0, 8, …, 128. Both diffusion scales are unknown in the fit model
(HalfNormal 0.8; truth 0.35 and 0.45), together with the direct input effect
(Normal 0, 1; truth 0.7). The sparse state also drives the dense outcome.

`precise` uses observation standard deviation 0.05; `noisy` uses 2.0. Everything
else in their authored scientific models, generation controls and observation
layout is identical. This tests the frozen presence-based mask when point readings
carry very different amounts of information. The mask is not retuned by variant.

The production actions author the truth and fit models with `edit_model`, generate
one history with `simulate`, and import the masked CSV with `prepare_data`.
The compiler's identification checks require the input effect to be identified.
Each variant's `generation.json` records the exact requests, truth revision,
simulation revision, findings and hashes. `generated.csv` retains every generated
reading and support; the common `mask.json` removes only sparse values to produce
`observations.csv`. The recipes pin every source and provenance file by hash.

The declared simulation exceptions cover the one-replicate scale screen and the
noisy arm's intentionally weak transmission. The sparse observation gap and the
known input's repeated ±1 values are declared data-quality exceptions. None
relaxes the inference convergence gate. The production
SDE simulator and inference's Euler–Maruyama law have distinct discretizations;
recovery and solver-gap assessment remain part of the later pilot.

Regenerate a variant on an explicitly authorized isolated CPU stack:

```sh
uv run python -m evaluation.fixtures.recorded \
  --recipe evaluation/cases/dense_precise.json \
  --endpoint http://localhost:29100 --workspace CHARTDENSEREGEN \
  --output ../../scratchpad/dense-regenerated
```

This command runs no fit. Review its outputs and findings before updating the
pinned assets. The experiment declarations in `evaluation/experiments` use
`algorithm.parameterization` as their sole factor and three sampler seeds. They have not
been run; their local hardware assignment does not authorize a GPU campaign.
