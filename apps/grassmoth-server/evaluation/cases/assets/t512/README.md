# T=512 through production actions

This case keeps the historical stress test's nonlinear driver and outcome, static
baseline, alternating known input, deterministic accumulator, interval observations,
mixed likelihoods and sparse exact driver reading. Models are authored by `edit_model`;
the data come from `simulate`, then the declared mask, then `prepare_data`.

The case changes are deliberate:

- The grid has 512 nodes, three hours apart, from 2020-01-01 through 2020-03-04 21:00 UTC.
  This replaces the seed-721 irregular grid with spacings between 0.09 and 0.15 days.
  The input still alternates every eight intervals. The production SDE simulator
  replaces the handwritten Euler generator, so sharing seed 721 does not reproduce
  the historical draws.
- Mean-driver, standard-deviation-outcome and count readings use the production
  18-hour windows, retained at every sixth node. Historical windows overlapped,
  ended at every fourth node from row 7, and began off-grid.
- The accumulator has noisy point readings (standard deviation 0.2) at rows
  0, 32, ..., 480 and 511, plus its exact known zero origin at row 0. These retain
  and anchor it in the existing DSL. Its dynamics remain `da/dt = driver`, with
  zero initial uncertainty and zero diffusion; particle inference must replay it.
- Interval counts use Poisson instead of negative binomial. The categorical channel
  is omitted; the binary outcome remains. Fixed auxiliary dispersion and vector
  category coefficients are not supported as literal fitting inputs by the current
  compiler. No extra priors or new DSL support are introduced to imitate fixed values.
- Initial states are independent, with the historical marginal initial standard
  deviations. The current causal projection could not retain the historical shared
  initial uncertainty. The correlated driver/outcome innovations remain, with their
  explicit unmeasured process-noise parent in the scientific DAG.

`truth.json` contains literal generating coefficients. The fit model frees the input
and cross effects and retains the baseline's initial Normal uncertainty. The variant
`unknown_diffusion.json` additionally frees the driver diffusion scale (HalfNormal 0.6;
truth 0.45), using exactly the same observations. Both fit models preserve the
scientific parameter identities. Historical reference draws cannot qualify either case.

`generation-request.json` declares seed 721 and one replicate. `generation.json`
records the exact simulate request, its truth artifact revision, the resulting action
commit, findings and content hashes. `generated.csv` is exported directly from the
retained observation history, with UTC anchors and support. `mask.json` removes only
declared values to make `observations.csv`; it retains the original driver/outcome
missingness patterns and the exact driver reading at row 17. The regeneration command
is documented in the [evaluation guide](../../../README.md#saved-t512-data).
