# Synthetic hidden-norm policy benchmark

This archive supports the [README comparison](/docs/README.md). It preserves the selected results from 7 November 2026, the exact fitted Stan model and data, native study inputs, and a compact record of the Stan campaign. It contains no new fits or recomputed diagnostics. [comparison.json](comparison.json) records source hashes, original repository-relative artifact identifiers, settings, group statistics and timing definitions.

## Model and data

The synthetic model couples log incidence `x` and an unobserved protective norm `z`, with a known daily policy input `u`. It represents a possible decision about withdrawing a temporary protective-behavior campaign; it is not a field-validated policy model. There are 360 daily time points, 52 weekly noisy observations of `x`, and no observations of `z`. The posterior has 722 physical coordinates: 720 latent states and two global parameters. The known input is not a posterior coordinate. The sigmoid feedback in the norm can sustain distinct low- and high-protection histories. [model.stan](model.stan) is the exact selected Stan source; [data.json](data.json) is its exact input.

Both fits use Gaussian priors `a ~ Normal(0.2, 0.15)` and `h ~ Normal(0.5, 0.05)`, Gaussian initial states `x[1] ~ Normal(1.5, 0.5)` and `z[1] ~ Normal(0, 0.1)`, daily process standard deviations 0.08 and 0.03, and observation standard deviation 0.2. The fitted incidence transition includes the small drift anchor `-1e-6*x`. Both samplers target this same nonlinear model; there is no latent-path chart in the selected particle run. Its two Gaussian global priors are standardized. See [model.stan](model.stan) and the archived [native model request](model-request.json).

The original data generator used seed 5, fixed initial values `x=1.5, z=0`, and omitted that anchor. Thus this is not a prior-predictive dataset, and the generating model differs slightly from the shared fitted target. The approximate numerical reference also omits the anchor and retains 98.91% of its parameter-grid mass. Its roughly one-percentage-point resolution is an empirical limitation, not a uniform error bound. The README diagnostics do not depend on agreement with this reference. [comparison.json](comparison.json) preserves the generator and reference provenance.

## Selected comparison

| Saved configuration | Fit time, including warmup (s) | Worst R-hat | Minimum bulk ESS | Minimum tail ESS | Minimum bulk ESS/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPGibbs + particle-mGRAD, M=2, N=256, L40S | 125.40 | 1.0034 | 2255.06 | 1124.47 | 17.9833 |
| Pathfinder starts + centered Stan NUTS, M1 Max | 118.17 | 1.2814 | 20.98 | 50.05 | 0.17758 |

These are common-score statistics over all 722 physical coordinates, using ArviZ-stats 1.3.3, rank-normalized R-hat, bulk ESS and 5%/95% tail ESS in float64 diagnostic arithmetic. Both runs retain 8 × 16,000 draws, with seed 19 and no thinning. The particle run has 2,000 warmup iterations per chain; Stan has 1,000. A strict pass requires every R-hat below 1.01, every bulk/tail ESS at least 400, no frozen chain, and completion within the applicable budget. The selected particle run passes; Stan fails despite zero divergences and no tree-depth saturation. Exact values and resolved settings are in [comparison.json](comparison.json).

Fit time has different owners. The GPU's native fit duration includes initialization, JIT, warmup, sampling and native diagnostics. The Stan driver includes Pathfinder, initialization, warmup, sampling and CSV output using already compiled executables. Both exclude offline common scoring, installation and development checks. The end-to-end action/supervisor times are 176.346221 and 118.2798954169848 seconds respectively. Stan's collector additionally records 118.20119466699543 seconds for compile-plus-fit components; this is not the supervisor clock. Its earlier Pathfinder executable build cost 53.24115104202065 seconds, disclosed separately. ESS/s divides the minimum bulk ESS by the reported fit duration. These are not controlled, hardware-independent speed measurements. [comparison.json](comparison.json) retains each clock separately.

## Selection and campaign scope

The selected Stan run minimizes worst physical-coordinate R-hat among completed fixed-draw runs, without using closeness to the numerical reference. It remains the best by that rule after the later runs. The particle configuration is the saved M=2, N=256, eight-chain benchmark choice; the original selection artifact describes it as fixed. The adaptive campaign and earlier particle exploration are not a preregistered comparison. See the selection provenance in [comparison.json](comparison.json).

The compact campaign ledger contains all 20 valid fixed-draw Stan arms and three time-budget arms. None passes the combined completion/budget and mixing criteria. It covers centered and non-centered paths, whitening, dense metrics, longer warmup, acceptance/tree-depth changes, Pathfinder starts, exact threshold marginalization, exact sign summation, blocked innovations and additional threshold coordinates. Deadline termination is permitted for time-budget arms; their requested draw counts are ceilings. Incomplete chains use equal-length initial retained prefixes for diagnostics. A dense-metric arm with no retained draws is a budget-incomplete attempt, not evidence that a fully warmed dense metric cannot work. Excluded executable crashes and timing-contaminated attempts are recorded separately. See [comparison.json](comparison.json).

This is one dataset and one sampling seed per configuration, with adaptive rescue choices. GPU sampling uses float32 on NVIDIA L40S; Stan uses float64 on Apple M1 Max (10 CPU cores, 64 GB RAM). The results establish a practical separation in this campaign, not universal NUTS failure or performance across datasets.

## Archived inputs and integrity

- [model.stan](model.stan) and [data.json](data.json): byte-for-byte selected Stan inputs. The Pathfinder and NUTS sources had the same recorded source hash.
- [model-request.json](model-request.json), [data-request.json](data-request.json), [observations.csv](observations.csv), [revisions.json](revisions.json) and [fit-request.json](fit-request.json): byte-for-byte saved native inputs and identifiers. The data request's `tmp/input` is the original staged source name; the CSV preserves the 360-day grid, including days without incidence observations. Study revision identifiers are historical, not portable new-study handles.
- [comparison.json](comparison.json): exact selected group summaries, complete resolved particle settings, Stan settings, source commit and runtime digest, call ID, source hashes, and compact campaign summaries. Original scratchpad identifiers are provenance, not dependencies required to read this archive.

Archive checks confirmed exact copies and hashes, equality of all 52 Stan/native observations and 360 policy inputs, the 20+3 campaign count, and both displayed ESS/s calculations. Raw posterior draws, binary executables, figures and full logs are omitted. No experiment was rerun to create this archive.
