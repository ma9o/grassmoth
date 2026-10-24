# Suite records — 2026-10-24

## Final first scorecard

**The policy does not pass the suite: two native passes, three convergence failures, two infrastructure errors without saved statistical verdicts.** All seven valid fixtures were attempted at the unchanged full schedule on separate A10s, with C0 first, reservations and a passing HD timing forecast. [Complete scorecard](../../scratchpad/suite-scorecard-20261024/findings.md) contains worst latent ESS/second, R̂, every parameter's ESS, truth coverage, frozen-coordinate checks, phase timings, costs and failure localization.

| Case | Native verdict / disposition | Fit-app cost |
|---|---|---:|
| C0 | passed | $0.43489352 |
| NL | passed | $0.43421398 |
| MODE | failed | $0.07927063 |
| HD | infrastructure error; no verdict | $0.70628578 |
| OBS | failed | $0.49272463 |
| PRIOR | infrastructure error; no verdict | $1.18228225 |
| PARAM | failed | $0.44913184 |

Total **$3.86216469 / $10**, including the HD timing probe, first MODE pre-sampling refusal, automatic PRIOR/HD repeats and all Modal CPU work. [Billing reconciliation](../../scratchpad/suite-scorecard-20261024/cost-reconciliation.json). All suite jobs are stopped. PRIOR and HD completed sampling before SIGKILL/137 during result handling; their automatic replacements were stopped to protect reservations. They are not statistical convergence failures. [Infrastructure evidence](../../scratchpad/suite-scorecard-20261024/PRIOR-HD-infrastructure.md).

NL's bounded-slope drift and MODE's exact-EM conditioned-history fixture are valid; their [redesign record](../../scratchpad/nl-mode-redesign-20261024/findings.md) supersedes the rejected initial candidates below. MODE's corrected prepared data retains one null y row at an existing anchor, with no added observation. The selected requests and revisions are in [the manifest](fixture-manifest.json).

Numerical analysis and decoding ran on Modal CPU; subsequent saved-result analyses use one Volume and cached data. New uploads are serialized and paced to ≤2 MB/s. No sampler/library changes or commits. Local tests/lint were deferred at the user's request. The historical record below describes the initial preparation stage only and is superseded by this final record.

## Initial fixture record (before scorecard and redesign)

Fixture preparation only, on master `7e7062e8b85abc76943fefb56308766ef80cc428` in study `SUITE_20261024`. **No scorecard or HD timing request has run.** C0, HD, OBS, PRIOR and PARAM are fixture-valid. **NL is invalid; MODE multimodality is untested. The suite is incomplete.** C7/C8 confirm policy refusals on CPU. Incremental Modal charge: **$0** ([before](billing-before.json), [after](billing-after.json)). No sampler/library changes or commits.

The [full validity report](../../scratchpad/suite-build-20261024/findings.md), [pinned protocol](fixture-protocol.json) and [manifest of revisions/call IDs](fixture-manifest.json) are the review record. All declarations are in [cases/suite](../cases/suite); C0 remains in [stage00](../cases/stage00). NL's saved root documents are rejected q=100 declarations, not a valid selected case.

| Case | Pre-GPU evidence | Disposition |
| --- | --- | --- |
| C0 | Native model/finite simulation; identical NL probe control has zero times above 1 on truth and all 9 prior paths | Fixture valid; first full-schedule request unrun |
| NL | q=100,300,1000,3000 all produce non-finite native simulated paths; seeds and bar unchanged | Invalid, no request; core coverage blocker |
| MODE | 22/128 prior switch events, p=0.171875, Wilson 95% [0.116331,0.246540]; full posterior separated mass not established | Untested, no fit data/request |
| HD | Eight stochastic states retained; z3/z6 channels have zero readings, six others have 2,049 each | Fixture valid; timing required before full fit |
| OBS | 2,049 grid points; each x channel 1,793 readings, y 57; days [6,8) unobserved; active x curvature 631.02–775.22, y 6.25 | Fixture valid |
| PRIOR | 4,097 point times; positive sigma_y=0.0225; stochastic readings unchanged at 2,049 times | Fixture valid |
| PARAM | Native anchors/redundancy pass; independent aligned x indicators; all six priors pinned | Fixture valid, not a general identifiability proof |
| C7 | Public local zero-sweep fit refuses interval observations | Refused by policy |
| C8 | Corrected retained deterministic stages; public local zero-sweep fit refuses recurrence replay | Refused by policy |

HD uses declared potential indicators with every hidden reading masked; deleting those indicators caused native projection refusal. PRIOR uses exact known-input half-step timestamps while retaining the 675s reading clock. C8 places the study outcome at the final deterministic stage so authoring does not prune it. These are declaration/data changes, not framework workarounds in inference.

Requests: [C0](fit-C0-twisted-k1.json), [HD](fit-HD-twisted-k1.json), [OBS](fit-OBS-twisted-k1.json), [PRIOR](fit-PRIOR-twisted-k1.json), [PARAM](fit-PARAM-twisted-k1.json). Every full request pins the [approved policy and schedule](../README.md#policy), seed 19, A10, 3,600s and both supported movement/identity traces. All request bodies validate against the published OpenAPI schema.

[HD timing only](timing-HD-twisted-k1.json): 50 warmup/100 draws, 4 chains/128 particles, A10, 600s, diagnostics enabled. It is unrun and has no scorecard standing.

Review before spend. Run C0 first; stop GPU work if it fails. Then run valid independent cases on separate A10s, with HD full fit after timing. Reserve actual spend plus outstanding jobs, overhead and retries against $10 and $2/fit. Five $2 reservations plus timing already exceed the batch allowance: use waves, never a reduced scorecard schedule. Do not label the resulting partial coverage a completed suite while NL remains invalid.

Validation: root lint passed (7/7 tasks), scratch public-boundary lint passed, native fixture actions and saved-data assertions passed for the five valid cases. C7/C8 were CPU refusal checks only. The browser viewer remained at “Reading the journal…” and browser captures timed out; UI rendering is unverified. The full report records the safe stale-stack restart, persisted Temporal recovery and all failed draft attempts.
