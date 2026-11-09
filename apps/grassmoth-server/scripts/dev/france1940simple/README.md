# France development study

Rebuild the real study through the public HTTP API when saved results no longer
match the current contracts. Start the local stack using the
[integration guide](../../../../../docs/guides/agentic_integration_testing.md), then
run from the repository root:

```bash
bun run --cwd apps/grassmoth-server study:rebuild
```

The command prints the new workspace ID and workbench URL. Each default run uses
a fresh timestamped `FRANCE1940SIMPLE_…` workspace. It uploads the source CSV and
runs `edit_model` twice, then `prepare_data`, `simulate`, `data_diff`, and
`model_diff`, waiting for each saved result. It never calls `fit`. The simulation
retains the original 100 prior histories, seed 0, and daily 1936–1940 window.
`model_diff` compares the initial outcome-only model with the authored model.

The script pins dependencies to the produced artifact revisions returned by the
timeline. Simulated data uses the simulation call's commit. The `latest` values
in the authored inputs are replaced with these exact revisions before submission.
All scientific work and result storage belong to the service.

To resume after interruption, use the ID printed at startup:

```bash
bun run --cwd apps/grassmoth-server study:rebuild --workspace FRANCE1940SIMPLE_20261109T100000000000Z
```

The API reuses identical calls, including saved failures. A timeout stops this
client's polling; the server continues the call. Fix a failed request's inputs or
start a fresh workspace. After a contract change, start a fresh rebuild; this
command neither migrates old histories nor edits stored results. The API has no
reset endpoint, so the script does not replace an existing workspace.

`--api-url`, `--web-url`, and `--timeout` select another local stack and the polling
limit in seconds per action. `--check` parses the authored requests against the
current contracts without contacting the server:

```bash
bun run --cwd apps/grassmoth-server study:rebuild --check
```

The recipe consists of authored inputs, not captured API responses. Update these
files when their owning contracts change; parsing fails rather than translating
obsolete fields. [historical_inputs.csv](historical_inputs.csv) preserves the
selected source values. [selected_rows.csv](selected_rows.csv) and
[hours_regime_events.csv](hours_regime_events.csv) retain their provenance. The
model's assumptions, priors and citations remain in [02-edit_model.json](02-edit_model.json).

This study supports development of the non-fit workbench. Its prior comparisons
are scientific findings, including warnings and failed checks; successful action
execution does not mean those checks passed. Posterior views require a later fit.
