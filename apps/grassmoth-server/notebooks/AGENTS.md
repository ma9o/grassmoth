# Notebook guidance

- This directory contains marimo Python notebooks. For scientific notebook work, follow the [experiment guidance](../../../benchmarks/AGENTS.md), including its public-library checks and scientific model requirements.
- Name every top-level `@app.cell` function with a descriptive snake_case name. Keep inner helpers `_`-prefixed.
- From `apps/grassmoth-server`, validate edits with `uv run marimo check --strict notebooks/<name>.py`.
- Launch the editor from the repo root with `bun run --cwd apps/grassmoth-server notebooks`.
- For `marimo-pair`, open a specific notebook in the browser first; the file browser exposes no session. Attach with `execute-code.sh --url http://localhost:2718 --token <access_token>`, using the startup banner's token. Pass it via `--token` or `MARIMO_TOKEN`, never in the `--url` query string.
