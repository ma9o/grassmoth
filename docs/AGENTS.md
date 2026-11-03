# Documentation guidance

- Place references beside the claims they support or hyperlink the relevant terms.

- Each fact has one maintained owner: the [action charts](assets/action-flows) own control flow, code owns field meanings and the API, and [model assumptions](assumptions.md) own modeling commitments and limits. Link to the owner instead of restating it.

- A change to an action's behavior updates its chart in the same commit.

- The repository README lives here as [README.md](README.md); the root `README.md` is a symlink to it. Write its links as repository-root paths such as `/docs/assumptions.md`, which resolve from both locations, and link other documents to `docs/README.md`.

## Generated documentation

- Run `bun run --cwd docs codegen` to regenerate mathematical SVGs, then `bun run --cwd docs lint` to check generated math, Markdown and spelling. [The LaTeX generator](scripts/codegen_docs_latex.js) owns the rendering; retain its `docs-latex` source metadata alongside generated embeds.
- Follow the [API generation workflow](../packages/api-types/AGENTS.md#code-generation) for generated API documentation. Edit field and endpoint descriptions at their Python owners; the [API exporter](../apps/grassmoth-server/scripts/codegen/export_api.py) generates the study API reference and agent skill.

## Action-flow charts

Each chart in [`docs/assets/action-flows`](assets/action-flows) is a native Excalidraw scene, and the scene is the source. After editing a scene, export its SVG, then render and inspect the result. Use `edit_model` as the visual reference and keep its individual check nodes.

Keep the shared styling: 16 px sans-serif body text, monospace titles and outcomes, solid 2 px outlines, rounded stacked cards, slate connectors and pale gray section panels. Requests are indigo, warnings amber, failures red and commits green. Skipped or unavailable results and subsequent actions use dashed gray cards.
