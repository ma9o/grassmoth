# Documentation guidance

- Place references beside the claims they support or hyperlink the relevant terms.

- Each fact has one maintained owner: the [action charts](assets/action-flows) own control flow, code owns field meanings and the API, and [model assumptions](assumptions.md) own modeling commitments and limits. Link to the owner instead of restating it.

- A change to an action's behavior updates its chart in the same commit.

## Action-flow charts

Each chart in [`docs/assets/action-flows`](assets/action-flows) is a native Excalidraw scene, and the scene is the source. After editing a scene, export its SVG, then render and inspect the result. Use `edit_model` as the visual reference and keep its individual check nodes.

Keep the shared styling: 16 px sans-serif body text, monospace titles and outcomes, solid 2 px outlines, rounded stacked cards, slate connectors and pale gray section panels. Requests are indigo, warnings amber, failures red and commits green. Skipped or unavailable results and subsequent actions use dashed gray cards.
