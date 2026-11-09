# Web client architecture

- Follow the [workbench design principles](#workbench-design) before changing the UI. Design in client components, never in artifacts.

- Keep domain logic and statistical computations in the backend, except for frontend visualization: evaluate or sample saved probability laws, evaluate saved model expressions, and transform aligned draws. Preserve joint dependence and declared parameter transforms. These display evaluations never feed inference, trajectory generation, diagnostics, predictive checks, or scientific decisions, which remain backend-owned.

- The workbench (`/workspaces/{workspaceId}`) is the only interface.

- Consume backend contracts through [the API types package](../../packages/api-types/AGENTS.md). Shared UI logic must not depend on rendering components; [ESLint](eslint.config.mjs) owns that dependency check.

## Workbench design

The workbench (`/workspaces/{workspaceId}`) shows what the agent harness did to a study and the model each action left. Breaking a principle needs the user's approval.

### Scope

- Read-only: the harness changes studies; the UI never writes.
- Use the API's vocabulary; don't coin labels.
- A feature is what the workbench shows; never remove one without approval.
- Mobile layout is not a concern for now: design and check the workbench at desktop widths.

### Panes

- The selected timeline node scopes everything on screen.
- The action record shows only the action's recorded input and ordered execution messages; the graph and details show the state it left.
- The graph is a glance at the whole model that still shows as much as it can visually, above all what helps when reading the whole, such as how data flows through the model during a simulation, with particles or curves changing over time. Depth belongs to the details. What earns a place on the graph is the user's call as we iterate.
- A mechanism reads as a term of its effect, never as a variable.
- A node shows only its own output and its recorded inputs. Its marks are its own findings.
- What the UI needs and no action produces belongs in the owning action's output.

### Evidence

- Show distributions, draws, chains and observations; numbers may label a chart, never replace it. Action summaries appear only when emitted as log messages.
- Never hide information: outliers stay visible, and aggregation is only for overlap that already hides detail.
- Cut a duplicate only if the same evidence stays visible at the same resolution.
- One visual vocabulary: the same kind of evidence always looks the same, color means a role, never a draw, and time is calendar time.

### Code

- The backend's types define the UI's parts and how they nest; the UI adds no domain concepts of its own.
- Narrow saved assessment, effect and comparison alternatives before reading their evidence or absence reason. Consume the producer's scientific classifications and inference plot series directly.

## Development data

- Develop the workbench in the normal local stack against real saved studies. Keep exports, generated snapshots and mock APIs out of development.
- Pure unit tests use small typed values in `src/lib/__fixtures__`. Synthetic unit values are rendering examples, never benchmark evidence.
