# Workbench design

The workbench (`/workspaces/{workspaceId}`) shows what the agent harness did to a study and the model each action left. Breaking a principle needs the user's approval.

## Scope

- Read-only: the harness changes studies; the UI never writes.
- Use the API's vocabulary; don't coin labels.
- A feature is what the workbench shows; never remove one without approval.

## Panes

- The selected timeline node scopes everything on screen.
- The action record states what the action did and found; the graph and details show the state it left.
- The graph is a glance at the whole model that still shows as much as it can visually, above all what helps when reading the whole, such as how data flows through the model during a simulation, with particles or curves changing over time. Depth belongs to the details. What earns a place on the graph is the user's call as we iterate.
- A mechanism reads as a term of its effect, never as a variable.
- A node shows only its own output and its recorded inputs. Its marks are its own findings.
- What the UI needs and no action produces belongs in the owning action's output.

## Evidence

- Show distributions, draws, chains and observations; numbers may label a chart, never replace it. Summaries belong in the action record.
- Never hide information: outliers stay visible, and aggregation is only for overlap that already hides detail.
- Cut a duplicate only if the same evidence stays visible at the same resolution.
- One visual vocabulary: the same kind of evidence always looks the same, color means a role, never a draw, and time is calendar time.

## Code

- The backend's types define the UI's parts and how they nest; the UI adds no domain concepts of its own.
- Storybook shows the workbench and its panes exactly as the workbench renders them, in states the backend produced.
