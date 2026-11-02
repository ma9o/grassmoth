# Web client architecture

- Read the [workbench design principles](../../docs/design.md) before changing the UI. Design in client components and their stories, never in artifacts.

- Keep domain logic and statistical computations in the backend, except for frontend visualization: evaluate or sample saved probability laws, evaluate saved model expressions, and transform aligned draws. Preserve joint dependence and declared parameter transforms. These display evaluations never feed inference, trajectory generation, diagnostics, predictive checks, or scientific decisions, which remain backend-owned.

- The workbench (`/workspaces/{workspaceId}`) is the only interface.

- Consume backend contracts through [the API types package](../../packages/api-types/AGENTS.md). Shared UI logic must not depend on rendering components; [ESLint](eslint.config.mjs) owns that dependency check.
