import { modelComparison } from "@/lib/__fixtures__/action-results";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { authoredSnapshot } from "@/lib/__fixtures__/snapshot";
import { indexModel } from "@/lib/model-asset/entities";
import type { TimelineRevision } from "@grassmoth/api-types";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { ModelComparisonDetails } from "@/components/model/scopes/edit-details";
import { humanize } from "@/lib/model-asset/selection";

const hooks = vi.hoisted(() => ({ snapshot: vi.fn() }));
vi.mock("@/lib/hooks/use-model-snapshot", () => ({ useModelSnapshot: hooks.snapshot }));

const tick: TimelineRevision = {
  call_id: modelComparison.call_id,
  commit_id: modelComparison.commit_id,
  parent_ids: ["preceding-commit"],
  record: {
    seq: 5,
    ts: "2026-09-30T00:00:00Z",
    attempt: {
      action: "model_diff",
      request: {
        action: "model_diff",
        reasoning: null,
        input: {
          before_ref: "archived-authorship-base",
          after_ref: "edited-model",
        },
      },
      outcome: {
        status: "applied",
        result: null,
        effects: { produced: [], retracted: [], reports: {} },
      },
    },
  },
};
const context: ScopeContext = {
  ticks: [tick],
  dataDiff: null,
  result: modelComparison,
  modelSnapshot: authoredSnapshot,
  entities: indexModel(authoredSnapshot.dynamical_model_spec),
  select: vi.fn(),
};

describe("saved model comparison details", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    hooks.snapshot.mockReturnValue({ data: authoredSnapshot, error: null });
  });

  it("renders the saved spec patch against its recorded base using model-owned labels", () => {
    const edge = fixtureValue(context.entities.edges[0]);
    const parameter = fixtureValue(context.entities.parameters[0]);
    const html = renderToStaticMarkup(
      createElement(ModelComparisonDetails, {
        context: {
          ...context,
          result: {
            ...modelComparison,
            body: {
              changes: {
                constructs: {
                  [edge.cause.id]: {
                    name: fixtureValue(context.entities.constructById.get(edge.cause.id)).name,
                  },
                },
                edges: {
                  [edge.id]: {
                    reasoning: { text: "Changed causal assumption", reference_urls: [] },
                  },
                },
                parameters: {
                  [parameter.id]: {
                    reasoning: { text: "Changed prior justification", reference_urls: [] },
                  },
                },
                measurement_clock: null,
              },
            },
          },
        },
      }),
    );
    expect(hooks.snapshot).toHaveBeenCalledWith(
      context.modelSnapshot.workspace_id,
      "archived-authorship-base",
      true,
    );
    expect(html).toContain(
      humanize(fixtureValue(context.entities.constructById.get(edge.cause.id)).name),
    );
    expect(html).toContain(humanize(parameter.name));
    expect(html).toContain("Changed causal assumption");
    expect(html).toContain("Changed prior justification");
    expect(html).toContain("Measurement clock");
  });

  it("renders an empty patch as no spec changes", () => {
    const html = renderToStaticMarkup(
      createElement(ModelComparisonDetails, {
        context: { ...context, result: { ...modelComparison, body: { changes: {} } } },
      }),
    );
    expect(html).toContain("No spec changes.");
  });
});
