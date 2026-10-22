import { modelResult } from "@/components/__fixtures__/action-results";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { authoredSnapshot } from "@/lib/__fixtures__/snapshot";
import { indexModel } from "@/lib/model-asset/entities";
import type { Applied, ActionSuccess, TimelineRevision } from "@grassmoth/api-types";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { EditDetails } from "@/components/model/scopes/edit-details";
import { ActionFindings } from "@/components/model/action-findings";
import { humanize } from "@/lib/model-asset/selection";

const hooks = vi.hoisted(() => ({ diff: vi.fn(), snapshot: vi.fn() }));
vi.mock("@/lib/hooks/use-model-snapshot", () => ({ useModelSnapshot: hooks.snapshot }));
vi.mock("@/lib/hooks/use-model-diff", () => ({ useModelDiff: hooks.diff }));

const context: ScopeContext = {
  ticks: [],
  dataDiff: null,
  result: undefined,
  modelSnapshot: authoredSnapshot,
  entities: indexModel(authoredSnapshot.dynamical_model_spec),
  select: vi.fn(),
};
const applied: Applied<null> = {
  status: "applied",
  result: null,
  effects: {
    retracted: [],
    reports: {},
    produced: [
      {
        artifact_id: "model",
        revision: "edited-model",
        derived_from: { model: "archived-authorship-base" },
        produced_by: "edit_model",
        created_at: "2026-09-30T00:00:00Z",
        source: { kind: "files" },
      },
    ],
  },
};
const tick: TimelineRevision = {
  call_id: `call:${"2".repeat(64)}`,
  commit_id: "rewritten-edit",
  parent_ids: ["preceding-commit"],
  record: {
    seq: 5,
    ts: "2026-09-30T00:00:00Z",
    messages: [],
    trace_ids: [],
    attempt: {
      action: "edit_model",
      request: {
        action: "edit_model",
        reasoning: null,
        input: {
          parent_ref: "archived-authorship-base",
          dynamical_model_spec: fixtureValue(authoredSnapshot.dynamical_model_spec),
        },
      },
      outcome: applied,
    },
  },
};

describe("edit change summaries after history compaction", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    hooks.diff.mockReturnValue({ data: undefined, error: null });
    hooks.snapshot.mockReturnValue({ data: authoredSnapshot, error: null });
  });

  it("compares the exact authored base named by the call", () => {
    renderToStaticMarkup(createElement(EditDetails, { context, tick }));
    expect(hooks.diff).toHaveBeenCalledWith(
      context.modelSnapshot.workspace_id,
      "archived-authorship-base",
      "edited-model",
      context.ticks,
    );
  });

  it("shows an initial summary when the parent is a question", () => {
    const request = fixtureValue(tick.record.attempt.request);
    if (request.action !== "edit_model") throw new Error("Expected edit fixture");
    const created = {
      ...tick,
      record: {
        ...tick.record,
        attempt: {
          ...tick.record.attempt,
          request: {
            ...request,
            input: {
              ...request.input,
              parent_ref: fixtureValue(authoredSnapshot.state.current.question).revision,
            },
          },
          outcome: {
            ...applied,
            effects: {
              ...applied.effects,
              produced: applied.effects.produced.map((artifact) => ({
                ...artifact,
                derived_from: {
                  question: fixtureValue(authoredSnapshot.state.current.question).revision,
                },
              })),
            },
          },
        },
      },
    };
    const html = renderToStaticMarkup(createElement(EditDetails, { context, tick: created }));
    expect(hooks.diff).toHaveBeenCalledWith(
      context.modelSnapshot.workspace_id,
      null,
      null,
      context.ticks,
    );
    expect(html).toContain("Model created");
  });

  it("shows when no comparison has been recorded", () => {
    const html = renderToStaticMarkup(createElement(EditDetails, { context, tick }));
    expect(html).toContain("No saved comparison for these model versions.");
    expect(html).not.toContain("Reading model changes");
  });

  it("renders the saved spec patch and its changed values using model-owned labels", () => {
    const edge = fixtureValue(context.entities.edges[0]);
    const parameter = fixtureValue(context.entities.parameters[0]);
    hooks.diff.mockReturnValue({
      data: {
        changes: {
          constructs: {
            [edge.cause.id]: {
              name: fixtureValue(context.entities.constructById.get(edge.cause.id)).name,
            },
          },
          edges: { [edge.id]: { description: "Changed causal assumption" } },
          parameters: { [parameter.id]: { reasoning: "Changed prior justification" } },
          measurement_clock: null,
        },
      },
      error: null,
    });
    const html = renderToStaticMarkup(createElement(EditDetails, { context, tick }));
    expect(html).toContain(
      humanize(fixtureValue(context.entities.constructById.get(edge.cause.id)).name),
    );
    expect(html).toContain(humanize(parameter.name));
    expect(html).toContain("Changed causal assumption");
    expect(html).toContain("Changed prior justification");
    expect(html).toContain("Measurement clock");
    expect(html).not.toContain("compilation, belief");
  });

  it("renders an empty patch as no spec changes", () => {
    hooks.diff.mockReturnValue({ data: { changes: {} }, error: null });
    const html = renderToStaticMarkup(createElement(EditDetails, { context, tick }));
    expect(html).toContain("No spec changes.");
  });

  it("shows served check reasons unchanged and omits passing checks", () => {
    const indicator = fixtureValue(context.entities.indicators[0]);
    const served: ActionSuccess = {
      status: "success",
      call_id: `call:${"2".repeat(64)}`,
      action: "edit_model",
      commit_id: tick.commit_id,
      messages: [],
      body: {
        ...modelResult(context.modelSnapshot),
        checks: {
          question: { findings: [] },
          specification: [
            {
              code: "model_execution",
              kind: "evaluated",
              subject: "model_execution",
              outcome: "failed",
              evidence: `${indicator.observation.id} requires a likelihood.`,
            },
            {
              code: "passed_check",
              kind: "evaluated",
              subject: "passed_check",
              outcome: "passed",
              evidence: "Do not list passing checks.",
            },
          ],
        },
      },
    };
    const html = renderToStaticMarkup(
      createElement(ActionFindings, {
        context: { ...context, result: served },
        call: served,
      }),
    );
    expect(html).toContain(indicator.observation.id);
    expect(html).toContain("requires a likelihood.");
    expect(html).not.toContain("Do not list passing checks.");
  });
});
