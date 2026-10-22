import { dump } from "npyjs";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { SimulationReport } from "@grassmoth/api-types";
import { treatment as dose, outcome as symptoms } from "@/lib/__fixtures__/model";
import { fittedSnapshot } from "@/lib/__fixtures__/snapshot";
import type { SimulationPathsView } from "@/lib/model-asset/result-values";
import { indexModel } from "@/lib/model-asset/entities";
import type { DagGraphInput } from "@/lib/utils/dag-graph-layout";
import { formatModelDate } from "@/lib/utils/format";
import {
  LayeredCausalGraph,
  type LayeredCausalGraphVariant,
} from "@/components/dag/layered/layered-causal-graph";

// Only the asynchronous layout is stubbed; render the real graph and its overlays.
vi.mock("@/lib/hooks/use-dag-layout", () => ({
  useDagLayout: (graph: DagGraphInput) => ({
    width: 800,
    height: 400,
    isLayouting: false,
    nodes: graph.nodes.map((node, index) => ({ ...node, x: index * 100, y: 40 })),
    edges: graph.edges.map((edge) => ({
      ...edge,
      points: [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
      ],
    })),
  }),
}));

const base = fittedSnapshot;
const indexed = indexModel(base.dynamical_model_spec);
const modelSnapshot = base;
const buffer = { npy: new Uint8Array(dump([], [0], { dtype: "f8" })) };

const report: SimulationReport = {
  evidence: {
    draws: 1,
    seed: 0,
    state_ids: [dose.id, symptoms.id],
    parameter_draws: {},
    arms: {
      kind: "paired",
      causal: {
        kind: "not_evaluated",
        code: "causal_effect",
        subject: "causal_effect",
        reason: "CAUSAL_EVALUATION_FAILED",
        detail: "This fixture covers intervention rendering.",
      },
      action: { latent_paths: buffer, observations: buffer },
      reference: { latent_paths: buffer, observations: buffer },
    },
    observation_layout: {
      variables: [],
      support_start_times: buffer,
      support_end_times: buffer,
      mask: buffer,
    },
    time_origin: "2026-01-01T00:00:00Z",
    times: [0, 1, 2, 3, 4, 5, 6, 7],
    assignments: [
      { target: dose.id, time: 1, value: 10 },
      { target: dose.id, time: 3, value: 5 },
      { target: dose.id, time: 6, value: 0 },
    ],
  },
  summary: {
    state_frames: {},
    indicator_frames: {},
    action_category_probabilities: {},
    reference_category_probabilities: {},
  },
  law: { kind: "authored", interpretation: "prior_predictive" },
  findings: [],
  fit_reliability: "not_fitted",
};
const paths: SimulationPathsView = {
  action_category_probabilities: {},
  reference_category_probabilities: {},
  times: report.evidence.times,
  time_origin: report.evidence.time_origin,
  total_draws: 1,
  start: 0,
  count: 1,
  states: {
    [dose.id]: {
      label: dose.name,
      levels: null,
      action: [{ draw: 0, values: [9, 10, 9, 5, 6, 4, 0, 1] }],
      reference: [],
      frame: [0, 10],
    },
  },
  indicators: {},
  effect: null,
};

function render(variant: LayeredCausalGraphVariant, simulation: SimulationReport | null) {
  return renderToStaticMarkup(
    createElement(LayeredCausalGraph, {
      modelSnapshot,
      entities: indexed,
      simulation,
      simulationPaths: simulation ? paths : null,
      selection: null,
      onSelect: () => {},
      variant,
    }),
  );
}

describe("dated intervention overlay", () => {
  it.each([
    "asset",
    "workbench",
  ] as const)("preserves the model's arrows and their styling in the %s view", (variant) => {
    const arrows = (markup: string) => markup.match(/<path\b[^>]*marker-end=[^>]*>/g) ?? [];
    const baseline = arrows(render(variant, null));
    expect(baseline).toHaveLength(4);
    expect(arrows(render(variant, report))).toEqual(baseline);
  });

  it("shows every dated assignment without implying a continuously fixed value", () => {
    const markup = render("asset", report);
    expect(markup.match(/>3 assignments</g)).toHaveLength(1);
    for (const event of report.evidence.assignments) {
      expect(markup).toContain(
        `aria-label="${formatModelDate(event.time, report.evidence.time_origin)} (day ${event.time}): set to ${event.value}"`,
      );
    }
    expect(markup).not.toContain("do(");
    expect(markup).not.toContain("Cut by intervention");
  });
});
