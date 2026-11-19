import type {
  CoefficientExpression,
  ConstructId,
  DynamicalModelSpec,
  Expression,
  SimulationReport,
} from "@grassmoth/api-types";
import { dump } from "npyjs";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { LayeredCausalGraph } from "@/components/dag/layered/layered-causal-graph";
import {
  baseline as baselineConstruct,
  treatment as dose,
  modelFixture,
  outcome as symptoms,
} from "@/lib/__fixtures__/model";
import { authoredSnapshot, fittedSnapshot } from "@/lib/__fixtures__/snapshot";
import { buildLayeredCausalGraph, GRAPH_DENSITIES } from "@/lib/dag/build-layered-causal-graph";
import { specLayout } from "@/lib/dag/spec-layout";
import { presentEntries } from "@/lib/model-accessors";
import { indexModel } from "@/lib/model-asset/entities";
import type { SimulationPathsView } from "@/lib/model-asset/result-values";
import { modelSpecPlots } from "@/lib/model-asset/spec-plots";
import type { DagGraphInput } from "@/lib/utils/dag-graph-layout";
import { formatModelDate } from "@/lib/utils/format";

// Only the asynchronous layout is stubbed; render the real graph and its overlays.
vi.mock("@/lib/hooks/use-dag-layout", () => ({
  useDagLayout: (graph: DagGraphInput | null) => ({
    width: 800,
    height: 400,
    isLayouting: false,
    nodes: (graph?.nodes ?? []).map((node, index) => ({ ...node, x: index * 100, y: 40 })),
    edges: (graph?.edges ?? []).map((edge) => ({
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

function render(simulation: SimulationReport | null) {
  return renderToStaticMarkup(
    createElement(LayeredCausalGraph, {
      modelSnapshot,
      entities: indexed,
      simulation,
      simulationPaths: simulation ? paths : null,
      selection: null,
      onSelect: () => undefined,
    }),
  );
}

function arrows(markup: string) {
  return markup.match(/<path\b[^>]*marker-end=[^>]*>/g) ?? [];
}

function selectableGraphLabels(markup: string) {
  return [...markup.matchAll(/<g\b(?=[^>]*role="button")[^>]*aria-label="([^"]+)"/g)].map(
    (match) => match[1],
  );
}

describe("dated intervention overlay", () => {
  it("preserves the model's arrows and their styling", () => {
    const baseline = arrows(render(null));
    // Two edges, and the carry arrow of each of the two time-varying constructs.
    expect(baseline).toHaveLength(4);
    // An edge is instantaneous and solid; only a carry arrow leaves its time slice, dashed.
    expect(baseline.filter((arrow) => arrow.includes("stroke-dasharray"))).toHaveLength(2);
    expect(arrows(render(report))).toEqual(baseline);
  });

  it("shows every dated assignment without implying a continuously fixed value", () => {
    const markup = render(report);
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

describe("authored mechanism boxes", () => {
  it("makes the enclosing node the only target in an edit with retained fitted laws", () => {
    const markup = renderToStaticMarkup(
      createElement(LayeredCausalGraph, {
        modelSnapshot: fittedSnapshot,
        entities: indexed,
        step: "edit_model",
        selection: null,
        onSelect: () => undefined,
      }),
    );
    expect(selectableGraphLabels(markup).toSorted()).toEqual(
      indexed.constructs.map((construct) => construct.name).toSorted(),
    );
  });

  it("places one shared known input above its first effect and keeps every reading below its state", () => {
    const source = modelFixture.constructs[baselineConstruct.id];
    const state = modelFixture.constructs[dose.id];
    const effect = modelFixture.constructs[symptoms.id];
    const reading = state?.indicators["indicator:00000000000000000001"];
    if (!source || !state || !effect || !reading) throw new Error("Missing layout fixture.");
    const input = { kind: "state" as const, construct_id: baselineConstruct.id };
    const name = "a long indicator name that spans several lines";
    const spec: DynamicalModelSpec = {
      ...modelFixture,
      constructs: {
        [baselineConstruct.id]: {
          ...source,
          role: "exogenous",
          indicators: { "indicator:known": reading },
        },
        [dose.id]: {
          ...state,
          dynamics: { "mechanism:first": { kind: "drift", expression: input } },
          indicators: Object.fromEntries(
            ["one", "two", "three"].map((id) => [
              `indicator:${id}`,
              { ...reading, observation: { ...reading.observation, name } },
            ]),
          ),
        },
        [symptoms.id]: {
          ...effect,
          dynamics: { "mechanism:second": { kind: "drift", expression: input } },
        },
      },
    };
    const { constructs, edges } = indexModel(spec);
    const bundle = buildLayeredCausalGraph(constructs, edges, symptoms.id, GRAPH_DENSITIES.rich);
    const graph = specLayout(bundle, modelSpecPlots(spec), edges);
    const known = graph.nodes.filter((n) => n.id === baselineConstruct.id);
    const first = graph.nodes.find((n) => n.id === dose.id);
    if (!first || !known[0]) throw new Error("Missing state or known input.");
    expect(known).toHaveLength(1);
    expect(known[0].x).toBe(first.x);
    expect(known[0].y + known[0].height).toBeLessThan(first.y);
    expect(
      graph.links.filter((l) => l.source === baselineConstruct.id).map((l) => l.target),
    ).toEqual(["mechanism:first", "mechanism:second"]);
    expect(graph.indicators.has(baselineConstruct.id)).toBe(false);
    const boxes = graph.indicators.get(dose.id) ?? [];
    expect(boxes).toHaveLength(3);
    let bottom = first.height;
    for (const box of boxes) {
      expect(box.y).toBeGreaterThan(bottom);
      expect(box.x).toBeGreaterThan(0);
      expect(box.width).toBeLessThan(first.width);
      expect(box.name.join(" ")).toBe(name);
      const fn = graph.functions.find((f) => f.glyph === box.response);
      if (!fn) throw new Error("Missing reading function.");
      expect(fn.y).toBeGreaterThan(first.y + first.height);
      expect(fn.x + fn.width).toBeLessThan(first.x + box.x);
      expect(graph.links.filter((l) => l.source === fn.id).map((l) => l.target)).toEqual([
        box.distribution.selection.id,
      ]);
      bottom = box.y + box.height;
    }
  });

  it("connects all read states to one function and draws one output into the effect", () => {
    const initial: readonly CoefficientExpression[] = [
      { kind: "coefficient", role: "initial_mean", value: 0, construct_ids: [] },
      { kind: "coefficient", role: "initial_scale", value: 1, construct_ids: [] },
    ];
    const read = (id: ConstructId): Expression => ({ kind: "state", construct_id: id });
    const expression: Expression = {
      kind: "binary",
      operator: "add",
      left: read(baselineConstruct.id),
      right: {
        kind: "binary",
        operator: "add",
        left: read(dose.id),
        right: read(symptoms.id),
      },
    };
    const spec: DynamicalModelSpec = {
      ...modelFixture,
      constructs: Object.fromEntries(
        presentEntries(modelFixture.constructs).map(([id, construct]) => [
          id,
          {
            ...construct,
            coefficients: initial,
            indicators: {},
            dynamics:
              id === symptoms.id ? { "mechanism:combined": { kind: "drift", expression } } : {},
          },
        ]),
      ),
    };
    const markup = renderToStaticMarkup(
      createElement(LayeredCausalGraph, {
        modelSnapshot: { ...authoredSnapshot, dynamical_model_spec: spec },
        entities: indexModel(spec),
        step: "edit_model",
        selection: null,
        onSelect: () => undefined,
      }),
    );
    for (const id of [baselineConstruct.id, dose.id, symptoms.id]) {
      expect(markup).toContain(`data-source="${id}" data-target="mechanism:combined"`);
    }
    expect(selectableGraphLabels(markup).toSorted()).toEqual(
      [
        ...indexModel(spec).constructs.map((construct) => construct.name),
        "Mechanism combined",
      ].toSorted(),
    );
    expect(markup.match(/data-source="mechanism:combined"/g)).toHaveLength(1);
    expect(markup).toContain(`data-source="mechanism:combined" data-target="${symptoms.id}"`);
    expect(markup.match(/data-function-id="mechanism:combined"/g)).toHaveLength(1);
    expect(markup.match(/aria-label="Function of [^"]+"/g)).toEqual([
      'aria-label="Function of baseline"',
      'aria-label="Function of treatment"',
      'aria-label="Function of outcome"',
    ]);
  });
});
