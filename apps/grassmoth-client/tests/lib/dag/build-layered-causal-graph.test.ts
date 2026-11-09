import type { CausalEdgeSpec, ConstructSpec } from "@grassmoth/api-types";
import { describe, expect, it } from "vitest";
import { baseline, modelFixture, outcome, treatment } from "@/lib/__fixtures__/model";
import { buildLayeredCausalGraph, GRAPH_DENSITIES } from "@/lib/dag/build-layered-causal-graph";
import { placeComparisonOverlay } from "@/lib/dag/comparison-overlay";
import { modelEdges } from "@/lib/model-accessors";

const response: CausalEdgeSpec["mechanisms"][number] = {
  id: "mechanism:00000000000000000009",
  kind: "drift",
  expression: {
    kind: "binary",
    operator: "multiply",
    left: { kind: "state", construct_id: treatment.id },
    right: { kind: "state", construct_id: outcome.id },
  },
};
const constructs: ConstructSpec[] = [{ ...baseline, role: "exogenous" }, treatment, outcome];
const edges = modelEdges(modelFixture).map((edge) =>
  edge.effect.id === outcome.id ? { ...edge, mechanisms: [response] } : edge,
);
const build = (items = constructs) =>
  buildLayeredCausalGraph(items, edges, outcome.id, GRAPH_DENSITIES.rich);

describe("buildLayeredCausalGraph", () => {
  it("draws mechanisms as rows of the construct they drive, each edge ending on its reader", () => {
    const built = build();
    expect(built.graph.nodes.map((node) => node.id)).toEqual([
      baseline.id,
      treatment.id,
      outcome.id,
    ]);
    expect(built.nodeMeta.get(baseline.id)?.kind).toBe("exogenous");
    const card = built.nodeMeta.get(outcome.id);
    if (card?.kind !== "construct") throw new Error("The outcome is drawn as a card.");
    expect(card.rows.map((row) => [row.mechanism.id, row.owner.kind])).toEqual([
      [outcome.dynamics[0]?.id, "construct"],
      [response.id, "edge"],
    ]);
    expect(built.graph.edges).toEqual([
      expect.objectContaining({ source: baseline.id, target: treatment.id }),
      expect.objectContaining({ source: treatment.id, target: `${response.id}#in` }),
    ]);
    expect([...built.arrowMeta.values()].map(({ row }) => row?.mechanism.id ?? null)).toEqual([
      null,
      response.id,
    ]);
  });

  it("preserves topology and graph identity when every display name changes", () => {
    const before = build();
    const after = build(constructs.map((item, index) => ({ ...item, name: `renamed ${index}` })));
    expect(after.graph).toEqual(before.graph);
    const identities = (bundle: typeof before) =>
      [...bundle.arrowMeta].map(([id, { edge, cause, effect, row }]) => [
        id,
        edge.id,
        cause.id,
        effect.id,
        row?.mechanism.id ?? null,
      ]);
    expect(identities(after)).toEqual(identities(before));
  });
});

it("marks spec revisions without presenting a rename as a topology change", () => {
  const built = build();
  const definition = modelFixture.constructs[treatment.id];
  if (!definition) throw new Error("Missing fixture construct");
  const after = {
    ...modelFixture,
    constructs: { ...modelFixture.constructs, [treatment.id]: { ...definition, name: "Exposure" } },
  };
  const overlay = placeComparisonOverlay(
    {
      changes: { constructs: { [treatment.id]: { name: "Exposure" } } },
      beforeDynamicalModelSpec: modelFixture,
      afterDynamicalModelSpec: after,
    },
    built,
    {
      nodes: [{ id: treatment.id, x: 0, y: 0, width: 100, height: 60 }],
      edges: [],
      width: 200,
      height: 100,
    },
  );
  expect(overlay.constructChanges.get(treatment.id)).toBe("revised");
  expect(overlay.addedNodes).toEqual([]);
  expect(overlay.addedEdges).toEqual([]);
  expect(overlay.marks).toEqual([
    expect.objectContaining({ title: "revised construct", detail: "Exposure" }),
  ]);
});
