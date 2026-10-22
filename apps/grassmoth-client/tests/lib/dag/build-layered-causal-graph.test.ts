import { placeComparisonOverlay } from "@/lib/dag/comparison-overlay";
import { modelEdges } from "@/lib/model-accessors";
import { describe, expect, it } from "vitest";
import { baseline, modelFixture, outcome, treatment } from "@/lib/__fixtures__/model";
import { buildLayeredCausalGraph } from "@/lib/dag/build-layered-causal-graph";

const constructs = [baseline, treatment, outcome];
const edges = modelEdges(modelFixture);
const dynamicConstructIds = [treatment.id, outcome.id];

describe("buildLayeredCausalGraph", () => {
  it("keeps authored edges and persistence in distinct temporal slots", () => {
    const built = buildLayeredCausalGraph(constructs, edges, dynamicConstructIds);
    const varying = new Set(dynamicConstructIds);
    for (const id of varying) expect(built.edgeMeta.has(`self:${id}`)).toBe(true);
    for (const edge of edges) {
      expect(built.edgeMeta.get(edge.id)).toMatchObject({
        source: varying.has(edge.cause.id) ? `${edge.cause.id}__p` : edge.cause.id,
        target: edge.effect.id,
        isSelf: false,
      });
    }
    expect(built.edgeMeta.get(`self:${outcome.id}`)).toMatchObject({
      source: `${outcome.id}__p`,
      target: outcome.id,
      crossSlice: true,
      isSelf: true,
    });
  });

  it("preserves topology and graph identity when every display name changes", () => {
    const before = buildLayeredCausalGraph(constructs, edges, dynamicConstructIds);
    const after = buildLayeredCausalGraph(
      constructs.map((item, index) => ({ ...item, name: `renamed ${index}` })),
      edges,
      dynamicConstructIds,
    );
    expect(after.graph).toEqual(before.graph);
    const identities = (bundle: typeof before) =>
      [...bundle.edgeMeta.values()].map(({ cause, effect, ...edge }) => ({
        ...edge,
        cause: cause.id,
        effect: effect.id,
      }));
    expect(identities(after)).toEqual(identities(before));
    expect(after.segmentMeta).toEqual(before.segmentMeta);
  });
});

it("marks spec revisions without presenting a rename as a topology change", () => {
  const built = buildLayeredCausalGraph(constructs, edges, dynamicConstructIds);
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
    [{ id: treatment.id, x: 0, y: 0, width: 100, height: 60 }],
    200,
    100,
  );
  expect(overlay.constructChanges.get(treatment.id)).toBe("revised");
  expect(overlay.addedNodes).toEqual([]);
  expect(overlay.addedEdges).toEqual([]);
  expect(overlay.marks).toEqual([
    expect.objectContaining({ title: "revised construct", detail: "Exposure" }),
  ]);
});
