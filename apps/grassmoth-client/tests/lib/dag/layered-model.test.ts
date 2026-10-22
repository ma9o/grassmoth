import { presentEntries } from "@/lib/model-accessors";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { indexModel } from "@/lib/model-asset/entities";
import { describe, expect, it } from "vitest";
import type { ModelSnapshot, DynamicalModelSpec } from "@grassmoth/api-types";
import { baseline, modelFixture, outcome, treatment } from "@/lib/__fixtures__/model";
import { authoredSnapshot, emptySnapshot, fittedSnapshot } from "@/lib/__fixtures__/snapshot";
import { availableGraphLayers, graphEntities, graphStatus } from "@/lib/dag/layered-model";

const measuredModel: DynamicalModelSpec = {
  ...modelFixture,
  constructs: Object.fromEntries(
    presentEntries(modelFixture.constructs).map(([id, construct]) => [
      id,
      { ...construct, dynamics: {}, coefficients: [] },
    ]),
  ),
  parameters: {},
  distributions: {},
};
const structural: ModelSnapshot = {
  ...authoredSnapshot,
  dynamical_model_spec: {
    ...measuredModel,
    constructs: Object.fromEntries(
      presentEntries(measuredModel.constructs).map(([id, construct]) => [
        id,
        { ...construct, indicators: {} },
      ]),
    ),
  },
  identification: null,
};
const measured: ModelSnapshot = {
  ...authoredSnapshot,
  dynamical_model_spec: measuredModel,
};

describe("semantic graph layers", () => {
  it("uses the facts available in each snapshot", () => {
    expect(availableGraphLayers(emptySnapshot)).toEqual([]);
    expect(availableGraphLayers(structural)).toEqual(["structure"]);
    expect(availableGraphLayers(measured)).toEqual(["structure", "measurement", "design"]);
    expect(availableGraphLayers(fittedSnapshot)).toEqual([
      "structure",
      "measurement",
      "design",
      "specification",
      "fit",
    ]);
    expect(availableGraphLayers({ ...fittedSnapshot, fit: null })).not.toContain("fit");
  });

  it("renders the complete authored DAG, including latent constructs, independently of identification", () => {
    expect(
      graphEntities(indexModel(structural.dynamical_model_spec)).constructs.map((item) => item.id),
    ).toEqual([baseline.id, treatment.id, outcome.id]);
    const graph = graphEntities(indexModel(measured.dynamical_model_spec));
    expect(graph.constructs).toHaveLength(3);
    expect(graph.constructs.map((item) => item.id)).toContain(baseline.id);
    expect(graph.constructs.map((item) => item.id)).toEqual([
      baseline.id,
      treatment.id,
      outcome.id,
    ]);
    expect(graph.edges.map((item) => item.id)).toEqual([
      "edge:00000000000000000001",
      "edge:00000000000000000002",
    ]);
    // Identification findings decorate the authored graph without filtering it.
    const marked: ModelSnapshot = {
      ...measured,
      identification: {
        outcome: outcome.id,
        treatments: {
          [treatment.id]: { status: "not_identified", confounders: [baseline.id], notes: null },
        },
      },
    };
    expect(graphStatus(marked, treatment.id)).toBe("blocking");
    expect(graphStatus(measured, treatment.id)).toBe("observed");
    expect(graphStatus(measured, baseline.id)).toBe("latent");
    expect(graphStatus(structural, treatment.id)).toBe("latent");
    expect(graphEntities(indexModel(marked.dynamical_model_spec))).toEqual(graph);
    expect(Object.keys(fixtureValue(measured.dynamical_model_spec).edges)).toHaveLength(2);
    expect(graphEntities(indexModel(emptySnapshot.dynamical_model_spec)).constructs).toEqual([]);
  });
});
