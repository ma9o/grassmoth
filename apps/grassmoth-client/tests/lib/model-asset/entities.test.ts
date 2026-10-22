import { describe, expect, it } from "vitest";
import { modelFixture, outcome, treatment } from "@/lib/__fixtures__/model";
import { authoredSnapshot } from "@/lib/__fixtures__/snapshot";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { entityFailures } from "@/lib/model-asset/inspector";
import { modelConstructs, modelEdges } from "@/lib/model-accessors";
import { indexModel, resolveEntity } from "@/lib/model-asset/entities";
import type { EntitySelection } from "@/lib/model-asset/selection";

/** Reorder the same document without changing entity identity or ownership. */
function reversedGraph() {
  return {
    ...modelFixture,
    constructs: Object.fromEntries(Object.entries(modelFixture.constructs).reverse()),
    edges: Object.fromEntries(Object.entries(modelFixture.edges).reverse()),
  };
}

describe("scoped model inspection", () => {
  it("reads the same entities when document entries move", () => {
    const reversed = reversedGraph();
    const constructs = modelConstructs(modelFixture);
    const selections: EntitySelection[] = [
      ...modelEdges(modelFixture).map((item) => ({ kind: "edge" as const, id: item.id })),
      ...constructs.map((item) => ({ kind: "construct" as const, id: item.id })),
      ...constructs.flatMap((item) =>
        item.indicators.map((indicator) => ({
          kind: "indicator" as const,
          id: indicator.observation.id,
        })),
      ),
    ];
    for (const selection of selections) {
      expect(resolveEntity(indexModel(reversed), selection)).toEqual(
        resolveEntity(indexModel(modelFixture), selection),
      );
    }
  });

  it("attributes recorded data and identification issues without requiring a fit", () => {
    const indicator = fixtureValue(outcome.indicators[0]);
    const modelSnapshot = {
      ...authoredSnapshot,
      profile: {
        indicators: {
          [indicator.observation.id]: {
            profile: null,
            findings: [
              {
                code: "missing",
                kind: "evaluated" as const,
                subject: { kind: "indicator" as const, id: indicator.observation.id },
                outcome: "failed" as const,
                evidence: "Missing observations.",
              },
            ],
          },
        },
        findings: [],
      },
      identification: {
        outcome: outcome.id,
        treatments: {
          [treatment.id]: { status: "not_identified" as const, confounders: [], notes: null },
        },
      },
    };
    expect(entityFailures(modelSnapshot, indicator)).toEqual(["Data quality: outcome_reading"]);
    expect(entityFailures(modelSnapshot, outcome)).toEqual(["Data quality: outcome_reading"]);
    expect(entityFailures(modelSnapshot, treatment)).toEqual([
      "Identification against ★: treatment",
    ]);
    expect(entityFailures(authoredSnapshot, outcome)).toEqual([]);
  });
});
