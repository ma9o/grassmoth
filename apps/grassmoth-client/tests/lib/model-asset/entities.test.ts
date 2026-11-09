import { describe, expect, it } from "vitest";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { decay, modelFixture, outcome, treatment } from "@/lib/__fixtures__/model";
import { authoredSnapshot, fittedSnapshot } from "@/lib/__fixtures__/snapshot";
import { modelConstructs, modelEdges } from "@/lib/model-accessors";
import { indexModel, resolveEntity } from "@/lib/model-asset/entities";
import { entityFailures } from "@/lib/model-asset/inspector";
import type { EntitySelection } from "@/lib/model-asset/selection";

/** Reorder the same document without changing entity identity or ownership. */
function reversedGraph() {
  return {
    ...modelFixture,
    constructs: Object.fromEntries(Object.entries(modelFixture.constructs).toReversed()),
    edges: Object.fromEntries(Object.entries(modelFixture.edges).toReversed()),
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
          [treatment.id]: { status: "not_identified" as const, confounders: [] },
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
  it("keeps parameter and latent mixing advisories distinct from convergence failures", () => {
    const fit = fixtureValue(fittedSnapshot.fit);
    const evidence = {
      criterion: "r_hat",
      value: 1.02,
      lower: null,
      upper: 1.01,
      lower_inclusive: true,
      upper_inclusive: false,
      note: "Inspect mixing.",
      display_value: "",
      band_label: "< 1.01",
    };
    const modelSnapshot = {
      ...fittedSnapshot,
      fit: {
        ...fit,
        convergence: {
          checked: 2,
          status: "passed" as const,
          findings: [
            {
              kind: "evaluated" as const,
              code: "fit_mixing",
              outcome: "failed" as const,
              subject: {
                parameter: {
                  parameter_id: decay.id,
                  element_id: `element:${decay.id.slice("parameter:".length)}` as const,
                },
                label: "decay",
              },
              evidence,
            },
            {
              kind: "evaluated" as const,
              code: "fit_mixing",
              outcome: "failed" as const,
              subject: { state: { kind: "construct" as const, id: outcome.id }, label: "outcome" },
              evidence,
            },
          ],
        },
      },
    };
    const issues = entityFailures(modelSnapshot, outcome);
    expect(issues).toContain("Mixing advisory: decay");
    expect(issues).toContain("Mixing advisory: outcome");
    expect(issues.some((issue) => issue.includes("convergence:"))).toBe(false);
  });
});
