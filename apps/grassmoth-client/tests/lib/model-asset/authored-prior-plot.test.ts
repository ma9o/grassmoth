import type { NumPyroDistribution } from "@grassmoth/api-types";
import { describe, expect, it } from "vitest";
import { authoredPriorPlot } from "@/lib/model-asset/authored-prior-plot";

describe("authored prior plots", () => {
  it("derives a deterministic curve without mutating the native law", () => {
    const law: NumPyroDistribution = {
      distribution: "Gamma",
      params: { concentration: 2, rate: 3 },
    };
    const before = structuredClone(law);
    const plot = authoredPriorPlot(law);
    expect(plot.x).toHaveLength(129);
    expect(authoredPriorPlot(structuredClone(law))).toEqual(plot);
    expect(law).toEqual(before);
    expect(authoredPriorPlot({ ...law, params: { concentration: 2, rate: 6 } })).not.toEqual(plot);
  });

  it.each<NumPyroDistribution>([
    { distribution: "Delta", params: { v: 1 } },
    { distribution: "BernoulliProbs", params: { probs: 0.5 } },
    { distribution: "Normal", params: { loc: { array: [0, 1], dtype: "float32" }, scale: 1 } },
    { distribution: "MultivariateNormal", params: {} },
    { distribution: "TransformedDistribution", params: {} },
    { distribution: "MixtureSameFamily", params: {} },
  ])("throws for a $distribution constructor outside scalar authoring", (law) => {
    expect(() => authoredPriorPlot(law)).toThrow();
  });

  it("throws if density evaluation cannot produce finite plot points", () => {
    expect(() =>
      authoredPriorPlot({
        distribution: "LeftTruncatedDistribution",
        params: {
          low: 100,
          base_dist: { distribution: "Normal", params: { loc: 0, scale: 1 } },
        },
      }),
    ).toThrow("Cannot plot a finite scalar density for the LeftTruncatedDistribution prior.");
  });
});
