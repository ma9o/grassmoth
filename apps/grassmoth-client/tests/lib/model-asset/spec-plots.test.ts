import type { ConstructId, DynamicalModelSpec, Expression } from "@grassmoth/api-types";
import { describe, expect, it } from "vitest";
import { decay, modelFixture, outcome, treatment } from "@/lib/__fixtures__/model";
import { authoredSnapshot, fittedSnapshot } from "@/lib/__fixtures__/snapshot";
import { modelDraws } from "@/lib/model-asset/model-draws";
import { glyphKey, modelSpecPlots, viewedSpecPlots } from "@/lib/model-asset/spec-plots";

const mechanism = "mechanism:input_product" as const;
const indicator = "indicator:reading" as const;
const theta = { kind: "coefficient", role: "weight", value: decay.id, construct_ids: [] } as const;
const state = (id: ConstructId): Expression => ({ kind: "state", construct_id: id });
const x = outcome.id;
const other = treatment.id;

function model(): DynamicalModelSpec {
  const target = modelFixture.constructs[x];
  const parent = modelFixture.constructs[other];
  if (!target || !parent) throw new Error("Missing fixture constructs.");
  return {
    ...modelFixture,
    time: { ...modelFixture.time, step: "6h" },
    edges: {},
    constructs: {
      [x]: {
        ...target,
        coefficients: [
          { kind: "coefficient", role: "initial_mean", value: 2, construct_ids: [] },
          { kind: "coefficient", role: "initial_scale", value: 1, construct_ids: [] },
          { kind: "coefficient", role: "diffusion_scale", value: 3, construct_ids: [] },
        ],
        dynamics: {
          [mechanism]: {
            kind: "drift",
            expression: {
              kind: "binary",
              operator: "multiply",
              left: theta,
              right: { kind: "binary", operator: "add", left: state(x), right: state(other) },
            },
          },
        },
        indicators: {
          [indicator]: {
            observation: {
              name: "reading",
              measurement: { dtype: "continuous" },
              aggregation: "mean",
              observation_window: null,
            },
            reasoning: { text: "Synthetic conditional law for the chart.", reference_urls: [] },
            construct_polarity: "positive",
            likelihood: {
              standardized: false,
              law: { distribution: "Normal", loc: state(x), scale: theta },
            },
          },
        },
      },
      [other]: {
        ...parent,
        coefficients: [
          { kind: "coefficient", role: "initial_mean", value: decay.id, construct_ids: [] },
          { kind: "coefficient", role: "initial_scale", value: 0, construct_ids: [] },
        ],
      },
    },
  };
}

describe("authored state laws and functions", () => {
  it("uses the initial law's central range and the noise at the model step, including point masses", () => {
    const plots = modelSpecPlots(model());
    const initial = plots.get(`${x}/initial`);
    const noise = plots.get(`${x}/noise`);
    const point = plots.get(`${other}/initial`);
    if (initial?.kind !== "distribution" || initial.law.shape.kind !== "plot") {
      throw new Error("Missing initial law.");
    }
    expect(initial.law.shape.value.domain[0]).toBeCloseTo(2 - 2.326347874);
    expect(initial.law.shape.value.domain[1]).toBeCloseTo(2 + 2.326347874);
    expect(initial.law.shape.value.curve.every((p) => p.value === p.min && p.min === p.max)).toBe(
      true,
    );
    expect(noise).toMatchObject({ law: { family: "Normal(0, 1.5)" } });
    if (point?.kind !== "distribution" || point.law.shape.kind !== "plot") {
      throw new Error("Missing point law.");
    }
    expect(point.law.shape.value.atoms).toEqual(point.law.shape.value.domain);
    expect(point.law.shape.value.curve).toEqual([]);
  });

  it("sweeps each read state separately with all other states and transformed coefficients on their original draw", () => {
    const spec = model();
    const term = modelSpecPlots(spec).get(glyphKey({ kind: "construct", id: x, glyph: mechanism }));
    if (term?.kind !== "response") throw new Error("Missing mechanism.");
    expect(term.slices.map((s) => s.input)).toEqual([x, other]);
    const draws = modelDraws(spec);
    for (const slice of term.slices) {
      if (slice.plot.kind !== "plot") throw new Error(slice.plot.reason);
      const p = slice.plot.value.points.at(-1);
      if (!p) throw new Error("Missing function endpoint.");
      const expected = Array.from({ length: draws.count }, (_, i) => {
        const inputs = draws.inputs(i, 0);
        const coefficient = inputs.parameter(decay.id);
        if (typeof coefficient !== "number") throw new Error("Expected scalar.");
        return coefficient * (p.x + inputs.state(slice.input === x ? other : x));
      });
      const median = -Math.log(0.5);
      const reference = median * (p.x + (slice.input === x ? median : 2));
      expect(p.value).toBeCloseTo(reference, 7);
      expect(p.min).toBeCloseTo(Math.min(...expected));
      expect(p.max).toBeCloseTo(Math.max(...expected));
    }
  });

  it("keeps the conditional noise and uncertainty about its width separate from the expected reading", () => {
    const spec = model();
    const plots = modelSpecPlots(spec);
    const node = plots.get(indicator);
    const reading = plots.get(`${indicator}/response`);
    if (reading?.kind !== "response") throw new Error("Missing reading.");
    const slice = reading.slices[0]?.plot;
    if (slice?.kind !== "plot") throw new Error("Missing reading curve.");
    const p = slice.value.points[0];
    if (!p) throw new Error("Missing endpoint.");
    expect([p.min, p.max, p.value]).toEqual([p.x, p.x, p.x]);
    expect(reading.uses).toEqual([]);
    if (node?.kind !== "distribution" || node.law.shape.kind !== "plot") {
      throw new Error("Missing reading distribution.");
    }
    expect(node.law.label).toBe("At outcome = 2");
    expect(node.law.shape.value.curve.some((q) => q.min < q.max)).toBe(true);
    expect(node.uses.map((u) => u.parameterId)).toEqual([decay.id]);
  });

  it("gives a zero-centred weight a flat reference across a nonzero range of slopes", () => {
    const spec = model();
    const key = decay.distribution;
    const parameter = spec.parameters[decay.id];
    if (!key || !parameter) throw new Error("Missing weight fixture.");
    const zero = {
      ...spec,
      parameters: {
        ...spec.parameters,
        [decay.id]: { ...parameter, transform: { kind: "identity" as const } },
      },
      distributions: {
        ...spec.distributions,
        [key]: { distribution: "Normal", params: { loc: 0, scale: 0.04 } },
      },
    };
    const term = modelSpecPlots(zero).get(glyphKey({ kind: "construct", id: x, glyph: mechanism }));
    if (term?.kind !== "response") throw new Error("Missing mechanism.");
    const slice = term.slices[0]?.plot;
    if (slice?.kind !== "plot") throw new Error("Missing reference.");
    expect(slice.value.points.every((p) => p.value === 0)).toBe(true);
    expect(slice.value.points.some((p) => p.min < 0 && p.max > 0)).toBe(true);
  });

  it("uses discrete conditional mass at the labelled initial centre", () => {
    const spec = model();
    const target = spec.constructs[x];
    const obs = target?.indicators[indicator];
    if (!target || !obs) throw new Error("Missing observation.");
    const count: DynamicalModelSpec = {
      ...spec,
      constructs: {
        ...spec.constructs,
        [x]: {
          ...target,
          indicators: {
            [indicator]: {
              ...obs,
              likelihood: {
                standardized: false,
                law: {
                  distribution: "NegativeBinomial2",
                  mean: state(x),
                  concentration: { kind: "literal", value: 1 },
                },
              },
            },
          },
        },
      },
    };
    const node = modelSpecPlots(count).get(indicator);
    if (node?.kind !== "distribution" || node.law.shape.kind !== "plot") {
      throw new Error("Missing mass plot.");
    }
    expect(node.law.family).toBe("NegativeBinomial2 · concentration 1");
    expect(node.law.label).toBe("At outcome = 2");
    const { binWidth, curve } = node.law.shape.value;
    expect(binWidth).toBe(1);
    expect(curve[0]).toMatchObject({ x: 0, value: 1 });
    expect(curve[1]?.value).toBeCloseTo(2 / 3);
    expect(curve.every((p) => p.min === p.max)).toBe(true);
  });

  it("scopes authored glyphs to the edit revision", () => {
    expect(viewedSpecPlots(fittedSnapshot, "edit_model")).toBeNull();
    for (const step of ["prepare_data", "simulate", "data_diff", "model_diff"] as const) {
      expect(viewedSpecPlots(authoredSnapshot, step)).toBeNull();
    }
  });
});
