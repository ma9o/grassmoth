import type {
  ConstructId,
  DynamicalModelSpec,
  Expression,
  ParameterId,
} from "@grassmoth/api-types";
import { dump } from "npyjs";
import { describe, expect, it } from "vitest";
import { modelFixture, outcome, decay } from "@/lib/__fixtures__/model";
import {
  evaluateExpression,
  potentialDrift,
  type ExpressionInputs,
} from "@/lib/model-asset/expression-evaluation";
import { modelFlowPlots, type FlowPlot } from "@/lib/model-asset/flow-plots";
import { modelDraws } from "@/lib/model-asset/model-draws";
import { measurementDraw } from "@/lib/model-asset/measurement-draws";

const x = outcome.id;
const theta = decay.id;
const law = "distribution:00000000000000000007" as const;
const indicator = "indicator:00000000000000000007" as const;
const mechanism = "mechanism:00000000000000000007" as const;
const literal = (value: number): Expression => ({ kind: "literal", value });
const state: Expression = { kind: "state", construct_id: x };
const parameter: Expression = {
  kind: "coefficient",
  role: "weight",
  value: theta,
  construct_ids: [],
};
const binary = (
  operator: "add" | "multiply" | "power" | "subtract" | "divide",
  left: Expression,
  right: Expression,
): Expression => ({ kind: "binary", operator, left, right });

function authoredModel(): DynamicalModelSpec {
  const definition = modelFixture.constructs[x];
  if (!definition) throw new Error("Missing test construct.");
  return {
    ...modelFixture,
    constructs: {
      [x]: {
        ...definition,
        dynamics: {
          [mechanism]: { kind: "drift", expression: binary("multiply", parameter, state) },
        },
        coefficients: [
          { kind: "coefficient", role: "initial_mean", value: theta, construct_ids: [] },
          { kind: "coefficient", role: "initial_scale", value: 0, construct_ids: [] },
        ],
        indicators: {
          [indicator]: {
            observation: {
              name: "reading",
              measurement: { dtype: "continuous" },
              aggregation: "mean",
              observation_window: null,
            },
            construct_polarity: "positive",
            likelihood: {
              law: {
                distribution: "Delta",
                v: binary("add", binary("multiply", literal(2), state), parameter),
              },
              standardized: false,
              reasoning: "Test measurement",
              sources: [],
            },
          },
        },
      },
    },
    edges: {},
    parameters: {
      [theta]: {
        name: "theta",
        description: "Shared parameter",
        distribution: law,
        transform: { kind: "identity" },
        sources: [],
        reasoning: null,
      },
    },
    distributions: { [law]: { distribution: "Normal", params: { loc: 2, scale: 0.4 } } },
    law_layouts: {},
  };
}

function values(plot: FlowPlot | undefined): readonly (readonly number[])[] {
  if (plot?.kind !== "draws") throw new Error(plot?.reason ?? "Missing plot");
  return plot.rows;
}

function retainedModel(): DynamicalModelSpec {
  const model = authoredModel();
  const construct = model.constructs[x];
  const definition = model.parameters[theta];
  if (!construct || !definition) throw new Error("Missing test definition.");
  const array = (values: number[], shape: number[]) => ({
    npy: new Uint8Array(dump(values, shape, { dtype: "f8" })),
  });
  const element = "element:00000000000000000007" as const;
  return {
    ...model,
    constructs: { [x]: { ...construct, distribution: law } },
    parameters: { [theta]: { ...definition, transform: { kind: "identity" } } },
    distributions: {
      [law]: {
        distribution: "MixtureSameFamily",
        params: {
          mixing_distribution: {
            distribution: "CategoricalProbs",
            params: { probs: array([1 / 3, 1 / 3, 1 / 3], [3]) },
          },
          component_distribution: {
            distribution: "Delta",
            params: { v: array([3, 2, 4, 2, 3, 6, 1, 6, 12], [3, 3]), event_dim: 1 },
          },
        },
      },
    },
    law_layouts: {
      [law]: {
        parameters: [[theta, [element]]],
        constructs: [x],
        time_points: [0, 1],
        time_origin: "2026-01-01T00:00:00Z",
        labels: { [element]: "theta" },
      },
    },
  };
}

describe("local distributions at model seams", () => {
  it("uses the same prior draw through the initial state, nonlinear mechanism, and reading", () => {
    const model = authoredModel();
    const before = structuredClone(model);
    const plots = modelFlowPlots(model);
    const states = values(plots.constructs.get(x));
    const outputs = values(plots.intrinsic.get(x));
    const readings = values(plots.indicators.get(indicator));
    expect(states).toHaveLength(256);
    states.forEach(([value], draw) => {
      if (value === undefined) throw new Error("Missing test state.");
      expect(outputs[draw]).toEqual([value ** 2]);
      expect(readings[draw]?.[0]).toBeCloseTo(3 * value, 12);
    });
    expect(modelFlowPlots(structuredClone(model))).toEqual(plots);
    expect(model).toEqual(before);
  });

  it("preserves every retained joint row and its trajectory coordinates, without marginal recombination", () => {
    const plots = modelFlowPlots(retainedModel());
    expect(values(plots.constructs.get(x))).toEqual([
      [2, 4],
      [3, 6],
      [6, 12],
    ]);
    expect(values(plots.intrinsic.get(x))).toEqual([
      [6, 12],
      [6, 12],
      [6, 12],
    ]);
    expect(values(plots.indicators.get(indicator))).toEqual([[7], [8], [13]]);
  });

  it("applies authored interval transforms, while conditioned laws remain on their native scale", () => {
    const model = authoredModel();
    const specification = model.parameters[theta];
    if (!specification) throw new Error("Missing test parameter.");
    const persistence: DynamicalModelSpec = {
      ...model,
      measurement_clock: "12h",
      parameters: {
        [theta]: {
          ...specification,
          transform: { kind: "dt_persistence_to_ct_decay", interval_days: "model_clock" },
        },
      },
      distributions: { [law]: { distribution: "Delta", params: { v: 0.8 } } },
    };
    expect(modelDraws(persistence).inputs(0, 0).parameter(theta)).toBeCloseTo(-Math.log(0.8) / 0.5);
    expect(modelDraws(retainedModel()).inputs(0, 0).parameter(theta)).toBe(3);
  });

  it("retains initial-state correlation instead of independently redrawing the state marginals", () => {
    const model = authoredModel();
    const definition = model.constructs[x];
    if (!definition) throw new Error("Missing test construct.");
    const second: ConstructId = "construct:00000000000000000009";
    const mean = {
      kind: "coefficient",
      role: "initial_mean",
      value: 0,
      construct_ids: [],
    } as const;
    const scale = {
      kind: "coefficient",
      role: "initial_scale",
      value: 1,
      construct_ids: [],
    } as const;
    const draws = modelDraws({
      ...model,
      constructs: {
        [x]: { ...definition, coefficients: [mean, scale] },
        [second]: {
          ...definition,
          coefficients: [
            mean,
            scale,
            { kind: "coefficient", role: "initial_correlation", value: 0.8, construct_ids: [x] },
          ],
        },
      },
    });
    const difference =
      Array.from({ length: draws.count }, (_, draw) => {
        const inputs = draws.inputs(draw, 0);
        return (inputs.state(x) - inputs.state(second)) ** 2;
      }).reduce((sum, value) => sum + value, 0) / draws.count;
    expect(difference).toBeGreaterThan(0.25);
    expect(difference).toBeLessThan(0.65); // Independent marginals would give variance 2.
  });

  it("keeps missing or undefined outputs explicit instead of substituting coefficient PDFs", () => {
    const model = authoredModel();
    const definition = model.constructs[x];
    if (!definition) throw new Error("Missing test construct.");
    const missing = modelFlowPlots({
      ...model,
      constructs: { [x]: { ...definition, coefficients: [] } },
    });
    expect(missing.intrinsic.get(x)).toMatchObject({ kind: "unavailable" });
    const undefinedOutput = modelFlowPlots({
      ...model,
      constructs: {
        [x]: {
          ...definition,
          dynamics: {
            [mechanism]: { kind: "drift", expression: binary("divide", literal(1), literal(0)) },
          },
        },
      },
    });
    const undefinedPlot = undefinedOutput.intrinsic.get(x);
    expect(undefinedPlot?.kind).toBe("unavailable");
    if (undefinedPlot?.kind !== "unavailable") throw new Error("Expected an unavailable plot.");
    expect(undefinedPlot.reason).toContain("undefined");
  });
});

describe("saved expression and observation semantics", () => {
  const inputs: ExpressionInputs = { state: () => 2, parameter: () => 3 };
  it("evaluates nonlinear expressions and differentiates a potential into drift", () => {
    const potential = binary(
      "divide",
      binary("multiply", parameter, binary("power", state, literal(4))),
      literal(4),
    );
    expect(evaluateExpression(potential, inputs)).toBe(12);
    expect(potentialDrift(potential, inputs, x)).toBe(-24);
    expect(
      potentialDrift(binary("power", state, literal(2)), { ...inputs, state: () => -2 }, x),
    ).toBe(4);
  });

  it("includes observation noise and the law's link and parameterization", () => {
    const normal = { distribution: "Normal", loc: state, scale: literal(3) } as const;
    expect(measurementDraw(normal, inputs, 0.975)).toBeCloseTo(2 + 3 * 1.9599639845, 6);
    const gamma = { distribution: "Gamma", concentration: literal(1), rate: literal(2) } as const;
    expect(measurementDraw(gamma, inputs, 0.5)).toBeCloseTo(Math.log(2) / 2, 6);
    const bernoulli = {
      distribution: "BernoulliLogits",
      logits: { kind: "call", function: "exp", arguments: [state] },
    } as const;
    expect(measurementDraw(bernoulli, inputs, 0.5)).toBe(1);
    expect(measurementDraw({ distribution: "Poisson", rate: literal(0) }, inputs, 0.9)).toBe(0);
  });

  it("keeps ordered cutpoint gaps and categorical contrasts in their declared order", () => {
    const gaps: ParameterId = "parameter:00000000000000000009";
    const vectorInputs: ExpressionInputs = {
      state: () => 2,
      parameter: (id) => (id === gaps ? [1, 2] : [0, 1]),
    };
    const gapOperand: Expression = {
      kind: "coefficient",
      role: "cutpoint_gaps",
      value: gaps,
      construct_ids: [],
    };
    expect(
      evaluateExpression(
        { kind: "call", function: "ordered_cutpoints", arguments: [literal(-1), gapOperand] },
        vectorInputs,
      ),
    ).toEqual([-1, 0, 2]);
    expect(
      evaluateExpression(
        { kind: "call", function: "category_logits", arguments: [state, parameter, gapOperand] },
        vectorInputs,
      ),
    ).toEqual([0, 2, 5]);
  });

  it("draws count laws without overflowing their probability mass functions", () => {
    expect(measurementDraw({ distribution: "Poisson", rate: literal(1000) }, inputs, 0.5)).toBe(
      1000,
    );
    expect(
      measurementDraw(
        { distribution: "NegativeBinomial2", mean: literal(4), concentration: literal(1) },
        inputs,
        0.99,
      ),
    ).toBe(20);
  });
});
