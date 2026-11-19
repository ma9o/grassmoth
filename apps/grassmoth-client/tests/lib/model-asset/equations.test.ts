import type {
  BinaryOperator,
  CoefficientRole,
  DynamicalModelSpec,
  Expression,
  ObservationLawSpec,
  ParameterId,
} from "@grassmoth/api-types";
import { describe, expect, it } from "vitest";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import {
  baseline,
  decay,
  diffusion,
  modelFixture,
  outcome,
  treatment,
} from "@/lib/__fixtures__/model";
import { indexModel } from "@/lib/model-asset/entities";
import {
  type Equation,
  type EquationToken,
  modelEquations,
  numberParts,
} from "@/lib/model-asset/equations";

const x: Expression = { kind: "state", construct_id: outcome.id };
const u: Expression = { kind: "state", construct_id: treatment.id };
const literal = (value: number): Expression => ({ kind: "literal", value });
const binary = (operator: BinaryOperator, left: Expression, right: Expression): Expression => ({
  kind: "binary",
  operator,
  left,
  right,
});
const coefficient = (role: CoefficientRole, value: number | ParameterId | null): Expression => ({
  kind: "coefficient",
  role,
  value,
  construct_ids: [],
});
const quarter = binary("multiply", literal(0.25), u);
const reading = fixtureValue(outcome.indicators[0]).observation.id;
const edge = "edge:00000000000000000002";

function tokenText(token: EquationToken): string {
  switch (token.kind) {
    case "symbol":
    case "word": {
      return token.text.replaceAll(/\s/gu, " ");
    }
    case "number": {
      const { mantissa, exponent } = numberParts(token.value);
      return exponent === null ? mantissa : `${mantissa}e${exponent}`;
    }
    case "state": {
      return token.construct.name;
    }
    case "parameter": {
      return token.parameter.name;
    }
    case "unassigned": {
      return `?${token.role}`;
    }
    case "superscript": {
      return `^${token.tokens.map((part) => tokenText(part)).join("")}`;
    }
  }
}

/** An equation as plain text, one space between the phrases a line keeps whole. */
function text(equation: Equation): string {
  return equation.map((phrase) => phrase.map((token) => tokenText(token)).join("")).join(" ");
}

type Definition = NonNullable<DynamicalModelSpec["constructs"][typeof outcome.id]>;

function revised(
  id: typeof outcome.id,
  change: Partial<Definition>,
  model = modelFixture,
): DynamicalModelSpec {
  return {
    ...model,
    constructs: {
      ...model.constructs,
      [id]: { ...fixtureValue(model.constructs[id]), ...change },
    },
  };
}

function block(model: DynamicalModelSpec, id = outcome.id) {
  return fixtureValue(
    modelEquations(model, indexModel(model)).find((item) => item.construct.id === id),
  );
}

function drift(...expressions: readonly Expression[]): string[] {
  const dynamics = Object.fromEntries(
    expressions.map((expression, index) => [
      `mechanism:${index}`,
      { kind: "drift" as const, expression },
    ]),
  );
  return block(revised(outcome.id, { dynamics })).terms.map((term) => text(term.equation));
}

function law(value: ObservationLawSpec, model = modelFixture): string {
  const definition = fixtureValue(model.constructs[outcome.id]);
  const indicators = {
    [reading]: {
      ...fixtureValue(definition.indicators[reading]),
      likelihood: { law: value, standardized: false },
    },
  };
  return text(fixtureValue(block(revised(outcome.id, { indicators }, model)).readings[0]?.law));
}

describe("authored equations", () => {
  it("reads the model as the graph's cards do: terms of each effect, then its own laws", () => {
    const response = { kind: "drift" as const, expression: binary("multiply", u, x) };
    const model: DynamicalModelSpec = {
      ...modelFixture,
      edges: {
        ...modelFixture.edges,
        [edge]: {
          ...fixtureValue(modelFixture.edges[edge]),
          mechanisms: { "mechanism:00000000000000000009": response },
        },
      },
    };
    const blocks = modelEquations(model, indexModel(model));
    expect(blocks.map((item) => item.construct.id)).toEqual([
      baseline.id,
      treatment.id,
      outcome.id,
    ]);
    const [, driven, effect] = blocks;
    // An edge that declares no mechanism stays a bare link of its effect.
    expect(driven?.terms).toEqual([]);
    expect(driven?.links.map((link) => [link.edge.id, link.cause.id])).toEqual([
      ["edge:00000000000000000001", baseline.id],
    ]);
    expect(effect?.terms.map(({ term, equation }) => [term.owner, text(equation)])).toEqual([
      [{ kind: "construct", id: outcome.id }, "(−log(outcome_persistence) / 1 d) · (0 − outcome)"],
      [{ kind: "edge", id: edge }, "treatment · outcome"],
    ]);
    expect(effect?.links).toEqual([]);
    expect(effect?.coefficients.map((line) => [line.operand.role, text(line.equation)])).toEqual([
      ["diffusion_scale", "outcome_diffusion"],
    ]);
    expect(effect?.readings.map((line) => line.law)).toEqual([null]);
    expect(effect?.parameters.map((line) => [line.parameter.id, line.owner, line.law])).toEqual([
      [
        decay.id,
        { kind: "construct", id: outcome.id },
        { kind: "scalar", law: model.distributions["distribution:00000000000000000001"] },
      ],
      [
        diffusion.id,
        { kind: "construct", id: outcome.id },
        { kind: "scalar", law: model.distributions["distribution:00000000000000000002"] },
      ],
    ]);
  });

  it("typesets the saved tree without regrouping, simplifying or evaluating it", () => {
    const offset = binary("subtract", x, coefficient("center", 40));
    expect(
      drift(
        // The restoring force as its constructor builds it, then a second term that starts signed.
        binary(
          "subtract",
          binary("multiply", binary("multiply", literal(-1), coefficient("decay", 0.5)), offset),
          binary("multiply", coefficient("quartic", 0), binary("power", offset, literal(3))),
        ),
        binary("multiply", literal(-1), x),
      ),
    ).toEqual(["−0.5 · (outcome − 40) − 0 · (outcome − 40)^3", "(−outcome)"]);
    const grouping: readonly (readonly [Expression, string])[] = [
      [binary("subtract", x, binary("add", u, literal(1))), "outcome − (treatment + 1)"],
      [binary("add", binary("subtract", x, u), literal(1)), "outcome − treatment + 1"],
      [binary("multiply", x, binary("multiply", u, literal(2))), "outcome · (treatment · 2)"],
      [binary("multiply", binary("multiply", x, u), literal(2)), "outcome · treatment · 2"],
      [binary("multiply", binary("divide", x, u), literal(2)), "(outcome / treatment) · 2"],
      [
        binary("divide", binary("multiply", x, u), binary("multiply", literal(2), x)),
        "outcome · treatment / (2 · outcome)",
      ],
      [binary("power", binary("multiply", literal(-1), x), u), "(−outcome)^treatment"],
      [binary("add", x, literal(-2)), "outcome + (−2)"],
      [binary("maximum", x, literal(1e-12)), "max(outcome, 1e−12)"],
      [
        { kind: "call", function: "sigmoid", arguments: [binary("add", x, u)] },
        "logistic(outcome + treatment)",
      ],
    ];
    for (const [expression, expected] of grouping) expect(drift(expression)).toEqual([expected]);
    expect(law({ distribution: "Delta", v: coefficient("loading", null) })).toBe(
      "Delta(v = ?loading)",
    );
    for (const transform of [
      { kind: "identity" },
      { kind: "initial_state_correlation" },
      { kind: "dt_persistence_to_ct_decay", interval_days: 1 },
      { kind: "dt_effect_to_ct_rate", interval_days: "model_clock" },
    ] as const) {
      const model = {
        ...modelFixture,
        parameters: {
          ...modelFixture.parameters,
          [decay.id]: { ...fixtureValue(modelFixture.parameters[decay.id]), transform },
        },
      };
      expect(law({ distribution: "Delta", v: coefficient("decay", decay.id) }, model)).toBe(
        {
          identity: "Delta(v = outcome_persistence)",
          initial_state_correlation: "Delta(v = outcome_persistence)",
          dt_persistence_to_ct_decay: "Delta(v = −log(outcome_persistence) / 1 d)",
          dt_effect_to_ct_rate: "Delta(v = outcome_persistence / model clock tick)",
        }[transform.kind],
      );
    }
  });

  it("typesets every native observation family and follows construct identity after renaming", () => {
    const scale = literal(2);
    const laws = {
      Delta: { distribution: "Delta", v: quarter },
      Normal: { distribution: "Normal", loc: quarter, scale },
      StudentT: { distribution: "StudentT", df: scale, loc: quarter, scale },
      Poisson: { distribution: "Poisson", rate: quarter },
      Gamma: { distribution: "Gamma", concentration: scale, rate: quarter },
      BernoulliLogits: { distribution: "BernoulliLogits", logits: quarter },
      BernoulliProbs: { distribution: "BernoulliProbs", probs: quarter },
      NegativeBinomial2: { distribution: "NegativeBinomial2", mean: quarter, concentration: scale },
      Beta: { distribution: "Beta", concentration1: quarter, concentration0: scale },
      OrderedLogistic: { distribution: "OrderedLogistic", predictor: quarter, cutpoints: scale },
      Categorical: { distribution: "Categorical", logits: quarter },
    } satisfies Record<ObservationLawSpec["distribution"], ObservationLawSpec>;
    const renamed = revised(treatment.id, { name: "renamed" });
    for (const value of Object.values(laws)) {
      const typeset = law(value, renamed);
      expect(typeset.startsWith(`${value.distribution}(`)).toBe(true);
      expect(typeset).toContain("0.25 · renamed");
    }
    expect(law(laws.StudentT)).toBe("StudentT(df = 2, loc = 0.25 · treatment, scale = 2)");
  });

  it("keeps potentials symbolic and states each law once, where the model first names it", () => {
    const path = "distribution:00000000000000000009";
    const definition = fixtureValue(modelFixture.constructs[treatment.id]);
    const indicator = fixtureValue(treatment.indicators[0]).observation.id;
    const model: DynamicalModelSpec = {
      ...revised(
        outcome.id,
        {
          dynamics: {
            "mechanism:00000000000000000001": {
              kind: "potential",
              expression: binary("multiply", coefficient("decay", decay.id), x),
            },
          },
        },
        revised(
          treatment.id,
          {
            indicators: {
              [indicator]: {
                ...fixtureValue(definition.indicators[indicator]),
                likelihood: {
                  law: { distribution: "Delta", v: coefficient("loading", diffusion.id) },
                  standardized: false,
                },
              },
            },
          },
          revised(baseline.id, {
            role: "exogenous",
            temporal_status: "time_varying",
            distribution: path,
          }),
        ),
      ),
      parameters: {
        ...modelFixture.parameters,
        [decay.id]: { ...fixtureValue(modelFixture.parameters[decay.id]), distribution: null },
      },
      distributions: {
        ...modelFixture.distributions,
        [path]: { distribution: "Delta", params: { v: { array: [48, 40] }, event_dim: 1 } },
      },
      law_layouts: {
        [path]: {
          parameters: [],
          constructs: [baseline.id],
          time_points: [0, 300],
          time_origin: "1936-01-01T00:00:00Z",
          labels: {},
        },
        "distribution:00000000000000000002": {
          parameters: [[diffusion.id, []]],
          constructs: [],
          time_points: [],
          time_origin: "relative",
          labels: {},
        },
      },
    };
    expect(block(model, baseline.id).law).toEqual({
      kind: "path",
      points: [
        { time: 0, value: 48 },
        { time: 300, value: 40 },
      ],
      timeOrigin: "1936-01-01T00:00:00Z",
    });
    expect(text(fixtureValue(block(model).terms[0]).equation)).toBe(
      "−∂/∂outcome [(−log(outcome_persistence) / 1 d) · outcome]",
    );
    // The reading that names the diffusion parameter first owns its law line.
    expect(
      block(model, treatment.id).parameters.map((line) => [line.owner, line.law.kind]),
    ).toEqual([[{ kind: "indicator", id: indicator }, "joint"]]);
    expect(block(model).parameters.map((line) => [line.parameter.id, line.law.kind])).toEqual([
      [decay.id, "unassigned"],
    ]);
  });
});
