import type { Expression, ObservationLawSpec } from "@grassmoth/api-types";
import { renderToString } from "katex";
import { describe, expect, it } from "vitest";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { decay, modelFixture, outcome, treatment } from "@/lib/__fixtures__/model";
import { indexModel } from "@/lib/model-asset/entities";
import { constructEquation, observationEquation } from "@/lib/model-asset/equations";

const literal: Expression = { kind: "literal", value: 0.25 };
const state: Expression = { kind: "state", construct_id: treatment.id };
const crossLoading: Expression = {
  kind: "binary",
  operator: "multiply",
  left: literal,
  right: state,
};
const entities = indexModel(modelFixture);
const indicator = fixtureValue(outcome.indicators[0]);
function equation(law: ObservationLawSpec, indexed = entities) {
  return fixtureValue(
    observationEquation(
      {
        ...indicator,
        likelihood: {
          law,
          standardized: false,
          reasoning: "Authored test law.",
          sources: [],
        },
      },
      indexed,
    ),
  );
}

describe("authored equation display", () => {
  it("renders every native observation family and follows construct identity after renaming", () => {
    const laws = {
      Delta: { distribution: "Delta", v: crossLoading },
      Normal: { distribution: "Normal", loc: crossLoading, scale: literal },
      StudentT: { distribution: "StudentT", df: literal, loc: crossLoading, scale: literal },
      Poisson: { distribution: "Poisson", rate: crossLoading },
      Gamma: { distribution: "Gamma", concentration: literal, rate: crossLoading },
      BernoulliLogits: { distribution: "BernoulliLogits", logits: crossLoading },
      BernoulliProbs: { distribution: "BernoulliProbs", probs: crossLoading },
      NegativeBinomial2: {
        distribution: "NegativeBinomial2",
        mean: crossLoading,
        concentration: literal,
      },
      Beta: { distribution: "Beta", concentration1: crossLoading, concentration0: literal },
      OrderedLogistic: {
        distribution: "OrderedLogistic",
        predictor: crossLoading,
        cutpoints: literal,
      },
      Categorical: { distribution: "Categorical", logits: crossLoading },
    } satisfies Record<ObservationLawSpec["distribution"], ObservationLawSpec>;
    const renamed = indexModel({
      ...modelFixture,
      constructs: {
        ...modelFixture.constructs,
        [treatment.id]: {
          ...fixtureValue(modelFixture.constructs[treatment.id]),
          name: "Renamed & safe_{label}",
        },
      },
    });
    for (const law of Object.values(laws)) {
      const latex = equation(law, renamed);
      expect(latex).toContain(`\\operatorname{${law.distribution}}`);
      expect(latex).toContain("0.25");
      expect(latex).toContain("Renamed \\& safe \\{label\\}");
      expect(() => renderToString(latex, { throwOnError: true })).not.toThrow();
    }
    expect(observationEquation(indicator, entities)).toBeNull();
  });

  it("formats every expression operation, parameter transform and unfinished operand without evaluation", () => {
    const operators = ["add", "subtract", "multiply", "divide", "power", "maximum"] as const;
    const functions = [
      "exp",
      "sigmoid",
      "normal_cdf",
      "ordered_cutpoints",
      "category_logits",
    ] as const;
    const unfinished: Expression = {
      kind: "coefficient",
      role: "loading",
      value: null,
      construct_ids: [],
    };
    const expressions: Expression[] = [
      { kind: "literal", value: 1e-9 },
      unfinished,
      { kind: "coefficient", role: "loading", value: -2, construct_ids: [] },
      ...operators.map(
        (operator): Expression => ({
          kind: "binary",
          operator,
          left: crossLoading,
          right: unfinished,
        }),
      ),
      ...functions.map(
        (name): Expression => ({ kind: "call", function: name, arguments: [crossLoading] }),
      ),
    ];
    for (const value of expressions)
      expect(() =>
        renderToString(equation({ distribution: "Delta", v: value }), { throwOnError: true }),
      ).not.toThrow();
    expect(equation({ distribution: "Delta", v: unfinished })).toContain("\\underbrace{?}");
    for (const transform of [
      { kind: "identity" },
      { kind: "initial_state_correlation" },
      { kind: "dt_persistence_to_ct_decay", interval_days: 1 },
      { kind: "dt_effect_to_ct_rate", interval_days: 2 },
    ] as const) {
      const indexed = indexModel({
        ...modelFixture,
        parameters: {
          ...modelFixture.parameters,
          [decay.id]: { ...fixtureValue(modelFixture.parameters[decay.id]), transform },
        },
      });
      const latex = equation(
        {
          distribution: "Delta",
          v: { kind: "coefficient", role: "decay", value: decay.id, construct_ids: [] },
        },
        indexed,
      );
      expect(latex.includes("\\Delta")).toBe(transform.kind.startsWith("dt_"));
      expect(latex.includes("\\log")).toBe(transform.kind === "dt_persistence_to_ct_decay");
      expect(() => renderToString(latex, { throwOnError: true })).not.toThrow();
    }
  });

  it("keeps potentials symbolic and displays direct authored links without execution classifications", () => {
    const construct = {
      ...outcome,
      dynamics: [
        {
          id: "mechanism:00000000000000000002" as const,
          kind: "potential" as const,
          expression: crossLoading,
        },
      ],
    };
    const drift = fixtureValue(constructEquation(construct, entities));
    expect(drift.title).toBe("Authored drift");
    expect(drift.latex).toContain("\\partial");
    expect(() => renderToString(drift.latex, { throwOnError: true })).not.toThrow();
    const links = fixtureValue(
      constructEquation(
        { ...treatment, dynamics: [] },
        { ...entities, edges: entities.edges.filter((edge) => edge.effect.id !== treatment.id) },
      ),
    );
    expect(links.title).toBe("Authored connections");
    expect(links.latex).toContain("\\to");
    expect(links.latex).toContain("outcome");
    expect(() => renderToString(links.latex, { throwOnError: true })).not.toThrow();
  });
});
