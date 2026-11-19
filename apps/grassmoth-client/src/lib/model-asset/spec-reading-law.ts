import type {
  ConstructId,
  DynamicalModelSpec,
  Expression,
  IndicatorSpec,
  ObservationLawSpec,
} from "@grassmoth/api-types";
import { type CoefficientUse, coefficientUses } from "@/lib/model-accessors";
import { formatSignificant } from "@/lib/utils/format";
import {
  type ExpressionInputs,
  evaluateExpression,
  finiteValue,
  PlotUnavailable,
  scalarValue,
} from "./expression-evaluation";
import { expressionStates } from "./flow-plots";
import {
  type ConditionalReading,
  conditionalReading,
  type ReadingMeasure,
} from "./measurement-draws";
import type { ModelDraws } from "./model-draws";
import { observationArguments } from "./observation-law";
import { humanize } from "./selection";
import {
  plotGrid,
  projectPlot,
  type Range,
  type SpecLawPlot,
  type StateShape,
} from "./spec-state-law";

const scalar = (expr: Expression, inputs: ExpressionInputs) =>
  finiteValue(scalarValue(evaluateExpression(expr, inputs)));
const direct = (mean: Expression, spread: readonly Expression[]) => ({
  response: [mean],
  spread,
  expected: (inputs: ExpressionInputs) => scalar(mean, inputs),
});
/** The saved location and family arguments that determine the response and the conditional spread. */
function readingOperands(law: ObservationLawSpec) {
  const all = observationArguments(law).map(([, expr]) => expr);
  const family = (spread: readonly Expression[]) => ({
    response: all,
    spread,
    expected: (inputs: ExpressionInputs) => conditionalReading(law, inputs).mean(),
  });
  switch (law.distribution) {
    case "Normal": {
      return direct(law.loc, [law.scale]);
    }
    case "StudentT": {
      return {
        ...direct(law.loc, [law.df, law.scale]),
        expected: (inputs: ExpressionInputs) => {
          if (scalar(law.df, inputs) <= 1) {
            throw new PlotUnavailable("The StudentT mean is undefined for df ≤ 1.");
          }
          return scalar(law.loc, inputs);
        },
      };
    }
    case "NegativeBinomial2": {
      return direct(law.mean, [law.concentration]);
    }
    case "Gamma": {
      return law.rate.kind === "binary" && law.rate.operator === "divide"
        ? direct(law.rate.right, [law.concentration])
        : family(all);
    }
    case "Beta": {
      const alpha = law.concentration1;
      return alpha.kind === "binary" && alpha.operator === "multiply"
        ? direct(alpha.left, [alpha.right])
        : family(all);
    }
    case "OrderedLogistic": {
      return family([law.cutpoints]);
    }
    case "Categorical": {
      return family(law.logits.kind === "call" ? law.logits.arguments.slice(1) : all);
    }
    case "Delta": {
      return direct(law.v, []);
    }
    case "BernoulliProbs":
    case "BernoulliLogits":
    case "Poisson": {
      return family([]);
    }
  }
}

export function readingArguments(indicator: IndicatorSpec) {
  if (!indicator.likelihood) {
    return {
      response: [],
      spread: [],
      expected: () => {
        throw new PlotUnavailable("No observation law assigned.");
      },
    };
  }
  const law = indicator.likelihood.law;
  return readingOperands(law);
}

function readingFamily(
  law: ObservationLawSpec,
  model: DynamicalModelSpec,
  center: ExpressionInputs,
): string {
  const args = [...observationArguments(law)];
  if (law.distribution === "Beta" && law.concentration1.kind === "binary") {
    args.push(["concentration", law.concentration1.right]);
  }
  const fixed = args.flatMap(([name, expr]) => {
    if (expr.kind !== "literal" && expr.kind !== "coefficient") return [];
    if (typeof expr.value === "number") return [`${name} ${formatSignificant(expr.value)}`];
    if (!expr.value || typeof expr.value !== "string") return [];
    const id = model.parameters[expr.value]?.distribution;
    if (!id || model.distributions[id]?.distribution !== "Delta") return [];
    const coordinate = projectPlot(() => evaluateExpression(expr, center));
    return coordinate.kind === "plot" && typeof coordinate.value === "number"
      ? [`${name} ${formatSignificant(coordinate.value)}`]
      : [];
  });
  return [law.distribution, ...fixed].join(" · ");
}

function measureHeight(measure: ReadingMeasure, x: number, binWidth: number): number {
  if (measure.kind === "point") return 0;
  if (measure.kind === "density") return finiteValue(measure.height(x));
  // A wide count axis groups only adjacent outcomes into exact probability masses.
  const first = Math.round(x - (binWidth - 1) / 2);
  return finiteValue(Math.max(0, measure.cdf(first + binWidth - 1) - measure.cdf(first - 1)));
}

function readingShape(
  reference: ConditionalReading,
  samples: readonly ConditionalReading[],
): StateShape {
  const laws = [reference, ...samples];
  const lastCategory =
    reference.measure.kind === "mass" ? reference.measure.lastCategory : undefined;
  const low =
    lastCategory === undefined
      ? Math.min(...laws.map((law) => finiteValue(law.quantile(0.01))))
      : 0;
  const high = lastCategory ?? Math.max(...laws.map((law) => finiteValue(law.quantile(0.99))));
  if (reference.measure.kind === "point") {
    return {
      domain: [low, high],
      curve: [],
      referenceAtom: reference.measure.value,
      atoms: [low, high],
    };
  }
  const discrete = reference.measure.kind === "mass";
  const binWidth = discrete ? Math.ceil((high - low + 1) / 128) : 0;
  const count = discrete ? Math.ceil((high - low + 1) / binWidth) : 0;
  const domain: Range = discrete ? [low - 0.5, low + count * binWidth - 0.5] : [low, high];
  const xs = discrete
    ? Array.from({ length: count }, (_, i) => low + (binWidth - 1) / 2 + i * binWidth)
    : [
        ...new Set([
          ...plotGrid(domain, 96),
          ...plotGrid([0.01, 0.99], 64).map((p) => reference.quantile(p)),
        ]),
      ].toSorted((a, b) => a - b);
  const profile = (law: ConditionalReading) => {
    const height = (x: number) => measureHeight(law.measure, x, binWidth);
    const peak = Math.max(...xs.map((x) => height(x)));
    if (!(peak > 0)) throw new PlotUnavailable("No finite probability in the displayed range.");
    return { height, peak };
  };
  const central = profile(reference);
  const envelope = samples.length > 0 ? samples.map((law) => profile(law)) : [central];
  const commonPeak = Math.max(central.peak, ...envelope.map((p) => p.peak));
  // Density outlines compare widths at equal peak height. Discrete bars share a mass scale.
  const ordinate = (p: Readonly<ReturnType<typeof profile>>, x: number) =>
    p.height(x) / (discrete ? commonPeak : p.peak);
  return {
    domain,
    curve: xs.map((x) => {
      const heights = envelope.map((p) => ordinate(p, x));
      return {
        x,
        value: ordinate(central, x),
        min: Math.min(...heights),
        max: Math.max(...heights),
      };
    }),
    atoms: null,
    referenceAtom: null,
    ...(discrete ? { binWidth } : {}),
  };
}

/** Hold the state at its initial location and the response coefficients at their authored medians. */
export function readingLawPlot(
  indicator: IndicatorSpec,
  model: DynamicalModelSpec,
  draws: ModelDraws,
  args: { readonly spread: readonly Expression[] },
): SpecLawPlot {
  const uses: readonly CoefficientUse[] = coefficientUses(args.spread);
  const shapeOperands = new Set(uses.map((u) => `${u.parameterId}/${u.role}`));
  const time = draws.times[0] ?? 0;
  const center = draws.inputs("median", time, indicator);
  const likelihood = indicator.likelihood;
  const ids = likelihood
    ? expressionStates(observationArguments(likelihood.law).map(([, expr]) => expr))
    : [];
  const at = projectPlot(() => ids.map((id) => [id, center.state(id)] as const));
  const level = (id: ConstructId, v: number) =>
    `${humanize(model.constructs[id]?.name ?? id)} = ${formatSignificant(v)}`;
  const label =
    at.kind === "plot" && at.value.length > 0
      ? `At ${at.value.map(([id, v]) => level(id, v)).join(", ")}`
      : "Conditional reading law";
  return {
    label,
    family: likelihood ? readingFamily(likelihood.law, model, center) : "Observation law",
    uses,
    shape: projectPlot(() => {
      if (!likelihood) throw new PlotUnavailable("No observation law assigned.");
      if (at.kind === "unavailable") throw new PlotUnavailable(at.reason);
      const reference = conditionalReading(likelihood.law, center);
      const samples = Array.from({ length: uses.length > 0 ? draws.count : 0 }, (_, i) => {
        const draw = draws.inputs(i, time, indicator);
        return conditionalReading(likelihood.law, {
          state: center.state,
          parameter: (id, role) =>
            shapeOperands.has(`${id}/${role}`)
              ? draw.parameter(id, role)
              : center.parameter(id, role),
        });
      });
      return readingShape(reference, samples);
    }),
  };
}
