import type { ObservationLawSpec } from "@grassmoth/api-types";
import { jStat } from "jstat";
import {
  type ExpressionInputs,
  evaluateExpression,
  finiteValue,
  PlotUnavailable,
  scalarValue,
  sigmoid,
} from "./expression-evaluation";

function positive(value: number): number {
  if (!(value > 0)) {
    throw new PlotUnavailable("Some input draws leave the observation law's positive domain.");
  }
  return value;
}

function category(probabilities: readonly number[], probability: number): number {
  if (!probabilities.every((mass) => mass >= 0 && mass <= 1)) {
    throw new PlotUnavailable("Some inputs do not define category probabilities.");
  }
  let cumulative = 0;
  for (const [index, mass] of probabilities.entries()) {
    cumulative += mass;
    if (probability < cumulative) return index;
  }
  throw new PlotUnavailable("The category probabilities do not sum to one.");
}

function countQuantile(cdf: (value: number) => number, probability: number): number {
  let low = 0;
  let high = 1;
  while (cdf(high) < probability && high < 1_048_576) high *= 2;
  if (!(cdf(high) >= probability)) {
    throw new PlotUnavailable("This count law exceeds the display sampler's range.");
  }
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (cdf(middle) >= probability) high = middle;
    else low = middle + 1;
  }
  return low;
}

export type ReadingMeasure =
  | { readonly kind: "density"; readonly height: (x: number) => number }
  | { readonly kind: "mass"; readonly lastCategory?: number; readonly cdf: (x: number) => number }
  | { readonly kind: "point"; readonly value: number };

export interface ConditionalReading {
  readonly measure: ReadingMeasure;
  /** Expected reading, in level-index units for a discrete codebook. */
  readonly mean: () => number;
  readonly quantile: (probability: number) => number;
}

function discreteReading(mass: readonly number[]): ConditionalReading {
  return {
    measure: {
      kind: "mass",
      lastCategory: mass.length - 1,
      cdf: (x) => mass.slice(0, Math.max(0, Math.floor(x) + 1)).reduce((a, b) => a + b, 0),
    },
    mean: () => mass.reduce((sum, p, i) => sum + p * i, 0),
    quantile: (p) => category(mass, p),
  };
}

/** Evaluate the authored family once for both its response and its conditional noise. */
export function conditionalReading(
  law: ObservationLawSpec,
  inputs: ExpressionInputs,
): ConditionalReading {
  const evaluate = (expression: Parameters<typeof evaluateExpression>[0]) =>
    evaluateExpression(expression, inputs);
  const scalar = (expression: Parameters<typeof evaluateExpression>[0]) =>
    finiteValue(scalarValue(evaluate(expression)));
  switch (law.distribution) {
    case "Delta": {
      const value = scalar(law.v);
      return { measure: { kind: "point", value }, mean: () => value, quantile: () => value };
    }
    case "Normal": {
      const loc = scalar(law.loc);
      const scale = positive(scalar(law.scale));
      return {
        measure: { kind: "density", height: (x) => jStat.normal.pdf(x, loc, scale) },
        mean: () => loc,
        quantile: (p) => jStat.normal.inv(p, loc, scale),
      };
    }
    case "StudentT": {
      const loc = scalar(law.loc);
      const scale = positive(scalar(law.scale));
      const df = positive(scalar(law.df));
      return {
        measure: {
          kind: "density",
          height: (x) => jStat.studentt.pdf((x - loc) / scale, df) / scale,
        },
        mean: () => {
          if (df <= 1) throw new PlotUnavailable("The StudentT mean is undefined for df ≤ 1.");
          return loc;
        },
        quantile: (p) => loc + scale * jStat.studentt.inv(p, df),
      };
    }
    case "Poisson": {
      const rate = scalar(law.rate);
      if (rate < 0) throw new PlotUnavailable("Some input draws give a negative Poisson rate.");
      const cdf = (k: number) => (k < 0 ? 0 : 1 - jStat.lowRegGamma(k + 1, rate));
      return {
        measure: { kind: "mass", cdf },
        mean: () => rate,
        quantile: (p) => (rate === 0 ? 0 : countQuantile(cdf, p)),
      };
    }
    case "Gamma": {
      const shape = positive(scalar(law.concentration));
      const rate = positive(scalar(law.rate));
      return {
        measure: { kind: "density", height: (x) => jStat.gamma.pdf(x, shape, 1 / rate) },
        mean: () => shape / rate,
        quantile: (p) => jStat.gamma.inv(p, shape, 1 / rate),
      };
    }
    case "Beta": {
      const a = positive(scalar(law.concentration1));
      const b = positive(scalar(law.concentration0));
      return {
        measure: { kind: "density", height: (x) => jStat.beta.pdf(x, a, b) },
        mean: () => a / (a + b),
        quantile: (p) => jStat.beta.inv(p, a, b),
      };
    }
    case "BernoulliLogits":
    case "BernoulliProbs": {
      const p =
        law.distribution === "BernoulliLogits" ? sigmoid(scalar(law.logits)) : scalar(law.probs);
      return discreteReading([1 - p, p]);
    }
    case "NegativeBinomial2": {
      const size = positive(scalar(law.concentration));
      const mean = positive(scalar(law.mean));
      const prob = size / (size + mean);
      const cdf = (k: number) => (k < 0 ? 0 : jStat.ibeta(prob, size, k + 1));
      return {
        measure: { kind: "mass", cdf },
        mean: () => mean,
        quantile: (p) => countQuantile(cdf, p),
      };
    }
    case "OrderedLogistic": {
      const cutpoints = evaluate(law.cutpoints);
      const values = typeof cutpoints === "number" ? [cutpoints] : cutpoints;
      const cumulative = [0, ...values.map((value) => sigmoid(value - scalar(law.predictor))), 1];
      return discreteReading(
        cumulative.slice(1).map((value, index) => value - (cumulative[index] ?? 0)),
      );
    }
    case "Categorical": {
      const logits = evaluate(law.logits);
      if (typeof logits === "number") {
        throw new PlotUnavailable("Category logits require named category coordinates.");
      }
      const peak = Math.max(...logits);
      const mass = logits.map((value) => Math.exp(value - peak));
      const total = mass.reduce((sum, value) => sum + value, 0);
      return discreteReading(mass.map((value) => value / total));
    }
  }
}

/** A single display draw on the same conditional law as the reading glyph. */
export function measurementDraw(
  law: ObservationLawSpec,
  inputs: ExpressionInputs,
  probability: number,
): number {
  return conditionalReading(law, inputs).quantile(probability);
}
