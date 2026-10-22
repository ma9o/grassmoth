import type { ObservationLawSpec } from "@grassmoth/api-types";
import { jStat } from "jstat";
import { assertNever } from "@/lib/assert-never";
import {
  evaluateExpression,
  type ExpressionInputs,
  finiteValue,
  PlotUnavailable,
  scalarValue,
  sigmoid,
} from "./expression-evaluation";

function positive(value: number): number {
  if (!(value > 0))
    throw new PlotUnavailable("Some input draws leave the observation law's positive domain.");
  return value;
}

function category(probabilities: readonly number[], probability: number): number {
  if (!probabilities.every((mass) => mass >= 0 && mass <= 1))
    throw new PlotUnavailable("Some inputs do not define category probabilities.");
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
  while (cdf(high) < probability && high < 1048576) high *= 2;
  if (!(cdf(high) >= probability))
    throw new PlotUnavailable("This count law exceeds the display sampler's range.");
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (cdf(middle) >= probability) high = middle;
    else low = middle + 1;
  }
  return low;
}

/** Draw from the local conditional reading law; windowed predictive histories remain saved results. */
export function measurementDraw(
  law: ObservationLawSpec,
  inputs: ExpressionInputs,
  probability: number,
): number {
  const evaluate = (expression: Parameters<typeof evaluateExpression>[0]) =>
    evaluateExpression(expression, inputs);
  const scalar = (expression: Parameters<typeof evaluateExpression>[0]) =>
    finiteValue(scalarValue(evaluate(expression)));
  switch (law.distribution) {
    case "Delta":
      return scalar(law.v);
    case "Normal":
      return jStat.normal.inv(probability, scalar(law.loc), positive(scalar(law.scale)));
    case "StudentT":
      return (
        scalar(law.loc) +
        positive(scalar(law.scale)) * jStat.studentt.inv(probability, positive(scalar(law.df)))
      );
    case "Poisson": {
      const rate = scalar(law.rate);
      if (rate < 0) throw new PlotUnavailable("Some input draws give a negative Poisson rate.");
      return rate === 0
        ? 0
        : countQuantile((value) => 1 - jStat.lowRegGamma(value + 1, rate), probability);
    }
    case "Gamma":
      return jStat.gamma.inv(
        probability,
        positive(scalar(law.concentration)),
        1 / positive(scalar(law.rate)),
      );
    case "Beta":
      return jStat.beta.inv(
        probability,
        positive(scalar(law.concentration1)),
        positive(scalar(law.concentration0)),
      );
    case "BernoulliLogits":
      return category([1 - sigmoid(scalar(law.logits)), sigmoid(scalar(law.logits))], probability);
    case "BernoulliProbs": {
      const p = scalar(law.probs);
      return category([1 - p, p], probability);
    }
    case "NegativeBinomial2": {
      const size = positive(scalar(law.concentration));
      const p = size / (size + positive(scalar(law.mean)));
      return countQuantile((value) => jStat.ibeta(p, size, value + 1), probability);
    }
    case "OrderedLogistic": {
      const cutpoints = evaluate(law.cutpoints);
      const values = typeof cutpoints === "number" ? [cutpoints] : cutpoints;
      const cumulative = [0, ...values.map((value) => sigmoid(value - scalar(law.predictor))), 1];
      return category(
        cumulative.slice(1).map((value, index) => value - (cumulative[index] ?? 0)),
        probability,
      );
    }
    case "Categorical": {
      const logits = evaluate(law.logits);
      if (typeof logits === "number")
        throw new PlotUnavailable("Category logits require named category coordinates.");
      const peak = Math.max(...logits);
      const mass = logits.map((value) => Math.exp(value - peak));
      const sum = mass.reduce((sum, value) => sum + value, 0);
      return category(
        mass.map((value) => value / sum),
        probability,
      );
    }
    default:
      return assertNever(law);
  }
}
