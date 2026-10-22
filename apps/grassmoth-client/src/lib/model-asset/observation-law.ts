import type { Expression, ObservationLawSpec } from "@grassmoth/api-types";
import { assertNever } from "@/lib/assert-never";

/** Named operands of each authored observation law, in its declared order. */
export function observationArguments(
  law: ObservationLawSpec,
): readonly (readonly [string, Expression])[] {
  switch (law.distribution) {
    case "Delta":
      return [["v", law.v]];
    case "Normal":
      return [
        ["loc", law.loc],
        ["scale", law.scale],
      ];
    case "StudentT":
      return [
        ["df", law.df],
        ["loc", law.loc],
        ["scale", law.scale],
      ];
    case "Poisson":
      return [["rate", law.rate]];
    case "Gamma":
      return [
        ["concentration", law.concentration],
        ["rate", law.rate],
      ];
    case "BernoulliLogits":
      return [["logits", law.logits]];
    case "BernoulliProbs":
      return [["probs", law.probs]];
    case "NegativeBinomial2":
      return [
        ["mean", law.mean],
        ["concentration", law.concentration],
      ];
    case "Beta":
      return [
        ["concentration1", law.concentration1],
        ["concentration0", law.concentration0],
      ];
    case "OrderedLogistic":
      return [
        ["predictor", law.predictor],
        ["cutpoints", law.cutpoints],
      ];
    case "Categorical":
      return [["logits", law.logits]];
    default:
      return assertNever(law);
  }
}
