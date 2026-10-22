import { modelParameters } from "@/lib/model-accessors";
import type {
  CausalEdgeSpec,
  ConstructSpec,
  DensityCurve,
  IndicatorSpec,
  ModelSnapshot,
  ParameterSpec,
  PosteriorMarginal,
} from "@grassmoth/api-types";
import { type CoefficientUse, coefficientUses } from "@/lib/model-accessors";
import { observationArguments } from "./observation-law";
import { authoredPriorPlot } from "./authored-prior-plot";
import { humanize } from "./selection";

/** One parameter's recorded laws at the viewed version, arranged on a single axis. */
export interface LawCurve {
  use: CoefficientUse;
  parameter: ParameterSpec;
  /**
   * `fitted`: the fit's conditioned input law beside its posteriors, on the quantity scale.
   * `authored`: the law as written, on its authoring scale.
   */
  kind: "fitted" | "authored";
  prior: DensityCurve;
  posteriors: readonly PosteriorMarginal[];
  /** The authored law's family; a fitted law is known by its posteriors. */
  family: string | null;
}

/** Laws named by an entity's own terms: construct dynamics and noise, edge mechanisms, or measurement. */
export function ownLawUses(entity: ConstructSpec | CausalEdgeSpec | IndicatorSpec) {
  if ("dynamics" in entity)
    return coefficientUses([...entity.dynamics.map((d) => d.expression), ...entity.coefficients]);
  if ("mechanisms" in entity) return coefficientUses(entity.mechanisms.map((m) => m.expression));
  return coefficientUses(
    entity.likelihood ? observationArguments(entity.likelihood.law).map(([, value]) => value) : [],
  );
}

/** A fit's posteriors supersede the laws authored at its version. */
export function lawCurves(
  modelSnapshot: ModelSnapshot,
  uses: readonly CoefficientUse[],
): LawCurve[] {
  const parameters = new Map(
    modelParameters(modelSnapshot.dynamical_model_spec).map((parameter) => [
      parameter.id,
      parameter,
    ]),
  );
  const fit = modelSnapshot.fit;
  return uses.flatMap((use): LawCurve[] => {
    const parameter = parameters.get(use.parameterId);
    if (!parameter) return [];
    const posteriors = (fit?.posterior_marginals ?? []).filter(
      (marginal) => marginal.subject.parameter_id === use.parameterId,
    );
    if (fit && posteriors.length > 0)
      return [
        {
          use,
          parameter,
          kind: "fitted",
          prior: fit.prior_densities[use.parameterId] ?? { x: [], density: [] },
          posteriors,
          family: null,
        },
      ];
    const law = parameter.distribution
      ? modelSnapshot.dynamical_model_spec?.distributions[parameter.distribution]
      : null;
    // A point mass or a member of a joint law has no standalone authored scalar PDF.
    if (
      law?.distribution === "Delta" ||
      (parameter.distribution &&
        modelSnapshot.dynamical_model_spec?.law_layouts[parameter.distribution])
    )
      return [];
    return law
      ? [
          {
            use,
            parameter,
            kind: "authored",
            prior: authoredPriorPlot(law),
            posteriors: [],
            family: law.distribution,
          },
        ]
      : [];
  });
}

/** Authored persistence and interval-effect laws describe another quantity than their role. */
export function lawLabel(curve: LawCurve): string {
  if (curve.kind === "authored") {
    if (curve.parameter.transform.kind === "dt_persistence_to_ct_decay") return "persistence";
    if (curve.parameter.transform.kind === "dt_effect_to_ct_rate") return "interval effect";
  }
  return humanize(curve.use.role);
}
