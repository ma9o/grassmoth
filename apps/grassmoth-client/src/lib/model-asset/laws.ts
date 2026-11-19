import type {
  CausalEdgeSpec,
  ConstructSpec,
  DensityCurve,
  DynamicalModelSpec,
  IndicatorSpec,
  ModelSnapshot,
  NumPyroDistribution,
  ParameterSpec,
  PosteriorMarginal,
} from "@grassmoth/api-types";
import { type CoefficientUse, coefficientUses, modelParameters } from "@/lib/model-accessors";
import { authoredPriorPlot } from "./authored-prior-plot";
import { nativeArray } from "./law-draws";
import { observationArguments } from "./observation-law";
import { humanize } from "./selection";

/** One parameter's recorded laws at the viewed version, arranged on a single axis. */
export interface LawCurve {
  readonly use: CoefficientUse;
  readonly parameter: ParameterSpec;
  /**
   * `fitted`: the fit's conditioned input law beside its posteriors, on the quantity scale.
   * `authored`: the law as written, on its authoring scale.
   */
  readonly kind: "fitted" | "authored";
  readonly prior: DensityCurve;
  readonly posteriors: readonly PosteriorMarginal[];
  /** The authored law's family; a fitted law is known by its posteriors. */
  readonly family: string | null;
}

/** Laws named by an entity's own terms: construct dynamics and noise, edge mechanisms, or measurement. */
export function ownLawUses(entity: ConstructSpec | CausalEdgeSpec | IndicatorSpec) {
  if ("dynamics" in entity) {
    return coefficientUses([...entity.dynamics.map((d) => d.expression), ...entity.coefficients]);
  }
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
    if (fit && posteriors.length > 0) {
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
    }
    const law = parameter.distribution
      ? modelSnapshot.dynamical_model_spec?.distributions[parameter.distribution]
      : null;
    // A point mass or a member of a joint law has no standalone authored scalar PDF.
    if (
      law?.distribution === "Delta" ||
      (parameter.distribution &&
        modelSnapshot.dynamical_model_spec?.law_layouts[parameter.distribution])
    ) {
      return [];
    }
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

/** The interval over which a persistence or interval-effect law is authored. */
export function intervalText(days: number | "model_clock"): string {
  return days === "model_clock" ? "model clock tick" : `${days} d`;
}

/** Authored persistence and interval-effect laws describe another quantity than their role. */
export function lawLabel(curve: LawCurve): string {
  if (curve.kind === "authored") {
    if (curve.parameter.transform.kind === "dt_persistence_to_ct_decay") return "persistence";
    if (curve.parameter.transform.kind === "dt_effect_to_ct_rate") return "interval effect";
  }
  return humanize(curve.use.role);
}

/** How a parameter's law is held: on its own, as a member of a joint law, or not yet. */
export type ParameterLaw =
  | { readonly kind: "scalar"; readonly law: NumPyroDistribution }
  | { readonly kind: "joint"; readonly law: NumPyroDistribution }
  | { readonly kind: "unassigned" };

/** A construct's trajectory law: the values an exogenous path holds from each time, or a law. */
export type StateLaw =
  | {
      readonly kind: "path";
      readonly points: readonly { readonly time: number; readonly value: number }[];
      readonly timeOrigin: string;
    }
  | { readonly kind: "law"; readonly law: NumPyroDistribution };

export function parameterLaw(model: DynamicalModelSpec, parameter: ParameterSpec): ParameterLaw {
  const law = parameter.distribution ? model.distributions[parameter.distribution] : undefined;
  if (!parameter.distribution || !law) return { kind: "unassigned" };
  return { kind: model.law_layouts[parameter.distribution] ? "joint" : "scalar", law };
}

export function stateLaw(model: DynamicalModelSpec, construct: ConstructSpec): StateLaw | null {
  const law = construct.distribution ? model.distributions[construct.distribution] : undefined;
  if (!construct.distribution || !law) return null;
  const layout = model.law_layouts[construct.distribution];
  const member = layout?.constructs.indexOf(construct.id) ?? -1;
  if (construct.role !== "exogenous" || law.distribution !== "Delta" || !layout || member === -1) {
    return { kind: "law", law };
  }
  // Trajectories follow the law's parameters, one run of time points per construct.
  const count = layout.time_points.length;
  const start =
    layout.parameters.reduce((sum, [, elements]) => sum + elements.length, 0) + member * count;
  const values = nativeArray(law.params.v).values.slice(start, start + count);
  const points = layout.time_points.flatMap((time, index) => {
    const value = values[index];
    return value === undefined ? [] : [{ time, value }];
  });
  return points.length === count
    ? { kind: "path", points, timeOrigin: layout.time_origin }
    : { kind: "law", law };
}
