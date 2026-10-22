import type {
  ConstructId,
  EdgeId,
  IndicatorId,
  ModelSnapshot,
  ParameterId,
} from "@grassmoth/api-types";
import { indexModel } from "./entities";
import { ownLawUses } from "./laws";

const messages = new WeakMap<
  ModelSnapshot,
  Readonly<Partial<Record<ConstructId | EdgeId | IndicatorId, readonly string[]>>>
>();

/** Format retained verdicts by the model's declared coefficient references. */
export function recordedEntityFailures(
  modelSnapshot: ModelSnapshot,
): Readonly<Partial<Record<ConstructId | EdgeId | IndicatorId, readonly string[]>>> {
  const cached = messages.get(modelSnapshot);
  if (cached) return cached;
  const fit = modelSnapshot.fit;
  const entities = indexModel(modelSnapshot.dynamical_model_spec);
  const convergence = (parameters: readonly ParameterId[]) =>
    (fit?.convergence.findings ?? []).flatMap((finding) =>
      finding.kind === "evaluated" &&
      finding.outcome === "failed" &&
      typeof finding.subject !== "string" &&
      "parameter" in finding.subject &&
      parameters.includes(finding.subject.parameter.parameter_id)
        ? [`Parameter convergence: ${finding.subject.label}`]
        : [],
    );
  const indicators = Object.fromEntries(
    entities.indicators.map((indicator) => {
      const issues = [
        ...(modelSnapshot.profile?.indicators[indicator.observation.id]?.findings ?? []),
        ...(modelSnapshot.fit_checks?.data.indicators[indicator.observation.id]?.findings ?? []),
      ];
      return [
        indicator.observation.id,
        [
          ...convergence(ownLawUses(indicator).map((use) => use.parameterId)),
          ...(issues.some(
            (finding) => finding.kind === "not_evaluated" || finding.outcome === "failed",
          )
            ? [`Data quality: ${indicator.observation.name}`]
            : []),
        ],
      ];
    }),
  );
  const result = {
    ...indicators,
    ...Object.fromEntries(
      entities.edges.map((edge) => [
        edge.id,
        [...new Set(convergence(ownLawUses(edge).map((use) => use.parameterId)))],
      ]),
    ),
    ...Object.fromEntries(
      entities.constructs.map((construct) => {
        const uses = [
          ...ownLawUses(construct),
          ...entities.edges
            .filter((edge) => edge.cause.id === construct.id || edge.effect.id === construct.id)
            .flatMap(ownLawUses),
        ];
        return [
          construct.id,
          [
            ...new Set([
              ...convergence(uses.map((use) => use.parameterId)),
              ...(fit?.convergence.findings ?? []).flatMap((finding) =>
                typeof finding.subject !== "string" &&
                "state" in finding.subject &&
                finding.subject.state.id === construct.id &&
                (finding.kind === "not_evaluated" || finding.outcome === "failed")
                  ? [`Latent convergence: ${finding.subject.label}`]
                  : [],
              ),
              ...construct.indicators.flatMap(
                (indicator) => indicators[indicator.observation.id] ?? [],
              ),
            ]),
          ],
        ];
      }),
    ),
  };
  messages.set(modelSnapshot, result);
  return result;
}
