import type {
  ConstructId,
  ConstructSpec,
  CausalEdgeSpec,
  IndicatorId,
  IndicatorSpec,
  ModelSnapshot,
} from "@grassmoth/api-types";
import type { ScopeContext } from "./scope";
import { recordedEntityFailures } from "./entity-findings";

/** These selectors only arrange recorded findings for their owning entities. */
export function constructPresentation(context: ScopeContext, id: ConstructId) {
  const { modelSnapshot, entities } = context;
  const construct = entities.constructById.get(id);
  if (!construct) return null;
  const indicators = construct.indicators;

  return {
    modelSnapshot,
    entities,
    construct,
    indicators,
  };
}

export function indicatorPresentation(context: ScopeContext, id: IndicatorId) {
  const indicator = context.entities.indicatorById.get(id);
  if (!indicator) return null;
  const profile = context.modelSnapshot.profile?.indicators[id];
  const compatibility = context.modelSnapshot.fit_checks?.data.indicators[id];
  const audit =
    profile || compatibility
      ? {
          profile: profile?.profile ?? null,
          findings: [...(profile?.findings ?? []), ...(compatibility?.findings ?? [])],
        }
      : null;
  const likelihood = indicator.likelihood;
  const findings =
    audit?.findings.filter(
      (finding) => finding.kind === "not_evaluated" || finding.outcome === "failed",
    ) ?? [];
  return {
    indicator,
    audit,
    likelihood,
    findings,
  };
}

/** Aggregate recorded failures at their visible owner; never judge diagnostic numbers here. */
export function entityFailures(
  modelSnapshot: ModelSnapshot,
  entity: ConstructSpec | CausalEdgeSpec | IndicatorSpec,
): string[] {
  const identity = "observation" in entity ? entity.observation.id : entity.id;
  return [
    ...(recordedEntityFailures(modelSnapshot)[identity] ?? []),
    ...("dynamics" in entity &&
    modelSnapshot.identification?.treatments[entity.id]?.status === "not_identified"
      ? [`Identification against ★: ${entity.name}`]
      : []),
  ];
}
