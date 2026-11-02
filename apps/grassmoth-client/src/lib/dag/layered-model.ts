import type { ConstructId, ModelSnapshot, SimulationReport } from "@grassmoth/api-types";
import { modelConstructs, presentEntries } from "@/lib/model-accessors";
import type { ConstructStatus } from "./construct-statuses";
import type { ModelEntities } from "@/lib/model-asset/entities";

export const CAUSAL_GRAPH_LAYER_ORDER = [
  "structure",
  "measurement",
  "design",
  "specification",
  "fit",
  "simulation",
] as const;
export type CausalGraphLayerId = (typeof CAUSAL_GRAPH_LAYER_ORDER)[number];

/** Display every construct and directed edge in the saved scientific model. */
export function graphEntities(indexed: ModelEntities) {
  return {
    constructs: indexed.constructs,
    dynamicConstructIds: indexed.constructs
      .filter((construct) => construct.temporal_status === "time_varying")
      .map((construct) => construct.id),
    edges: indexed.edges,
    indicators: indexed.indicators,
  };
}

/** Overlay saved identification findings on explicitly observed or latent constructs. */
export function graphStatus(modelSnapshot: ModelSnapshot, id: ConstructId): ConstructStatus | null {
  const construct = modelConstructs(modelSnapshot.dynamical_model_spec).find(
    (item) => item.id === id,
  );
  if (!construct) return null;
  const blocking = presentEntries(modelSnapshot.identification?.treatments ?? {}).some(
    ([target, finding]) =>
      finding.status === "not_identified" && (target === id || finding.confounders.includes(id)),
  );
  return blocking
    ? "blocking"
    : construct.indicators.length > 0 || construct.role === "exogenous"
      ? "observed"
      : "latent";
}

/** Layer visibility reflects facts in the selected revision, including partial models. */
export function availableGraphLayers(
  modelSnapshot: ModelSnapshot,
  simulation?: SimulationReport | null,
): CausalGraphLayerId[] {
  const available = {
    structure: modelConstructs(modelSnapshot.dynamical_model_spec).length > 0,
    measurement: modelConstructs(modelSnapshot.dynamical_model_spec).some(
      (construct) => construct.indicators.length > 0,
    ),
    design: modelSnapshot.identification !== null,
    specification: modelConstructs(modelSnapshot.dynamical_model_spec).some(
      (c) => c.dynamics.length > 0 || c.indicators.some((i) => i.likelihood != null),
    ),
    fit: modelSnapshot.fit !== null,
    simulation: simulation != null,
  };
  return CAUSAL_GRAPH_LAYER_ORDER.filter((layer) => available[layer]);
}
