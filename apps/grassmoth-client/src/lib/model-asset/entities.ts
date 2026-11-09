import type {
  CausalEdgeSpec,
  ConstructId,
  ConstructSpec,
  DynamicalModelSpec,
  EdgeId,
  IndicatorId,
  IndicatorSpec,
  ParameterId,
  ParameterSpec,
} from "@grassmoth/api-types";
import { modelConstructs, modelEdges, modelParameters } from "@/lib/model-accessors";
import { ownLawUses } from "./laws";
import { type EntitySelection, humanize } from "./selection";

/** Index the authored graph; these maps are derived, never a second model definition. */
export function indexModel(
  dynamicalModelSpec: DynamicalModelSpec | null | undefined,
): ModelEntities {
  const constructs = modelConstructs(dynamicalModelSpec);
  const edges = modelEdges(dynamicalModelSpec);
  const indicators = constructs.flatMap((construct) => construct.indicators);
  const parameters = modelParameters(dynamicalModelSpec);
  return {
    constructs,
    edges,
    indicators,
    parameters,
    constructById: new Map(constructs.map((entity) => [entity.id, entity])),
    edgeById: new Map(edges.map((entity) => [entity.id, entity])),
    indicatorOwnerById: new Map(
      constructs.flatMap((construct) =>
        construct.indicators.map((indicator) => [indicator.observation.id, construct] as const),
      ),
    ),
    indicatorById: new Map(indicators.map((entity) => [entity.observation.id, entity])),
    parameterById: new Map(parameters.map((entity) => [entity.id, entity])),
  };
}

export interface ModelEntities {
  readonly constructs: readonly ConstructSpec[];
  readonly edges: readonly CausalEdgeSpec[];
  readonly indicators: readonly IndicatorSpec[];
  readonly parameters: readonly ParameterSpec[];
  readonly constructById: ReadonlyMap<ConstructId, ConstructSpec>;
  readonly edgeById: ReadonlyMap<EdgeId, CausalEdgeSpec>;
  readonly indicatorOwnerById: ReadonlyMap<IndicatorId, ConstructSpec>;
  readonly indicatorById: ReadonlyMap<IndicatorId, IndicatorSpec>;
  readonly parameterById: ReadonlyMap<ParameterId, ParameterSpec>;
}

export interface EntityLink {
  readonly selection: EntitySelection;
  readonly label: string;
}

export interface EntityPresentation extends EntityLink {
  readonly relationships: readonly EntityLink[];
}

function constructLink(construct: ConstructSpec): EntityLink {
  return {
    selection: { kind: "construct", id: construct.id },
    label: humanize(construct.name),
  };
}

/** Resolve ownership, labels and fields regardless of where endpoints serialize. */
export function resolveEntity(
  entities: ModelEntities,
  selection: EntitySelection,
): EntityPresentation | undefined {
  switch (selection.kind) {
    case "edge": {
      const edge = entities.edgeById.get(selection.id);
      if (!edge) return undefined;
      const cause = entities.constructById.get(edge.cause.id);
      const effect = entities.constructById.get(edge.effect.id);
      if (!cause || !effect) return undefined;
      const relationships = [constructLink(cause), constructLink(effect)];
      return {
        selection,
        label: relationships.map((link) => link.label).join(" → "),
        relationships,
      };
    }
    case "indicator": {
      const indicator = entities.indicatorById.get(selection.id);
      const owner = entities.indicatorOwnerById.get(selection.id);
      return indicator && owner
        ? {
            selection,
            label: humanize(indicator.observation.name),
            relationships: [constructLink(owner)],
          }
        : undefined;
    }
    case "construct": {
      const entity = entities.constructById.get(selection.id);
      return entity ? { selection, label: humanize(entity.name), relationships: [] } : undefined;
    }
  }
}

/** A parameter link opens the entity whose own law section displays it. */
export function parameterOwner(entities: ModelEntities, id: ParameterId) {
  const owners = [
    ...entities.edges.map((entity) => ({
      entity,
      selection: { kind: "edge", id: entity.id } as const,
    })),
    ...entities.indicators.map((entity) => ({
      entity,
      selection: { kind: "indicator", id: entity.observation.id } as const,
    })),
    ...entities.constructs.map((entity) => ({
      entity,
      selection: { kind: "construct", id: entity.id } as const,
    })),
  ];
  const owner = owners.find(({ entity }) =>
    ownLawUses(entity).some((use) => use.parameterId === id),
  );
  return owner ? resolveEntity(entities, owner.selection) : undefined;
}
