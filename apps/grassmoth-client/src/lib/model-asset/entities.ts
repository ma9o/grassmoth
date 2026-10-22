import { modelConstructs, modelEdges, modelParameters } from "@/lib/model-accessors";
import type { ConstructSpec, DynamicalModelSpec, ParameterId } from "@grassmoth/api-types";
import { humanize, type EntitySelection } from "./selection";
import { ownLawUses } from "./laws";

/** Index the authored graph; these maps are derived, never a second model definition. */
export function indexModel(dynamicalModelSpec: DynamicalModelSpec | null | undefined) {
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

export type ModelEntities = ReturnType<typeof indexModel>;

export interface EntityLink {
  selection: EntitySelection;
  label: string;
}

export interface EntityPresentation extends EntityLink {
  relationships: EntityLink[];
}

/** Resolve ownership, labels and fields regardless of where endpoints serialize. */
export function resolveEntity(
  entities: ModelEntities,
  selection: EntitySelection,
): EntityPresentation | undefined {
  const constructLink = (construct: ConstructSpec): EntityLink => ({
    selection: { kind: "construct", id: construct.id },
    label: humanize(construct.name),
  });
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
