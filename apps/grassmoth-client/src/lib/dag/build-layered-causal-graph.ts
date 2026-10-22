import type { CausalEdgeSpec, ConstructSpec, ConstructId, EdgeId } from "@grassmoth/api-types";
import type { DagGraphInput } from "@/lib/utils/dag-graph-layout";
import { ghostId, isGhost } from "@/lib/dag/unroll";

export const LAYERED_NODE_WIDTH = 250;
export const LAYERED_NODE_HEIGHT = 158;
export const LAYERED_HISTORY_WIDTH = 156;
export const LAYERED_HISTORY_HEIGHT = 54;
export const LAYERED_EDGE_SLOT_WIDTH = 96;
export const LAYERED_EDGE_SLOT_HEIGHT = 40;

export type LayeredGraphNodeMeta =
  | { kind: "construct"; construct: ConstructSpec }
  | { kind: "history"; construct: ConstructSpec }
  | { kind: "edge_slot"; edge: LayeredGraphEdgeMeta };

export type LayeredGraphEdgeMeta = {
  cause: ConstructSpec;
  effect: ConstructSpec;
  source: string;
  target: string;
  crossSlice: boolean;
  slotId: string;
} & ({ isSelf: false; id: EdgeId } | { isSelf: true; id: `self:${ConstructId}` });

export interface LayeredGraphSegmentMeta {
  edgeId: string;
  markerEnd: boolean;
}

export interface LayeredGraphBundle {
  graph: DagGraphInput;
  nodeMeta: Map<string, LayeredGraphNodeMeta>;
  edgeMeta: Map<string, LayeredGraphEdgeMeta>;
  segmentMeta: Map<string, LayeredGraphSegmentMeta>;
}

const partition = (value: 0 | 1 | 2): Record<string, string> => ({
  "elk.partitioning.partition": String(value),
});

/**
 * Lay out the backend-selected constructs and edges for a committed checkpoint.
 * Comparisons decorate this layout without moving its existing nodes.
 */
export function buildLayeredCausalGraph(
  constructs: readonly ConstructSpec[],
  edges: readonly CausalEdgeSpec[],
  dynamicConstructIds: readonly ConstructId[],
): LayeredGraphBundle {
  const constructById = new Map(constructs.map((construct) => [construct.id, construct] as const));
  const timeVaryingIds = new Set(dynamicConstructIds);
  const constructPartition = (construct: ConstructSpec): 0 | 2 =>
    timeVaryingIds.has(construct.id) ? 2 : 0;
  const selfDynamicConstructs = constructs.filter((construct) => timeVaryingIds.has(construct.id));
  const causalLinks = edges
    .filter((edge) => edge.cause.id !== edge.effect.id)
    .map((edge) => ({
      ...edge,
      source: timeVaryingIds.has(edge.cause.id) ? ghostId(edge.cause.id) : edge.cause.id,
      target: edge.effect.id,
    }));
  const ghosts = new Set([
    ...causalLinks.filter((edge) => isGhost(edge.source)).map((edge) => edge.source),
    ...selfDynamicConstructs.map((construct) => ghostId(construct.id)),
  ]);

  const edgeDefinitions = [
    ...causalLinks.map((edge) => ({
      id: edge.id,
      cause: edge.cause.id,
      effect: edge.effect.id,
      source: edge.source,
      target: edge.target,
      crossSlice: isGhost(edge.source),
      isSelf: false as const,
    })),
    ...selfDynamicConstructs.map((construct) => ({
      id: `self:${construct.id}` as const,
      cause: construct.id,
      effect: construct.id,
      source: ghostId(construct.id),
      target: construct.id,
      crossSlice: true,
      isSelf: true as const,
    })),
  ];

  const nodeMeta = new Map<string, LayeredGraphNodeMeta>();
  const nodes: DagGraphInput["nodes"] = [];
  for (const construct of constructs) {
    nodeMeta.set(construct.id, { kind: "construct", construct });
    nodes.push({
      id: construct.id,
      width: LAYERED_NODE_WIDTH,
      height: LAYERED_NODE_HEIGHT,
      layoutOptions: partition(constructPartition(construct)),
    });
  }
  for (const construct of constructs.filter((construct) => ghosts.has(ghostId(construct.id)))) {
    const ghost = ghostId(construct.id);
    nodeMeta.set(ghost, { kind: "history", construct });
    nodes.push({
      id: ghost,
      width: LAYERED_HISTORY_WIDTH,
      height: LAYERED_HISTORY_HEIGHT,
      layoutOptions: partition(1),
    });
  }

  const edgeMeta = new Map<string, LayeredGraphEdgeMeta>();
  const segmentMeta = new Map<string, LayeredGraphSegmentMeta>();
  const segments: DagGraphInput["edges"] = [];
  edgeDefinitions.forEach((definition, index) => {
    const slot = {
      id: `G__${index}`,
      width: LAYERED_EDGE_SLOT_WIDTH,
      height: LAYERED_EDGE_SLOT_HEIGHT,
    };
    const sourceConstruct = constructById.get(definition.cause);
    const targetConstruct = constructById.get(definition.effect);
    if (!sourceConstruct || !targetConstruct) {
      throw new Error(`Edge slot '${slot.id}' has an unknown endpoint.`);
    }
    const sourcePartition = definition.source.endsWith("__p")
      ? 1
      : constructPartition(sourceConstruct);
    const targetPartition = constructPartition(targetConstruct);
    const slotPartition = sourcePartition === 0 && targetPartition === 2 ? 1 : sourcePartition;

    nodes.push({
      ...slot,
      layoutOptions: partition(slotPartition),
    });
    const edge: LayeredGraphEdgeMeta = {
      ...definition,
      cause: sourceConstruct,
      effect: targetConstruct,
      slotId: slot.id,
    };
    nodeMeta.set(slot.id, { kind: "edge_slot", edge });
    edgeMeta.set(edge.id, edge);
    segmentMeta.set(`e${index}s`, { edgeId: definition.id, markerEnd: false });
    segmentMeta.set(`e${index}t`, { edgeId: definition.id, markerEnd: true });
    segments.push({ id: `e${index}s`, source: definition.source, target: slot.id });
    segments.push({ id: `e${index}t`, source: slot.id, target: definition.target });
  });

  return {
    graph: {
      nodes,
      edges: segments,
      direction: "RIGHT",
      layoutOptions: {
        "elk.partitioning.activate": "true",
        "elk.layered.spacing.nodeNodeBetweenLayers": "54",
        "elk.spacing.nodeNode": "30",
        "elk.spacing.edgeNode": "24",
        "elk.spacing.edgeEdge": "14",
        "elk.layered.spacing.edgeNodeBetweenLayers": "26",
      },
    },
    nodeMeta,
    edgeMeta,
    segmentMeta,
  };
}
